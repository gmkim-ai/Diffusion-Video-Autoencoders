import copy
import json
import os
import re

import numpy as np
import pytorch_lightning as pl
import torch
from numpy.lib.function_base import flip
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import *
from torch import nn
from torch.cuda import amp
from torch.distributions import Categorical
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import ConcatDataset, TensorDataset
from torchvision.utils import make_grid, save_image

from config import *
from dataset import *
from dist_utils import *
from lmdb_writer import *
from metrics import *
from renderer import *


class LitModel(pl.LightningModule):
    def __init__(self, conf: TrainConfig):
        super().__init__()
        assert conf.train_mode != TrainMode.manipulate
        if conf.seed is not None:
            pl.seed_everything(conf.seed)

        self.save_hyperparameters(conf.as_dict_jsonable())

        self.conf = conf

        self.model = conf.make_model_conf().make_model()
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        model_size = 0
        for param in self.model.parameters():
            model_size += param.data.nelement()
        print('Model params: %.2f M' % (model_size / 1024 / 1024))

        self.sampler = conf.make_diffusion_conf().make_sampler()
        self.eval_sampler = conf.make_eval_diffusion_conf().make_sampler()

        # this is shared for both model and latent
        self.T_sampler = conf.make_T_sampler()

        self.latent_sampler = None
        self.eval_latent_sampler = None
        self.last_batch_imgs = None
        self.last_real = None
        self.decom_batch_imgs = None
        self.decom_batch_imgs2 = None
        self.other_batch_imgs = None
        self.cond_batch_imgs = None
        self.x0_estimated = None
        self.x0_estimated2 = None

        if conf.pretrain is not None:
            print(f'loading pretrain ... {conf.pretrain.name}')
            state = torch.load(conf.pretrain.path, map_location='cpu')
            print('step:', state['global_step'])
            self.load_state_dict(state['state_dict'], strict=False)

        if conf.latent_infer_path is not None:
            print('loading latent stats ...')
            state = torch.load(conf.latent_infer_path)
            self.conds = state['conds']
            self.register_buffer('conds_mean', state['conds_mean'][None, :])
            self.register_buffer('conds_std', state['conds_std'][None, :])
        else:
            self.conds_mean = None
            self.conds_std = None

    def load_model(self, path, strict=True):
        self.load_state_dict(torch.load(path, map_location=self.device)['state_dict'], strict=strict)
    
    def normalize(self, cond):
        cond = (cond - self.conds_mean.to(self.device)) / self.conds_std.to(
            self.device)
        return cond

    def denormalize(self, cond):
        cond = (cond * self.conds_std.to(self.device)) + self.conds_mean.to(
            self.device)
        return cond

    def sample(self, N, device, T=None, T_latent=None):
        if T is None:
            sampler = self.eval_sampler
            latent_sampler = self.latent_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()
            latent_sampler = self.conf._make_latent_diffusion_conf(T_latent).make_sampler()

        noise = torch.randn(N,
                            3,
                            self.conf.img_size,
                            self.conf.img_size,
                            device=device)
        pred_img = render_uncondition(
            self.conf,
            self.ema_model,
            noise,
            sampler=sampler,
            latent_sampler=latent_sampler,
            conds_mean=self.conds_mean,
            conds_std=self.conds_std,
        )
        pred_img = (pred_img + 1) / 2
        return pred_img

    def render(self, noise, cond=None, T=None, cond_other=None, postprocess_fn=None):
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()

        if cond is not None:
            pred_img = render_condition(self.conf,
                                        self.ema_model,
                                        noise,
                                        sampler=sampler,
                                        cond=cond,
                                        cond_other=cond_other,
                                        postprocess_fn=postprocess_fn)
        else:
            pred_img = render_uncondition(self.conf,
                                          self.ema_model,
                                          noise,
                                          sampler=sampler,
                                          latent_sampler=None)
        pred_img = (pred_img + 1) / 2
        return pred_img

    def encode(self, x):
        # TODO:
        assert self.conf.model_type.has_autoenc()
        cond = self.ema_model.encoder.forward(x)
        return cond, None

    def encode_stochastic(self, x, cond, T=None, use_ema=True, cond_other=None):
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()
        if use_ema:
            out = sampler.ddim_reverse_sample_loop(self.ema_model,
                                                x,
                                                model_kwargs={'cond': cond, 'cond_other': cond_other})
        else:
            out = sampler.ddim_reverse_sample_loop(self.model,
                                                x,
                                                model_kwargs={'cond': cond, 'cond_other': cond_other})
        return out['sample']

    def forward(self, noise=None, x_start=None, ema_model: bool = False):
        with amp.autocast(False):
            if ema_model:
                model = self.ema_model
            else:
                model = self.model
            gen = self.eval_sampler.sample(model=model,
                                           noise=noise,
                                           x_start=x_start)
            return gen

    def setup(self, stage=None) -> None:
        """
        make datasets & seeding each worker separately
        """
        ##############################################
        # NEED TO SET THE SEED SEPARATELY HERE

        # checkpoint_path = f'{self.conf.logdir}/last.ckpt'
        # if os.path.exists(checkpoint_path):
        #     state_dict = torch.load(checkpoint_path)
        #     self.load_state_dict(state_dict["state_dict"])
        #     del state_dict

        if self.conf.seed is not None:
            seed = self.conf.seed * get_world_size() + self.global_rank
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print('local seed:', seed)
        ##############################################

        if self.conf.data_name == 'vox256':
            self.train_data = self.conf.make_dataset(path=self.conf.vox_path, split='train', batch_size=self.batch_size)
            self.val_data = self.conf.make_dataset(path=self.conf.vox_path, split='test', batch_size=self.batch_size) 
        else:
            self.train_data = self.conf.make_dataset()
            self.val_data = self.train_data
        print('train data:', len(self.train_data))
        print('val data:', len(self.val_data))

    def _train_dataloader(self, drop_last=True):
        """
        really make the dataloader
        """
        # make sure to use the fraction of batch size
        # the batch size is global!
        conf = self.conf.clone()
        conf.batch_size = self.batch_size

        dataloader = conf.make_loader(self.train_data,
                                      shuffle=True,
                                      drop_last=drop_last)
        return dataloader

    def train_dataloader(self):
        """
        return the dataloader, if diffusion mode => return image dataset
        if latent mode => return the inferred latent dataset
        """
        #self.global_step = 0    # EDIT
        print('on train dataloader start ...')
        return self._train_dataloader()

    @property
    def batch_size(self):
        """
        local batch size for each worker
        """
        ws = get_world_size()
        assert self.conf.batch_size % ws == 0
        return self.conf.batch_size // ws

    @property
    def num_samples(self):
        """
        (global) batch size * iterations
        """
        # batch size here is global!
        # global_step already takes into account the accum batches
        return self.global_step * self.conf.batch_size_effective

    def is_last_accum(self, batch_idx):
        """
        is it the last gradient accumulation loop? 
        used with gradient_accum > 1 and to see if the optimizer will perform "step" in this iteration or not
        """
        return (batch_idx + 1) % self.conf.accum_batches == 0

    def infer_whole_dataset(self,
                            with_render=False,
                            T_render=None,
                            render_save_path=None):
        """
        predicting the latents given images using the encoder

        Args:
            both_flips: include both original and flipped images; no need, it's not an improvement
            with_render: whether to also render the images corresponding to that latent
            render_save_path: lmdb output for the rendered images
        """
        loader = self.conf.make_loader(self.val_data,
                                      shuffle=True,
                                      drop_last=False)
        
        model = self.ema_model
        model.eval()
        conds = []

        if with_render:
            sampler = self.conf._make_diffusion_conf(
                T=T_render or self.conf.T_eval).make_sampler()

            if self.global_rank == 0:
                writer = LMDBImageWriter(render_save_path,
                                         format='webp',
                                         quality=100)
            else:
                writer = nullcontext()
        else:
            writer = nullcontext()

        with writer:
            for batch in tqdm(loader, total=len(loader), desc='infer'):
                with torch.no_grad():
                    # (n, c)
                    # print('idx:', batch['index'])
                    if self.conf.data_name == 'vox256' and isinstance(batch['img'], list):
                        imgs_temp = batch['img']
                        batch['img'] = torch.stack(imgs_temp).squeeze()
                    
                    cond = model.encoder.id_forward(batch['img'].to(self.device))
                    # (k, n, c)
                    cond = self.all_gather(cond)

                    if cond.dim() == 3:
                        # (k*n, c)
                        cond = cond.flatten(0, 1)
                # break
        model.train()
        # (N, c) cpu

        conds.append(cond.cpu())  #EDIT
        conds = torch.cat(conds).float()
        
        return conds

    def training_step(self, batch, batch_idx):

        """
        given an input, calculate the loss function
        no optimization at this stage.
        """
        if self.conf.data_name == 'vox256' and isinstance(batch['img'], list):
            imgs_temp = batch['img']
            batch['img'] = torch.stack(imgs_temp).squeeze()

        with amp.autocast(False):
            # batch size here is local!
            # forward
            imgs, idxs = batch['img'], batch['index']
            # print(f'(rank {self.global_rank}) batch size:', len(imgs))
            x_start = imgs
            """
            main training mode!!!
            """
            # with numpy seed we have the problem that the sample t's are related!
            t, weight = self.T_sampler.sample(len(x_start), x_start.device)
            losses = self.sampler.training_losses(model=self.model,
                                                    x_start=x_start,
                                                    t=t)

            loss = 0.
            if 'loss' in losses:
                loss += losses['loss'].mean()
            if 'xT_reg' in losses:
                loss += losses['xT_reg'].mean()
            if 'x0_estimated' in losses:
                self.x0_estimated = losses["x0_estimated"].clamp(-1, 1).detach()
                self.x0_estimated2 = losses["x0_estimated2"].clamp(-1, 1).detach()
            
            self.last_real = self.last_batch_imgs if self.last_batch_imgs is not None else x_start.detach()
            self.last_batch_imgs = x_start.detach()

            # divide by accum batches to make the accumulated gradient exact!
            for key in ['loss', 'vae', 'latent', 'mmd', 'chamfer', 'arg_cnt', 'z_const', 'z_decom', 'other_decom', 'xT_reg']:
                if key in losses:
                    losses[key] = self.all_gather(losses[key]).mean()

            if self.global_rank == 0:
                self.logger.experiment.add_scalar('total_loss', loss,
                                                  self.num_samples)
                for key in ['loss', 'vae', 'latent', 'mmd', 'chamfer', 'arg_cnt', 'z_const', 'z_decom', 'other_decom', 'xT_reg']:
                    if key in losses:
                        self.logger.experiment.add_scalar(
                            f'loss/{key}', losses[key], self.num_samples)

        return {'loss': loss}

    def on_train_batch_end(self, outputs, batch, batch_idx: int,
                           dataloader_idx: int) -> None:
        """
        after each training step ...
        """
        #del outputs['loss']
        if self.is_last_accum(batch_idx):
            # only apply ema on the last gradient accumulation step,
            # if it is the iteration that has optimizer.step()
            if self.conf.train_mode == TrainMode.latent_diffusion:
                # it trains only the latent hence change only the latent
                ema(self.model.latent_net, self.ema_model.latent_net,
                    self.conf.ema_decay)
            else:
                ema(self.model, self.ema_model, self.conf.ema_decay)

            # logging
            if self.conf.train_mode.require_dataset_infer():
                imgs = None
            else:
                if self.conf.data_name == 'vox256' and isinstance(batch['img'], list):
                    imgs_temp = batch['img']
                    batch['img'] = torch.stack(imgs_temp).squeeze()
                imgs = batch['img']
            self.log_sample(x_start=imgs)
            self.evaluate_scores()

    def on_before_optimizer_step(self, optimizer: Optimizer,
                                 optimizer_idx: int) -> None:
        # fix the fp16 + clip grad norm problem with pytorch lightinng
        # this is the currently correct way to do it
        if self.conf.grad_clip > 0:
            # from trainer.params_grads import grads_norm, iter_opt_params
            params = [
                p for group in optimizer.param_groups for p in group['params']
            ]
            # print('before:', grads_norm(iter_opt_params(optimizer)))
            torch.nn.utils.clip_grad_norm_(params,
                                           max_norm=self.conf.grad_clip)
            # print('after:', grads_norm(iter_opt_params(optimizer)))

    def log_sample(self, x_start):
        """
        put images to the tensorboard
        """
        def do(model,
               postfix,
               use_xstart,
               save_real=False):
            model.eval()
            #self.optimizers().zero_grad()

            if self.x0_estimated is not None:
                x0_estimated_imgs = self.all_gather(self.x0_estimated)
                x0_estimated2_imgs = self.all_gather(self.x0_estimated2)
                #last_batch_imgs = self.all_gather(self.last_batch_imgs)
              
                if x0_estimated_imgs.dim() == 5:
                    # (n, c, h, w)
                    x0_estimated_imgs = x0_estimated_imgs.flatten(0, 1)
                    x0_estimated2_imgs = x0_estimated2_imgs.flatten(0, 1)
                    #last_batch_imgs = last_batch_imgs.flatten(0, 1)

                if self.global_rank == 0:
                    # save samples to the tensorboard
                    x0_estimated_imgs = (make_grid(x0_estimated_imgs) + 1) / 2
                    x0_estimated2_imgs = (make_grid(x0_estimated2_imgs) + 1) / 2
                    #last_batch_imgs = (make_grid(last_batch_imgs) + 1) / 2
                    sample_dir = os.path.join(self.conf.logdir,
                                                f'x0_estimated{postfix}')
                    if not os.path.exists(sample_dir):
                        os.makedirs(sample_dir)
                    save_image(x0_estimated_imgs, os.path.join(sample_dir, '%d_x0_estimated.png' % self.num_samples))
                    save_image(x0_estimated2_imgs, os.path.join(sample_dir, '%d_x0_estimated2.png' % self.num_samples))
                    #save_image(last_batch_imgs, os.path.join(sample_dir, '%d_real.png' % self.num_samples))
                    self.logger.experiment.add_image(f'x0_estimated{postfix}/1', x0_estimated_imgs, self.num_samples)
                    self.logger.experiment.add_image(f'x0_estimated{postfix}/2', x0_estimated2_imgs, self.num_samples)
                    #self.logger.experiment.add_image(f'x0_estimated{postfix}/real', last_batch_imgs, self.num_samples)

            with torch.no_grad():

                ### EDIT ###
                cond = model.encoder.forward(x_start)
                cond_other = None
                stochastic_x_T = self.encode_stochastic(x_start, cond, use_ema=('ema' in postfix), cond_other=cond_other)
                random_x_T = torch.randn(len(x_start), 3, self.conf.img_size, self.conf.img_size, device=self.device)
                # Gen = []
                # for x_T in loader:
                x_T = stochastic_x_T
                if use_xstart:
                    assert len(x_T) == len(x_start)
                    _xstart = x_start   # [:len(x_T)]
                else:
                    _xstart = None

                gen = self.eval_sampler.sample(model=model,
                                                noise=x_T,
                                                cond=cond,
                                                x_start=_xstart,
                                                cond_other=cond_other)
            
                x_T = random_x_T
                gen_random = self.eval_sampler.sample(model=model,
                                                noise=x_T,
                                                cond=cond,
                                                x_start=_xstart,
                                                cond_other=cond_other)
            
                x_T = stochastic_x_T
                cond_decom = model.encoder.forward(id=_xstart, lnd=self.last_real)
                gen_decom = self.eval_sampler.sample(model=model,
                                                    noise=x_T,
                                                    cond=cond_decom,
                                                    x_start=_xstart,
                                                    cond_other=None)

                # gen = torch.cat(Gen)

                gen = self.all_gather(gen)
                gen_random = self.all_gather(gen_random)
                gen_decom = self.all_gather(gen_decom) if gen_decom is not None else None

                if gen.dim() == 5:
                    # (n, c, h, w)
                    gen = gen.flatten(0, 1)
                    gen_random = gen_random.flatten(0, 1)
                    gen_decom = gen_decom.flatten(0, 1) if gen_decom is not None else None

                if save_real and use_xstart:
                    # save the original images to the tensorboard
                    real = self.all_gather(_xstart)
                    last_real = self.all_gather(self.last_real)
                    if real.dim() == 5:
                        real = real.flatten(0, 1)
                        last_real = last_real.flatten(0, 1)

                    if self.global_rank == 0:
                        grid_real = (make_grid(real) + 1) / 2
                        self.logger.experiment.add_image(
                            f'sample{postfix}/real', grid_real,
                            self.num_samples)
                        real_sample_dir = os.path.join(self.conf.logdir,
                                              f'real_sample{postfix}')
                        if not os.path.exists(real_sample_dir):
                            os.makedirs(real_sample_dir)
                        path = os.path.join(real_sample_dir,
                                        '%d.png' % self.num_samples)
                        save_image(grid_real, path)
                        if gen_decom is not None:
                            grid_real = (make_grid(last_real) + 1) / 2
                            self.logger.experiment.add_image(
                                f'sample{postfix}/last_batch', grid_real,
                                self.num_samples)
                            path = os.path.join(real_sample_dir,
                                            '%d_last_batch.png' % self.num_samples)
                            save_image(grid_real, path)

                if self.global_rank == 0:
                    # save samples to the tensorboard
                    grid = (make_grid(gen) + 1) / 2
                    sample_dir = os.path.join(self.conf.logdir,
                                              f'sample{postfix}')
                    if not os.path.exists(sample_dir):
                        os.makedirs(sample_dir)
                    path = os.path.join(sample_dir,
                                        '%d_recon.png' % self.num_samples)
                    save_image(grid, path)
                    self.logger.experiment.add_image(f'sample{postfix}/recon', grid,
                                                     self.num_samples)
                    grid = (make_grid(gen_random) + 1) / 2
                    path = os.path.join(sample_dir,
                                        '%d_rand_xT.png' % self.num_samples)
                    save_image(grid, path)
                    self.logger.experiment.add_image(f'sample{postfix}/rand_xT', grid,
                                                     self.num_samples)
                    if gen_decom is not None:
                        grid = (make_grid(gen_decom) + 1) / 2
                        path = os.path.join(sample_dir,
                                            '%d_decom.png' % self.num_samples)
                        save_image(grid, path)
                        self.logger.experiment.add_image(f'sample{postfix}/decom', grid,
                                                        self.num_samples)
            model.train()

        if self.conf.sample_every_samples > 0 and is_time(
                self.num_samples, self.conf.sample_every_samples,
                self.conf.batch_size_effective):   
            do(self.model, '', use_xstart=True, save_real=True)
            do(self.ema_model, '_ema', use_xstart=True, save_real=True)

    def evaluate_scores(self):
        """
        evaluate FID and other scores during training (put to the tensorboard)
        For, FID. It is a fast version with 5k images (gold standard is 50k).
        Don't use its results in the paper!
        """
        def fid(model, postfix):
            # self.optimizers().zero_grad()
            score = evaluate_fid(self.eval_sampler,
                                 model,
                                 self.conf,
                                 device=self.device,
                                 train_data=self.train_data,
                                 val_data=self.val_data,
                                 latent_sampler=self.eval_latent_sampler,
                                 conds_mean=self.conds_mean,
                                 conds_std=self.conds_std)
            if self.global_rank == 0:
                self.logger.experiment.add_scalar(f'FID{postfix}', score,
                                                  self.num_samples)
                if not os.path.exists(self.conf.logdir):
                    os.makedirs(self.conf.logdir)
                with open(os.path.join(self.conf.logdir, 'eval.txt'),
                          'a') as f:
                    metrics = {
                        f'FID{postfix}': score,
                        'num_samples': self.num_samples,
                    }
                    f.write(json.dumps(metrics) + "\n")

        def lpips(model, postfix):
            if self.conf.model_type.has_autoenc(
            ) and self.conf.train_mode.is_autoenc():
                # {'lpips', 'ssim', 'mse'}
                score = evaluate_lpips(self.eval_sampler,
                                       model,
                                       self.conf,
                                       device=self.device,
                                       val_data=self.val_data,
                                       latent_sampler=self.eval_latent_sampler,
                                       use_inverted_noise=True)

                if self.global_rank == 0:
                    for key, val in score.items():
                        self.logger.experiment.add_scalar(
                            f'{key}{postfix}', val, self.num_samples)

        if self.conf.eval_every_samples > 0 and self.num_samples > 0 and is_time(
                self.num_samples, self.conf.eval_every_samples,
                self.conf.batch_size_effective):
            print(f'eval fid @ {self.num_samples}')
            fid(self.model, '')
            lpips(self.model, '')

        if self.conf.eval_ema_every_samples > 0 and self.num_samples > 0 and is_time(
                self.num_samples, self.conf.eval_ema_every_samples,
                self.conf.batch_size_effective):
            print(f'eval fid ema @ {self.num_samples}')
            fid(self.ema_model, '_ema')
            # it's too slow
            # lpips(self.ema_model, '_ema')

    def configure_optimizers(self):
        out = {}
        if self.conf.optimizer == OptimizerType.adam:
            optim = torch.optim.Adam(self.model.parameters(),
                                     lr=self.conf.lr,
                                     weight_decay=self.conf.weight_decay)
        elif self.conf.optimizer == OptimizerType.adamw:
            optim = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.conf.lr,
                                      weight_decay=self.conf.weight_decay)
        else:
            raise NotImplementedError()
        out['optimizer'] = optim
        if self.conf.warmup > 0:
            sched = torch.optim.lr_scheduler.LambdaLR(optim,
                                                      lr_lambda=WarmupLR(
                                                          self.conf.warmup))
            out['lr_scheduler'] = {
                'scheduler': sched,
                'interval': 'step',
            }
        return out

    def split_tensor(self, x):
        """
        extract the tensor for a corresponding "worker" in the batch dimension

        Args:
            x: (n, c)

        Returns: x: (n_local, c)
        """
        n = len(x)
        rank = self.global_rank
        world_size = get_world_size()
        # print(f'rank: {rank}/{world_size}')
        per_rank = n // world_size
        return x[rank * per_rank:(rank + 1) * per_rank]

    def test_step(self, batch, *args, **kwargs):
        """
        for the "eval" mode. 
        We first select what to do according to the "conf.eval_programs". 
        test_step will only run for "one iteration" (it's a hack!).
        
        We just want the multi-gpu support. 
        """
        # make sure you seed each worker differently!
        self.setup()

        # it will run only one step!
        print('global step:', self.global_step)
        """
        "infer" = predict the latent variables using the encoder on the whole dataset
        """
        if 'infer' in self.conf.eval_programs:
            print('infer ...')
            conds = self.infer_whole_dataset()
            conds = conds.float()
            # NOTE: always use this path for the latent.pkl files
            save_path = f'checkpoints/{self.conf.name}/latent.pkl'
        else:
            raise NotImplementedError()

        if self.global_rank == 0:
            conds_mean = conds.mean(dim=0)
            conds_std = conds.std(dim=0)
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            torch.save(
                {
                    'conds': conds,
                    'conds_mean': conds_mean,
                    'conds_std': conds_std,
                }, save_path)


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay +
                                    source_dict[key].data * (1 - decay))


class WarmupLR:
    def __init__(self, warmup) -> None:
        self.warmup = warmup

    def __call__(self, step):
        return min(step, self.warmup) / self.warmup


def is_time(num_samples, every, step_size):
    closest = (num_samples // every) * every
    return num_samples - closest < step_size


def train(conf: TrainConfig, gpus, nodes=1, mode: str = 'train'):
    #torch.autograd.set_detect_anomaly(True)

    print('conf:', conf.name)
    model = LitModel(conf)

    if not os.path.exists(conf.logdir):
        os.makedirs(conf.logdir)
    checkpoint = ModelCheckpoint(dirpath=f'{conf.logdir}',
                                 save_last=True,
                                 save_top_k=-1,
                                 every_n_train_steps=conf.save_every_samples //
                                 conf.batch_size_effective)
    checkpoint_path = f'{conf.logdir}/last.ckpt'
    print('ckpt path:', checkpoint_path)
    if os.path.exists(checkpoint_path):
        resume = checkpoint_path
        model.load_model(resume, strict=True)
        print('resume!')  
    else:
        resume = None

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=conf.logdir,
                                             name=None,
                                             version='')

    plugins = []
    if len(gpus) == 1 and nodes == 1:
        accelerator = None
    else:
        accelerator = 'ddp'
        from pytorch_lightning.plugins import DDPPlugin

        # important for working with gradient checkpoint
        plugins.append(DDPPlugin(find_unused_parameters=False))

    trainer = pl.Trainer(
        max_steps=conf.total_samples // conf.batch_size_effective,
        resume_from_checkpoint=resume,
        gpus=gpus,
        num_nodes=nodes,
        accelerator=accelerator,
        precision=16 if conf.fp16 else 32,
        callbacks=[
            checkpoint,
            LearningRateMonitor(),
        ],
        # clip in the model instead
        # gradient_clip_val=conf.grad_clip,
        replace_sampler_ddp=True,
        logger=tb_logger,
        accumulate_grad_batches=conf.accum_batches,
        plugins=plugins
    )

    if mode == 'train':
        trainer.fit(model)
    elif mode == 'eval':
        # load the latest checkpoint
        # perform lpips
        # dummy loader to allow calling "test_step"
        dummy = DataLoader(TensorDataset(torch.tensor([0.] * conf.batch_size)),
                           batch_size=conf.batch_size)
        eval_path = conf.eval_path or checkpoint_path
        print('loading from:', eval_path)
        state = torch.load(eval_path, map_location='cpu')
        print('step:', state['global_step'])
        model.load_state_dict(state['state_dict'])
        # trainer.fit(model)
        out = trainer.test(model, dataloaders=dummy)
        # first (and only) loader
        out = out[0]
        print(out)
    else:
        raise NotImplementedError()
