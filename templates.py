from experiment import *

def diffusion_video_autoencoder(gpus):
    conf = TrainConfig()
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = 'linear'
    conf.data_name = 'vox256'
    conf.vox_path = "/home/server03/voxceleb/result/stit_out_vox1"
    conf.diffusion_type = 'beatgans'
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_autoenc
    conf.beatgans_loss_type = LossType.l1
    conf.loss_coef = 1
    conf.xT_reg_coef = 1
    conf.z_const_coef = 100
    conf.other_decom_coef = 1
    conf.z_decom_coef = 1
    conf.net_attn = (16, )
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 512
    conf.net_beatgans_resnet_two_cond = True
    conf.net_enc_pool = 'adaptivenonzero'
    conf.sample_size = 32
    conf.T_eval = 20
    conf.T = 1000
    conf.batch_size = 4
    conf.sample_every_samples = 10_000
    conf.scale_up_gpus(len(gpus))
    conf.img_size = 256
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4) 
    conf.eval_num_images = 1_000
    conf.eval_every_samples = 1_000_000
    conf.eval_ema_every_samples = 1_000_000
    conf.total_samples = 200_000_000 
    conf.make_model_conf()
    conf.name = 'diffusion_video_autoencoder'
    return conf

def diffusion_video_autoencoder_eval(gpus):
    conf = diffusion_video_autoencoder(gpus)
    conf.eval_programs = ['infer']
    conf.eval_path = "checkpoints/diffusion_video_autoencoder/last.ckpt"
    conf.batch_size = 64
    conf.base_dir = 'checkpoints'
    return conf