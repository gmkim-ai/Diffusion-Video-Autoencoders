from templates import *

def diffusion_video_autoencoder_cls(gpus):
    conf = diffusion_video_autoencoder(gpus)
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.celebahq_all
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f'checkpoints/{diffusion_video_autoencoder(gpus).name}/latent.pkl' 
    conf.batch_size = 32
    conf.lr = 1e-3
    conf.total_samples = 300_000
    conf.data_name = 'celebalmdb'
    conf.name = 'diffusion_video_autoencoder_cls'
    return conf