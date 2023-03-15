from templates_cls import *
from experiment_classifier import *

if __name__ == '__main__':
    gpus = [0]
    conf = diffusion_video_autoencoder_cls(gpus)
    train_cls(conf, gpus=gpus)

    # after this you can do the manipulation!