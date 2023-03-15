from templates import *

if __name__ == '__main__':
    # We use 4 GPUs for training
    gpus = [0,1,2,3]
    nodes = 1
    conf = diffusion_video_autoencoder(gpus)
    train(conf, gpus=gpus, nodes=nodes)
    