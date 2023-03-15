from templates import *

if __name__ == '__main__':
    gpus = [0]
    nodes = 1
    conf = diffusion_video_autoencoder_eval(gpus)
    train(conf, gpus=gpus, nodes=nodes, mode='eval')