# Diffusion Video Autoencoders: Toward Temporally Consistent Face Video Editing via Disentangled Video Encoding (CVPR 2023)

[![arXiv](https://img.shields.io/badge/arXiv-2212.02802-b31b1b.svg)](https://arxiv.org/abs/2212.02802)

### [[Project Page](https://diff-video-ae.github.io/)]
> **Diffusion Video Autoencoders: Toward Temporally Consistent Face Video Editing via Disentangled Video Encoding**<br>
> Gyeongman Kim, Hajin Shim, Hyunsu Kim, Yunjey Choi, Junho Kim, Eunho Yang <br>

>**Abstract**: <br>
> Inspired by the impressive performance of recent face image editing methods, several studies have been naturally proposed to extend these methods to the face video editing task. One of the main challenges here is temporal consistency among edited frames, which is still unresolved. To this end, we propose a novel face video editing framework based on diffusion autoencoders that can successfully extract the decomposed features - for the first time as a face video editing model - of identity and motion from a given video. This modeling allows us to edit the video by simply manipulating the temporally invariant feature to the desired direction for the consistency. Another unique strength of our model is that, since our model is based on diffusion models, it can satisfy both reconstruction and edit capabilities at the same time, and is robust to corner cases in wild face videos (e.g. occluded faces) unlike the existing GAN-based methods.


## Requirements

See `dva.yaml`
```
conda env create -f dva.yaml
conda activate dva
```

To perform StyleCLIP edits, install clip with:

```
pip install git+https://github.com/openai/CLIP.git
```

### Pretrained models

In order to use this project you need to download pretrained models and datasets.

Use the `download_requirements.sh` script
```
sh download_requirements.sh
```
This script downloads 
- Model(diffusion video autoencoder, classifier) checkpoints for reproducibility in `checkpoints` folder. 
- Pre-trained models for id encoder, landmark encoder, background prediction, etc. in `pretrained_models` folder.
- CelebA-HQ datasets for training your own classifier in `datasets` folder. (don't need this if you use provided checkpoints only.)


## Splitting videos into frames

Our code expects videos in the form of a directory with individual frame images.

To produce such a directory from an existing video, we recommend using ffmpeg:
```
ffmpeg -i "video.mp4" "video_frames/out%04d.png"
```
We also provide sample frames in `sample_video`. You can use this directory without your own video.


## Editing

You can edit videos directly with our provided pre-trained models. We use NVIDIA GeForce RTX 3090 for editing experiments.

To adjust DDIM total diffusion step, use `--T` option (default: 1000)

### CLIP-based Editing 

To run CLIP-based editing:
```
python editing_CLIP.py --attribute ATTRIBUTE_NAME \
                       --src_txt NEUTRAL_TEXT \
                       --trg_txt TARGET_TEXT \
                       --lr LEARNING_RATE \
                       --scale EDITING_STEP_SIZE \
                       --clip_loss_w WEIGHT_OF_CLIP_LOSS \
                       --id_loss_w WEIGHT_OF_ID_LOSS \
                       --l1_loss_w WEIGHT_OF_L1_LOSS \
                       --video_path /path/to/frames_dir
```

`--attribute` option just determines the log directory name. Please refer to the appendix of the paper regarding the search space of hyperparameters.

For example:
```
python editing_CLIP.py --attribute "Beard" \
                       --src_txt "face" \
                       --trg_txt "face with beard" \
                       --lr 0.002 \
                       --scale 0.5 \
                       --clip_loss_w 3 \
                       --id_loss_w 1 \
                       --l1_loss_w 5 \
                       --video_path "sample_video"
```

### Classifier-based Editing 

To run Classifier-based editing:
```
python editing_classifier.py --attribute PREDEFINED_ATTRIBUTE \
                             --scale EDITING_SCALE \ 
                             --normalize \
                             --video_path /path/to/frames_dir
```

`--attribute` option should be one of pre-defined CelebA-HQ attributes, as follows:
```
['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
```

`--scale` option determines the editing scale and +, - both possible. (ex. "-" scale with "No_Beard" attribute will make a beard.)

For example:
```
python editing_classifier.py --attribute "Mustache" \
                             --scale 0.25 \ 
                             --normalize \
                             --video_path "sample_video"
```


## Training

In order to train diffusion video autoencoder with your own data, please follow the below steps for editing. We use 4x V100s for training.

**1. Train the model with your own data and get the model checkpoint `last.ckpt`.**

Set `conf.vox_path` in `templates.py` to the data path. Data path should be organized as follows:
```
└── Video_Dataset
    ├── train
    │   ├── video_1
    │   │   ├── 0000000.png
    │   │   │        :
    │   │   └── 1000000.png
    │   │    : 
    │   └── video_n
    │       ├── 0000000.png
    │       │        :
    │       └── 1000000.png
    └── test
        └── (same as train)
```
All images in dataset should be FFHQ-like aligned and cropped frames.

Then, train the model with:
```
python run_train.py
```

**2. (Optional) Compute the statistics of identity feature for normalization in classifier training phase.**

To run classifier-based editing, you need statistics information `latent.pkl` for training classifier. (You don't need this step if you edit the video with CLIP method)

Set `conf.eval_path` in `templates.py` to the model_checkpoint `checkpoints/diffusion_video_autoencoder/last.ckpt`. 

Then, remove the `checkpoints/diffusion_video_autoencoder/latent.pkl` which is downloaded and get new one with:
```
python run_eval.py
```

**3. (Optional) Train the classifier**

Same as above, remove the `checkpoints/diffusion_video_autoencoder_cls/last.ckpt` which is downloaded and get new one with:
```
python run_cls.py
```

`editing_classifier.py` will use this `checkpoints/diffusion_video_autoencoder_cls/last.ckpt` checkpoint.


## Credits
Diffusion Autoencoders implementation:  
https://github.com/phizaz/diffae  
License (MIT) https://github.com/phizaz/diffae/blob/master/LICENSE

STIT implementation:  
https://github.com/rotemtzaban/STIT  
License (MIT) https://github.com/rotemtzaban/STIT/blob/main/LICENSE

PyTorch Face Landmark implementation:   
https://github.com/cunjian/pytorch_face_landmark  


## Citation

If you make use of our work, please cite our paper:

```
@InProceedings{Kim_2023_CVPR,
    author    = {Kim, Gyeongman and Shim, Hajin and Kim, Hyunsu and Choi, Yunjey and Kim, Junho and Yang, Eunho},
    title     = {Diffusion Video Autoencoders: Toward Temporally Consistent Face Video Editing via Disentangled Video Encoding},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {6091-6100}
}
```
