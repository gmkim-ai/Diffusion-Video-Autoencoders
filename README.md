Official Implementations
- project page
Requirements
- Env
- Pretrained models & CelebA (download.sh)
Splitting videos into frames
- jim video sample
Editing
- classifier, CLIP
Training
Eval
classifier training
Credits
- DiffAE STIT
Citation


# Implementation of Diffusion Video Autoencoders

This repository is written based on CVPR 2022 diffusion autoencoder paper. ([paper](https://openaccess.thecvf.com/content/CVPR2022/html/Preechakul_Diffusion_Autoencoders_Toward_a_Meaningful_and_Decodable_Representation_CVPR_2022_paper.html))


### Prerequisites

See `3090_dva.yaml`

```
conda env create -f 3090_dva.yaml
conda activate dva
pip install git+https://github.com/openai/CLIP.git
sh download_requirements.sh
```

### Quick start

Basically, we follow the below steps.
1. 'run_train.py' for training diffusion video autoencoder.
2. 'run_eval.py' for computing statistics (latent.pkl) of identity feature for normalization in classifier training phase.
3. 'run_cls.py' for training classifier based on CelebA_annotation. (You don't need this step if you edit the video with CLIP method)
4. 'editing_classifier.py' or 'editing_CLIP.py' for editing video.

However, we need VoxCeleb1 and CelebA annotation dataset for training model and classifier.
Therefore, you can only run editing experiments code 'editing_classifier.py', 'editing_CLIP.py' with sample data in these supplementary.

For reproducibility, we provide the pre-trained model parameter, classifier parameter, etc. in anonymous google account's drive.
 
1. Download files in Google Drive: (https://drive.google.com/drive/folders/1if93J3cGhNTb6JGHfZIFD-lpD9o4ZntD?usp=sharing) (This account is anonymous account)
2. Put 'epoch=51-step=1000000.ckpt' and 'latent.pkl' in 'checkpoints/diffusion_video_autoencoder' folder.
3. Put 'last.ckpt' in 'checkpoints/diffusion_video_autoencoder_cls' folder.
4. Unzip 'pretrained_models.zip'.
5. run 'editing_classifier.py' or 'editing_CLIP.py' with prefer setting about editing.

['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']


python editing_classifier.py --attribute "Mustache" \
                                --scale 0.4 \
                                --video_path "jim"

python editing_CLIP.py --attribute "Beard" \
                                --src_txt "face" \
                                --trg_txt "face with beard" \
                                --video_path "jim"

python editing_CLIP.py --attribute "Beard_2000_scale0.5" --src_txt "face" --trg_txt "face with beard" --video_path "jim" --scale 0.5 --id_loss_w 1 --l1_loss_w 5
python editing_CLIP.py --attribute "Beard_test" --src_txt "face" --trg_txt "face with beard" --video_path "jim"
python editing_CLIP.py --attribute "Beard_2000_scale0.5" --src_txt "face" --trg_txt "face with beard" --video_path "jim" --scale 0.5 --id_loss_w 3 --l1_loss_w 1
python editing_CLIP.py --attribute "Beard_l1mask_2000_scale0.5" --src_txt "face" --trg_txt "face with beard" --video_path "jim" --id_loss_w 1 --l1_loss_w 5 --scale 0.5

run_train.py
templates.py conf.vox_path = "home/server03/voxceleb/result/stit_out_vox1"
vox_path explain
=> get last.ckpt

run_eval.py
templates.py diffusion_video_autoencoder_eval conf.eval_path = "checkpoints/diffusion_video_autoencoder/last.ckpt"
=> get latent.pkl

run_cls.py
=> get classifier last.ckpt
=> use when editing_classifier.py
