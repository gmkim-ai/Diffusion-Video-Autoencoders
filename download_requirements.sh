echo "Creating empty directory.."
mkdir checkpoints
mkdir checkpoints/diffusion_video_autoencoder
mkdir checkpoints/diffusion_video_autoencoder_cls
mkdir editing_CLIP
mkdir editing_classifier

echo "Downloading model checkpoints.."
wget "https://drive.google.com/uc?export=download&id=1J_eAs5-KWn--kmcqx5S-QjVPskTczTMU" -O "checkpoints/diffusion_video_autoencoder/latent.pkl"
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Q6z7Wmo9xve5SgfDJExohpDSSNi4QxcH" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=1Q6z7Wmo9xve5SgfDJExohpDSSNi4QxcH" -o "checkpoints/diffusion_video_autoencoder/epoch=51-step=1000000.ckpt"

echo "Downloading classifier checkpoints.."
wget "https://drive.google.com/uc?export=download&id=1nddbwuAvIVEqKkkmUF96nJj91jVmAdtR" -O "checkpoints/diffusion_video_autoencoder_cls/last.ckpt"

echo "Downloading pre-trained models checkpoints.."
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1X2u83hSO2jObO6pCCpZXa45qpQU2ECKJ" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=1X2u83hSO2jObO6pCCpZXa45qpQU2ECKJ" -o "pretrained_models.zip"
echo "Unzip pre-trained models.."
unzip pretrained_models.zip

echo "Downloading CelebA datasets for training classifier.."
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Kk7U6aeuQCY2-lhOOXxzTbxa5AYidsK2" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=1Kk7U6aeuQCY2-lhOOXxzTbxa5AYidsK2" -o "datasets.zip"
echo "Unzip CelebA datasets.."
unzip datasets.zip