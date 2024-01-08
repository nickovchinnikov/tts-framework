# CUDA drivers
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb 
# sudo dpkg -i cuda-keyring_1.0-1_all.deb
# sudo apt-get update

# If you have dead loop and CUDA out of memory error, run the following commands
# pkill -9 python

# Update the system
sudo apt update && sudo apt upgrade -y

# Install CUDA drivers
# sudo apt -y install cuda-drivers
# sudo apt -y install nvidia-cuda-toolkit

sudo apt install ffmpeg libasound2-dev -y

# Connect the bucket to the VM instance, cd tts_framework
cp ../tts-training-bucket/vocoder.ckpt ./checkpoints/
cp ../tts-training-bucket/en_us_cmudict_ipa_forward.pt ./checkpoints/
cp ../tts-training-bucket/epoch\=5537-step\=615041.ckpt ./checkpoints/

# Copy the datasets_cache folder metadata
mkdir -p ./datasets_cache/LIBRITTS/LibriTTS/
cp ../tts-training-bucket/datasets_cache/LIBRITTS/LibriTTS/BOOKS.TXT ./datasets_cache/LIBRITTS/LibriTTS/
cp ../tts-training-bucket/datasets_cache/LIBRITTS/LibriTTS/CHAPTERS.TXT ./datasets_cache/LIBRITTS/LibriTTS/
cp ../tts-training-bucket/datasets_cache/LIBRITTS/LibriTTS/eval_sentences10.tsv ./datasets_cache/LIBRITTS/LibriTTS/
cp ../tts-training-bucket/datasets_cache/LIBRITTS/LibriTTS/LICENSE.txt ./datasets_cache/LIBRITTS/LibriTTS/
cp ../tts-training-bucket/datasets_cache/LIBRITTS/LibriTTS/NOTE.txt ./datasets_cache/LIBRITTS/LibriTTS/
cp ../tts-training-bucket/datasets_cache/LIBRITTS/LibriTTS/reader_book.tsv ./datasets_cache/LIBRITTS/LibriTTS/
cp ../tts-training-bucket/datasets_cache/LIBRITTS/LibriTTS/README_librispeech.txt ./datasets_cache/LIBRITTS/LibriTTS/
cp ../tts-training-bucket/datasets_cache/LIBRITTS/LibriTTS/README_libritts.txt ./datasets_cache/LIBRITTS/LibriTTS/
cp ../tts-training-bucket/datasets_cache/LIBRITTS/LibriTTS/speakers.tsv ./datasets_cache/LIBRITTS/LibriTTS/
cp ../tts-training-bucket/datasets_cache/LIBRITTS/LibriTTS/SPEAKERS.txt ./datasets_cache/LIBRITTS/LibriTTS/

# Download the dataset
curl -O https://us.openslr.org/resources/141/train_clean_360.tar.gz
mv train_clean_360.tar.gz ./datasets_cache/LIBRITTS/

tar -xzvf ./datasets_cache/LIBRITTS/train_clean_360.tar.gz -C ./datasets_cache/LIBRITTS/
mv ./datasets_cache/LIBRITTS/LibriTTS_R/train-clean-360 ./datasets_cache/LIBRITTS/LibriTTS
rm -r ./datasets_cache/LIBRITTS/LibriTTS_R

conda env create -f environment.yml
conda activate tts_framework

pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

cp training/scripts/train_dist.py ./

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib
