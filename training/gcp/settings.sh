# Stop instance:
# sudo shutdown -h now

# Start instance:
# gcloud workbench instances start prod-train  --location=europe-west4-a

# CUDA drivers
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb 
# sudo dpkg -i cuda-keyring_1.0-1_all.deb
# sudo apt-get update

# If you have dead loop and CUDA out of memory error, run the following commands
# pkill -9 python

# Update the system
sudo apt update && sudo apt upgrade -y

# espeak-ng
sudo apt-get install espeak-ng -y

# Install CUDA drivers
# sudo apt -y install cuda-drivers
# sudo apt -y install nvidia-cuda-toolkit

sudo apt install ffmpeg libasound2-dev -y

mkdir ./tts-training-bucket

# Mount the bucket
gcsfuse tts-training-bucket ./tts-training-bucket

# Clone the repo
# git clone git@github.com:nickovchinnikov/tts-framework.git

# Connect the bucket to the VM instance, cd tts_framework
cp ./tts-training-bucket/vocoder.ckpt ./tts-framework/checkpoints/
cp ./tts-training-bucket/en_us_cmudict_ipa_forward.pt ./tts-framework/checkpoints/
cp ./tts-training-bucket/epoch\=5537-step\=615041.ckpt ./tts-framework/checkpoints/

# Copy the datasets_cache folder metadata
mkdir -p ./tts-framework/datasets_cache/LIBRITTS/LibriTTS/

cp ./tts-training-bucket/datasets_cache/LIBRITTS/LibriTTS/BOOKS.txt ./tts-framework/datasets_cache/LIBRITTS/LibriTTS/
cp ./tts-training-bucket/datasets_cache/LIBRITTS/LibriTTS/CHAPTERS.txt ./tts-framework/datasets_cache/LIBRITTS/LibriTTS/
cp ./tts-training-bucket/datasets_cache/LIBRITTS/LibriTTS/eval_sentences10.tsv ./tts-framework/datasets_cache/LIBRITTS/LibriTTS/
cp ./tts-training-bucket/datasets_cache/LIBRITTS/LibriTTS/LICENSE.txt ./tts-framework/datasets_cache/LIBRITTS/LibriTTS/
cp ./tts-training-bucket/datasets_cache/LIBRITTS/LibriTTS/NOTE.txt ./tts-framework/datasets_cache/LIBRITTS/LibriTTS/
cp ./tts-training-bucket/datasets_cache/LIBRITTS/LibriTTS/reader_book.tsv ./tts-framework/datasets_cache/LIBRITTS/LibriTTS/
cp ./tts-training-bucket/datasets_cache/LIBRITTS/LibriTTS/README_librispeech.txt ./tts-framework/datasets_cache/LIBRITTS/LibriTTS/
cp ./tts-training-bucket/datasets_cache/LIBRITTS/LibriTTS/README_libritts.txt ./tts-framework/datasets_cache/LIBRITTS/LibriTTS/
cp ./tts-training-bucket/datasets_cache/LIBRITTS/LibriTTS/speakers.tsv ./tts-framework/datasets_cache/LIBRITTS/LibriTTS/
cp ./tts-training-bucket/datasets_cache/LIBRITTS/LibriTTS/SPEAKERS.txt ./tts-framework/datasets_cache/LIBRITTS/LibriTTS/

#############################################################################################################
######################################### LibriTTS-R dataset ################################################
#############################################################################################################

# Download the dataset train_clean_100
curl -O https://us.openslr.org/resources/141/train_clean_100.tar.gz
mv train_clean_100.tar.gz ./tts-framework/datasets_cache/LIBRITTS/

tar -xzvf ./tts-framework/datasets_cache/LIBRITTS/train_clean_100.tar.gz -C ./tts-framework/datasets_cache/LIBRITTS/
mv ./tts-framework/datasets_cache/LIBRITTS/LibriTTS_R/train-clean-100 ./tts-framework/datasets_cache/LIBRITTS/LibriTTS
rm -r ./tts-framework/datasets_cache/LIBRITTS/LibriTTS_R

# Download the dataset train_clean_360
curl -O https://us.openslr.org/resources/141/train_clean_360.tar.gz
mv train_clean_360.tar.gz ./tts-framework/datasets_cache/LIBRITTS/

tar -xzvf ./tts-framework/datasets_cache/LIBRITTS/train_clean_360.tar.gz -C ./tts-framework/datasets_cache/LIBRITTS/
mv ./tts-framework/datasets_cache/LIBRITTS/LibriTTS_R/train-clean-360 ./tts-framework/datasets_cache/LIBRITTS/LibriTTS
rm -r ./tts-framework/datasets_cache/LIBRITTS/LibriTTS_R

#############################################################################################################
#########################################  LibriTTS dataset  ################################################
#############################################################################################################

# curl -O https://us.openslr.org/resources/60/train-clean-100.tar.gz
# mv train-clean-100.tar.gz ./tts-framework/datasets_cache/LIBRITTS/

# tar -xzvf ./tts-framework/datasets_cache/LIBRITTS/train-clean-100.tar.gz -C ./tts-framework/datasets_cache/LIBRITTS/
# mv ./tts-framework/datasets_cache/LIBRITTS/LibriTTS/train-clean-100 ./tts-framework/datasets_cache/LIBRITTS/LibriTTS
# rm -r ./tts-framework/datasets_cache/LIBRITTS/LibriTTS_R

#############################################################################################################

conda env create -f ./tts-framework/environment.yml
conda activate tts_framework

# pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

cp ./tts-framework/training/scripts/train_dist.py ./tts-framework

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib
