# Update the system
sudo apt update && sudo apt upgrade -y
# Install deps
sudo apt install ffmpeg libasound2-dev -y

# Mount the bucket
mkdir ./tts-training-bucket
gcsfuse tts-training-bucket ./tts-training-bucket
# Unmount the bucket
# fusermount -u ./tts-training-bucket

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

# Download the dataset train_clean_360
curl -O https://us.openslr.org/resources/141/train_clean_360.tar.gz
mv train_clean_360.tar.gz ./tts-framework/datasets_cache/LIBRITTS/

tar -xzvf ./tts-framework/datasets_cache/LIBRITTS/train_clean_360.tar.gz -C ./tts-framework/datasets_cache/LIBRITTS/
mv ./tts-framework/datasets_cache/LIBRITTS/LibriTTS_R/train-clean-360 ./tts-framework/datasets_cache/LIBRITTS/LibriTTS
rm -r ./tts-framework/datasets_cache/LIBRITTS/LibriTTS_R

conda env create -f ./tts-framework/environment.yml
conda activate tts_framework

cp ./tts-framework/training/scripts/train_dist.py ./tts-framework

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib

# And then check the torch version and cuda version
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.backends.cudnn.version())"
