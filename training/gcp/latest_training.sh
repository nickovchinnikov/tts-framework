gsutil cp gs://tts-training-bucket/cache-hifitts-librittsr.tar.gz ./datasets_cache/
gsutil cp gs://tts-training-bucket/hifi.json.gz ./datasets_cache/
gsutil cp gs://tts-training-bucket/libri.json.gz ./datasets_cache/

# Unpack the cache files
tar -xzvf /dev/shm/cache-hifitts-librittsr.tar.gz -C /dev/shm/

# Manual steps
conda activate tts_framework

pip instal lhotse

# Download the datasets with selected.ipynb

mkdir logs_new
