gsutil cp gs://tts-training-bucket/cache-hifitts-librittsr.tar.gz ./datasets_cache/
gsutil cp gs://tts-training-bucket/hifi.json.gz ./datasets_cache/
gsutil cp gs://tts-training-bucket/libri.json.gz ./datasets_cache/

cp datasets_cache/hifi.json.gz /dev/shm/
cp datasets_cache/libri.json.gz /dev/shm/
tar -xzvf datasets_cache/cache-hifitts-librittsr.tar.gz -C /dev/shm/

# Unpack the cache files
tar -xzvf /dev/shm/cache-hifitts-librittsr.tar.gz -C /dev/shm/

# Manual steps
conda activate tts_framework

pip instal lhotse

# Download the datasets with selected.ipynb

mkdir logs_new
