# Look at the:
# https://gist.github.com/nickovchinnikov/93058db2c647621fb48974ac8902c20b

# Update the system
sudo apt update -y && sudo apt upgrade -y

# ubuntu-drivers devices

sudo apt install ubuntu-drivers-common -y

sudo ubuntu-drivers autoinstall

# Reboot the system and then run the following commands
sudo apt install nvidia-cuda-toolkit -y

sudo apt-get install gcc -y

curl -O https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh

bash ./Anaconda3-2023.09-0-Linux-x86_64.sh

# Install jupyterlab
conda install -c conda-forge jupyterlab

# Install gcsfuse
# export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
# echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list

# curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc

# sudo apt-get update -y & sudo apt-get install gcsfuse -y

# Install gsutil
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

sudo apt-get update && sudo apt-get install google-cloud-cli

# Manual steps
# Init gcloud
# gcloud init

# Run jupyter lab in the background
# nohup jupyter lab --ip=0.0.0.0 --no-browser --port=8888 &

# Tensorboard
# tensorboard --logdir ./logs --host 0.0.0.0 --port 6006