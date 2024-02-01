#!/bin/bash

# !Need to make this file executable!
# sudo chmod a+x setting.sh

# CUDA drivers
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb 
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Update the system
sudo apt update && sudo apt upgrade -y

# espeak-ng
sudo apt-get install espeak-ng -y

# Install CUDA drivers
sudo apt-get -y install cuda-drivers -y
sudo apt install nvidia-cuda-toolkit -y

# Remove all existing environments
eval "$(conda shell.bash hook)"

declare -a arr=("azureml_py38" "azureml_py310_sdkv2" "azureml_py38_PT_TF" "jupyter_env")

for env in "${arr[@]}"
do
    conda remove --name "${env}" --all -y
    conda create -n "${env}" python=3.8 -y
    conda activate "${env}"
    conda install jupyterlab -y
done

conda clean --all -y

# Setup the tts-train environment
conda create -n tts-train python=3.11 -y
conda activate tts-train

conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# pip install --upgrade --force-reinstall --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121/torch.html

pip install numba numpy scikit-learn tqdm

pip install pynini datasets chardet deep-phonemizer nemo-text-processing piq soundfile transformers unidecode tensorboard librosa gpustat

pip install lightning

pip install tensorflow tensorboard

pip install jupyterlab-nvdashboard

# If you are using Jupyter Lab 2 you will also need to run
jupyter labextension install jupyterlab-nvdashboard

# Install gsutil
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

sudo apt-get update && sudo apt-get install google-cloud-cli

# Init gcloud
gcloud init
