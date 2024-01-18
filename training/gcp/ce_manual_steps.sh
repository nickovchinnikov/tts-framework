# Create instance with this command:
gcloud compute instances create prod-train \
    --project=voiceservice-217021 \
    --zone=us-central1-a \
    --machine-type=a2-ultragpu-8g \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --service-account=default \
    --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append \
    --accelerator=count=8,type=nvidia-a100-80gb \
    --tags=http-server,https-server \
    --create-disk=auto-delete=yes,boot=yes,device-name=prod-train,image=projects/ml-images/global/images/c0-deeplearning-common-cu121-v20231209-debian-11,mode=rw,size=100,type=projects/voiceservice-217021/zones/us-central1-a/diskTypes/pd-ssd \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=any

# Do ce_manual_steps.sh first, then run this script
# Install Cloud Storage FUSE
# export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`

# echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list

# curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# sudo apt-get install fuse gcsfuse -y

# # Check the gcsfuse version
# gcsfuse -v

##########################################

# Update the system
# sudo apt update && sudo apt upgrade -y

# Install CUDA drivers
# sudo apt -y install cuda-drivers
# sudo apt -y install nvidia-cuda-toolkit


# Install git
# sudo apt install git -y

# # Clone the repo
# git clone git@github.com:nickovchinnikov/tts-framework.git

# Do this before running the script ce_settings.sh
# Install miniconda
# curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# bash Miniconda3-latest-Linux-x86_64.sh

# # bash Miniconda3-latest-Linux-x86_64.sh
# source ~/.bashrc

# # Do the gcould init (auth)
# gcloud init

# Create a ssh key and add it to the github
ssh-keygen -t rsa -f ~/.ssh/id_rsa -q -N ""

# Add the key to the github and add the github to the known hosts
ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts

###############################################
# Mount the disc
###############################################

# List the disks
# lsblk -o NAME,HCTL,SIZE,MOUNTPOINT | grep nvme

# Check the devices
ls -l /dev/disk/by-id/google-*

# Choose and format the disk
sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/disk/by-id/google-local-nvme-ssd-0

# Mount the disk
# mount point
sudo mkdir -p /mnt/disks/training-disk
# mount the disk
sudo mount -o discard,defaults /dev/disk/by-id/google-local-nvme-ssd-0 /mnt/disks/training-disk
# Change the owner
sudo chmod a+w /mnt/disks/training-disk


# Configure automatic mounting on VM restart
sudo cp /etc/fstab /etc/fstab.backup
# Use the blkid command to list the UUID for the disk.
sudo blkid /dev/disk/by-id/google-local-nvme-ssd-0


# Example:
# /dev/disk/by-id/google-local-nvme-ssd-0: UUID="35c0d7f9-bf93-4dce-8dd9-fe2b5ba8e867" BLOCK_SIZE="4096" TYPE="ext4"
# Change the /etc/fstab file to mount the disk automatically after a VM restart.
###############################################
# cat /etc/fstab:
# UUID=35c0d7f9-bf93-4dce-8dd9-fe2b5ba8e867 /mnt/disks/training-disk ext4 discard,defaults 0 2

# Choose the mounted disk
# Change the directory to the mounted disk
cd /mnt/disks/training-disk

# Clone the repo
git clone git@github.com:nickovchinnikov/tts-framework.git

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib

# After the setup
pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.backends.cudnn.version())"
