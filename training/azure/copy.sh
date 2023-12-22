# Description: Copy the code from cloud to local
rsync -avP --progress --checksum --exclude="datasets_cache" --exclude="notebooks" --exclude="__pycache__" --exclude="docs" --exclude="logs" --exclude=".ipynb_checkpoints" ./tts-framework/ ~/localfiles/tts-framework/ 

# Copy the datasets_cache from cloud to local (if needed)
rsync -avP --progress ~/cloudfiles/code/Users/nick/tts-framework/datasets_cache/** ./datasets_cache/

# Restart the nvidia driver if failed
# It won't fix the problem with cuda memory allocation, only restart the VM will fix it
# modprobe nvidia