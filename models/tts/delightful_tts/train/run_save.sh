/home/oaiw/anaconda3/envs/tts_framework/bin/python train_dist.py
if [ $? -ne 0 ]; then
    echo "An error occurred while running train_dist.py"
    sudo shutdown -h now
fi