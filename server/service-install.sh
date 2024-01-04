sudo apt-get install -y ffmpeg libavcodec-extra
conda env create -f environment.yml
sudo cp tts.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start tts.service
sudo systemctl enable tts.service