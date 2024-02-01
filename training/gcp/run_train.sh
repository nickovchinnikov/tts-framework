# for pid in $(pgrep -f python); do
#     if ! ps -p $pid -o args= | grep "jupyter-lab"; then
#         kill -9 $pid
#     fi
# done

# conda run -n tts_framework python ./tts-framework/train_dist.py

# Kill the training process
pkill -9 -f "train_dist.py"
