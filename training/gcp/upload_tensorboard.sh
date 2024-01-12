# install deps
pip install google-cloud-aiplatform[tensorboard]

tb-gcp-uploader --tensorboard_resource_name=tts-latest-training --logdir=./logs --experiment_name=prod-train --one_shot=True