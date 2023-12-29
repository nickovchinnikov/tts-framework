FROM condaforge/mambaforge:latest

# Set the working directory
WORKDIR /app

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y gcc

# Copy your entry point file
COPY ./config/ /app/config
COPY ./model/ /app/model
COPY ./server/ /app/server
COPY ./training/ /app/training
COPY ./web_server.py /app/web_server.py

# Copy env file
COPY environment.yml /app/environment.yml

# Copy checkpoints
COPY ./epoch=5155-step=524125.ckpt /app/checkpoints/epoch=5155-step=524125.ckpt
COPY ./vocoder.ckpt /app/checkpoints/vocoder.ckpt
COPY ./en_us_cmudict_ipa_forward.pt /app/checkpoints/en_us_cmudict_ipa_forward.pt

# Create environment
RUN mamba env create -f environment.yml

# Expose a port if needed
EXPOSE 8000

# Set the entry point
CMD ["conda", "run", "-n", "tts_framework", "python", "web_server.py"]
