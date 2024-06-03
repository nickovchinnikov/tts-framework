# TTS-Framework

Modified version of DelightfulTTS and UnivNet

## Install deps

```bash
sudo apt install ffmpeg libasound2-dev build-essential espeak-ng -y
```

Create env from the `environment.yml` file:

```bash
conda env create -f ./tts-framework/environment.yml python=3.11

# After the setup
conda activate tts_framework
```

## Generate docs:

```
# live preview server
mkdocs serve

# build a static site from your Markdown files
mkdocs build
```

## Test cases:

```
python -m unittest discover -v
```
