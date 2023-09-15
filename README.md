# TTS-Framework
Modified version of DelightfulTTS and UnivNet

### Conda env

Create / activate env

```
conda create --name tts_framework python=3.11
conda activate tts_framework
```

Export / import env

```
conda env export > environment.yml
conda env create -f environment.yml
```

Generate docs:


```
# live preview server
mkdocs serve

# build a static site from your Markdown files
mkdocs build
```