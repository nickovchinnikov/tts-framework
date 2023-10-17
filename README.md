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

```

By default, conda will export your environment with builds, but builds can be platform-specific.
A solution that worked for me is to use the `--no-build` flag:

```
conda env export --no-build > environment.yml
```

Create an env
```
conda env create -f environment.yml
```

If you have troubles with export, like:
```
InvalidVersionSpec: Invalid version '3.0<3.3': invalid character(s)                                                           
```

Find a problem by this way:

```
cd /mnt/Data/anaconda3/envs/tts_framework/lib/python3.11/site-packages/

grep -Rnw . -e "3.0<3.3"

```

A Faster Solver for Conda: [Libmamba](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community)


Generate docs:


```
# live preview server
mkdocs serve

# build a static site from your Markdown files
mkdocs build
```

Test cases:

```
python -m unittest discover -v
```