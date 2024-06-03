# %%
import base64
import io
import tempfile

from IPython.core.display import HTML
from IPython.display import Audio
import librosa
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
import soundfile as sf
import torch

from models import DelightfulHiFi
from models.config import PreprocessingConfig
from training.datasets.hifi_libri_dataset import selected_speakers_ids
from training.preprocess import TacotronSTFT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_rate = 44100

# %%
# Path to the checkpoints

delightful_checkpoint_path = "checkpoints/epoch=414-step=85176.ckpt"
hifi_checkpoint_path = "checkpoints/epoch=7-step=40160.ckpt"

# Load the model
model = DelightfulHiFi(
    delightful_checkpoint_path=delightful_checkpoint_path,
    hifi_checkpoint_path=hifi_checkpoint_path,
)

preprocess_config = PreprocessingConfig("english_only")
tacotronSTFT = TacotronSTFT(
    filter_length=preprocess_config.stft.filter_length,
    hop_length=preprocess_config.stft.hop_length,
    win_length=preprocess_config.stft.win_length,
    n_mel_channels=preprocess_config.stft.n_mel_channels,
    sampling_rate=preprocess_config.sampling_rate,
    mel_fmin=preprocess_config.stft.mel_fmin,
    mel_fmax=preprocess_config.stft.mel_fmax,
    center=False,
)
tacotronSTFT = tacotronSTFT.to(device)


# %%
def plot_spectrogram(mel: np.ndarray):
    r"""Plots the mel spectrogram."""
    plt.figure(dpi=80, figsize=(10, 3))

    img = librosa.display.specshow(mel, x_axis="time", y_axis="mel", sr=sample_rate)
    plt.title("Spectrogram")
    plt.colorbar(img, format="%+2.0f dB")

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # Convert the BytesIO object to a base64 string
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    return img_str


# %%
text_tts = """
As the snake shook its head, a deafening shout behind Harry made both of them jump.
‘DUDLEY! MR DURSLEY! COME AND LOOK AT THIS SNAKE! YOU WON’T BELIEVE WHAT IT’S DOING!’
"How did you know it was me?" she asked.
"My dear Professor, I’ve never seen a cat sit so stiffly."
"You’d be stiff if you’d been sitting on a brick wall all day," said Professor McGonagall.
"""

html = f"""<table border='1'>
<h4>TTS: </h4> {text_tts}
<h4>Speakers: </h4>
<tr>
    <th>SpeakerID</th>
    <th>Audio</th>
    <th>Mel</th>
</tr>
"""


for speaker_id in selected_speakers_ids.values():
    with torch.no_grad():
        speaker_id_ = torch.tensor([int(speaker_id)], device=device)
        wav = model.forward(text_tts, speaker_id_)

        mel = tacotronSTFT.get_mel_from_wav(wav)
        mel_base64 = plot_spectrogram(mel.detach().cpu().numpy())

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as wav_file:
            sf.write(wav_file.name, wav, sample_rate)
            # Convert wav to mp3
            audio = AudioSegment.from_wav(wav_file.name)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as mp3_file:
                audio.export(mp3_file.name, format="mp3")
                mp3_base64 = base64.b64encode(mp3_file.read()).decode("utf-8")

            # Add a row to the HTML table
            html += f"""<tr>
                <td>{speaker_id}</td>
                <td><audio controls><source src="data:audio/mp3;base64,{mp3_base64}"></audio></td>
                <td><img src='data:image/png;base64,{mel_base64}' /></td>
            </tr>"""

# Save result as HTML
with open("logs/demo.html", "w") as f:
    f.write(html)

HTML(html)
# %%
