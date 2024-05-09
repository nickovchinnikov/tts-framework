from gradio import Dropdown, Interface, Textbox
import torch

from models.delightful_hifi import DelightfulHiFi
from training.datasets.hifi_libri_dataset import speakers_hifi_ids

# Load the pretrained weights from the checkpoint
delightful_checkpoint_path = "checkpoints/logs_44100_tts_80_logs_new3_lightning_logs_version_7_checkpoints_epoch=2450-step=183470.ckpt"
hifi_checkpoint_path = (
    "checkpoints/logs_44100_vocoder_Mel44100_WAV44100_epoch=19-step=44480.ckpt"
)

device = torch.device("cuda")

module = DelightfulHiFi(
    delightful_checkpoint_path=delightful_checkpoint_path,
    hifi_checkpoint_path=hifi_checkpoint_path,
).to(device)

speakers_hifi_speakers = {spk: idx for idx, spk in enumerate(speakers_hifi_ids)}


def generate_audio(text: str, speaker_id: str):
    speaker = torch.tensor([speakers_hifi_speakers[speaker_id]], device=device)
    with torch.no_grad():
        wav_vf = module.forward(text, speaker)
        wav_vf = wav_vf.squeeze().detach().cpu().numpy()
    return module.sampling_rate, wav_vf


demo = Interface(
    generate_audio,
    [
        Textbox(
            label="Text",
            value="As the snake shook its head, a deafening shout behind Harry made both of them jump.",
        ),
        Dropdown(
            label="Speaker",
            choices=list(speakers_hifi_speakers.keys()),
            value=speakers_hifi_ids[0],
        ),
    ],
    outputs="audio",
)
demo.launch()
