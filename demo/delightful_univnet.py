from gradio import Dropdown, Interface, Textbox
import torch

from models.delightful_univnet import DelightfulUnivnet
from training.datasets.hifi_libri_dataset import speakers_hifi_ids

from .config import speakers_delightful_22050

delightful_checkpoint_path = "checkpoints/epoch=5816-step=390418.ckpt"

device = torch.device("cpu")

delightfulunivnet_22050 = DelightfulUnivnet(
    delightful_checkpoint_path=delightful_checkpoint_path,
).to(device)


def generate_audio(text: str, speaker_name: str):
    speaker = torch.tensor(
        [speakers_delightful_22050[speaker_name]],
        device=device,
    )
    with torch.no_grad():
        wav_vf = delightfulunivnet_22050.forward(text, speaker)
        wav_vf = wav_vf.squeeze().detach().cpu().numpy()

    return delightfulunivnet_22050.sampling_rate, wav_vf


interfaceDelightfulUnuvnet22050 = Interface(
    generate_audio,
    [
        Textbox(
            label="Text",
            value="As the snake shook its head, a deafening shout behind Harry made both of them jump.",
        ),
        Dropdown(
            label="Speaker",
            choices=list(speakers_delightful_22050.keys()),
            value=speakers_hifi_ids[0],
        ),
    ],
    outputs="audio",
    title=f"Delightful UnivNet, Sampling Rate: {delightfulunivnet_22050.sampling_rate}",
)
