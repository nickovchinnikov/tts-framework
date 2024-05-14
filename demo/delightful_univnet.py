import tempfile

from gradio import Checkbox, Dropdown, Interface, Textbox
import soundfile as sf
import torch
from voicefixer import VoiceFixer

from models.delightful_univnet import DelightfulUnivnet
from training.datasets.hifi_libri_dataset import speakers_hifi_ids

from .config import speakers_delightful_22050

delightful_checkpoint_path = "checkpoints/epoch=5816-step=390418.ckpt"

device = torch.device("cpu")

delightfulunivnet_22050 = DelightfulUnivnet(
    delightful_checkpoint_path=delightful_checkpoint_path,
).to(device)

voicefixer = VoiceFixer()


def generate_audio(text: str, speaker_name: str, fix_voice: bool):
    speaker = torch.tensor(
        [speakers_delightful_22050[speaker_name]],
        device=device,
    )
    with torch.no_grad():
        wav = delightfulunivnet_22050.forward(text, speaker)
        wav = wav.squeeze().detach().cpu().numpy()

    if fix_voice:
        # Save the numpy array to a temporary wav file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as input_file:
            # Write to the temp wav file
            sf.write(input_file.name, wav, delightfulunivnet_22050.sampling_rate)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as output_file:
                voicefixer.restore(
                    input=input_file.name,  # low quality .wav/.flac file
                    output=output_file.name,  # save file path
                    cuda=False,  # GPU acceleration off
                    mode=0,
                )

                # Read the wav file back into a numpy array
                wav_vf, sampling_rate = sf.read(output_file.name)

                return sampling_rate, wav_vf

    return delightfulunivnet_22050.sampling_rate, wav


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
        Checkbox(
            label="Fix voice (Voicefixer)",
            value=False,
        ),
    ],
    outputs="audio",
    title=f"Delightful UnivNet, Sampling Rate: {delightfulunivnet_22050.sampling_rate}. When Voicefixer is enabled, the Simpling Rate is 44100.",
)
