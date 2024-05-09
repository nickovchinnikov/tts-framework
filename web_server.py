import asyncio
import io
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import nltk
from pydantic import BaseModel, Field
from pydub import AudioSegment
import torch
import uvicorn

from config.best_speakers_list import selected_speakers
from models.delightful_hifi import DelightfulHiFi
from server.utils import (
    returnAudioBuffer,
    sentences_split,
)

nltk.download("punkt")

os.environ["PYTHONIOENCODING"] = "utf8"
os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"


# Load the pretrained weights from the checkpoint
delightful_checkpoint_path = "checkpoints/epoch=3265-step=313888.ckpt"
hifi_checkpoint_path = "checkpoints/hifi-gan_v1_universal_v1.ckpt"

device = torch.device("cuda")

module = DelightfulHiFi(
    delightful_checkpoint_path=delightful_checkpoint_path,
    hifi_checkpoint_path=hifi_checkpoint_path,
).to(
    device,
)


# Load the speaker information
existed_speakers = set(selected_speakers)

BIT_RATE = 320
SAMPLING_RATE = 22050
AUDIO_FORMAT = "mp3"

FRAME_SIZE = int((144 * BIT_RATE) / (SAMPLING_RATE * 0.001))


class TransformerParams(BaseModel):
    text: str
    speaker: str = Field(default="122")  # Default speaker "carnright"


# async def async_gen(text: str, speaker: torch.Tensor):
#     """Audio streaming async generator function."""
#     try:
#         paragraphs = sentences_split(text)

#         # Create an empty buffer to hold all the audio data
#         total_buffer = io.BytesIO()

#         for paragraph in paragraphs:
#             if paragraph.strip() == "":
#                 continue
#             with torch.no_grad():
#                 wav_prediction, wav_vf = module.forward(
#                     paragraph,
#                     speaker,
#                 )

#                 wav_vf = wav_vf.detach().cpu().numpy()

#                 buffer_ = returnAudioBuffer(
#                     wav_vf,
#                     44100,
#                     AUDIO_FORMAT,
#                 )

#                 # Append the buffer to the total buffer
#                 total_buffer.write(buffer_.getvalue())

#         total_buffer.seek(0)

#         audio = AudioSegment.from_file(
#             total_buffer,
#             format=AUDIO_FORMAT,
#         )

#         audio_bytes = audio.export(format=AUDIO_FORMAT, bitrate="64k")

#         yield audio_bytes.read()

#         # need this sleep in order to be able to catch the disconnect
#         await asyncio.sleep(0)

#     except asyncio.CancelledError:
#         print("Client disconnected")


async def async_gen(text: str, speaker: torch.Tensor):
    """Audio streaming async generator function."""
    try:
        # paragraphs = sentences_split(text)

        # # Create an empty buffer to hold all the audio data
        total_buffer = io.BytesIO()

        # for paragraph in paragraphs:
        #     if paragraph.strip() == "":
        #         continue
        with torch.no_grad():
            _, wav_vf = module.forward(
                text,
                speaker,
            )

            wav_vf = wav_vf.detach().cpu().numpy()

            total_buffer = returnAudioBuffer(
                wav_vf,
                44100,
                AUDIO_FORMAT,
            )

            # # Append the buffer to the total buffer
            # total_buffer.write(buffer_.getvalue())

        total_buffer.seek(0)

        audio = AudioSegment.from_file(
            total_buffer,
            format=AUDIO_FORMAT,
        )

        audio_bytes = audio.export(format=AUDIO_FORMAT, bitrate="64k")

        yield audio_bytes.read()

        # need this sleep in order to be able to catch the disconnect
        await asyncio.sleep(0)

    except asyncio.CancelledError:
        print("Client disconnected")


app = FastAPI()


@app.get("/")
def read_root():
    return {
        "version": "0.0.1",
        "message": "Welcome to the TTS server!",
        "existed_speakers": existed_speakers,
    }


@app.post("/generate/")
def generate(params: TransformerParams):
    try:
        speaker_id = int(params.speaker)

        if speaker_id not in existed_speakers:
            raise HTTPException(status_code=400, detail="Speaker not found")

        speaker = torch.tensor([speaker_id], device=device)

        return StreamingResponse(
            async_gen(params.text, speaker),
            media_type=f"audio/{AUDIO_FORMAT}",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "web_server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
