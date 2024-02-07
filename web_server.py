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

from server.utils import (
    returnAudioBuffer,
    sentences_split,
    speakers_info,
)
from training.modules.acoustic_module import AcousticModule

nltk.download("punkt")

os.environ["PYTHONIOENCODING"] = "utf8"
os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"


# Load the pretrained weights from the checkpoint
checkpoint = "checkpoints/epoch=5482-step=601951.ckpt"


device = torch.device("cuda")
module = AcousticModule.load_from_checkpoint(checkpoint).to(device)
# Set the module to eval mode
module.eval()


# Load the speaker information
existed_speakers = speakers_info()

BIT_RATE = 320
SAMPLING_RATE = 22050
AUDIO_FORMAT = "mp3"

FRAME_SIZE = int((144 * BIT_RATE) / (SAMPLING_RATE * 0.001))


class TransformerParams(BaseModel):
    text: str
    speaker: str = Field(default="122") # Default speaker "carnright"


async def async_gen(text: str, speaker: torch.Tensor):
    """Audio streaming async generator function."""
    try:
        paragraphs = sentences_split(text)

        # Create an empty buffer to hold all the audio data
        total_buffer = io.BytesIO()

        for paragraph in paragraphs:
            if paragraph.strip() == "": continue
            with torch.no_grad():
                wav_prediction = module(
                    paragraph,
                    speaker,
                ).detach().cpu().numpy()

                buffer_ = returnAudioBuffer(
                    wav_prediction,
                    SAMPLING_RATE,
                    AUDIO_FORMAT,
                )

                # Append the buffer to the total buffer
                total_buffer.write(buffer_.getvalue())

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
        if params.speaker not in existed_speakers:
            raise HTTPException(status_code=400, detail="Speaker not found")

        speaker = torch.tensor([
            int(params.speaker),
        ], device=device)

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
