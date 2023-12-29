import os

from pydantic import BaseModel, Field

import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn

import torch

import nltk
nltk.download('punkt')

from training.modules.acoustic_module import AcousticModule

from server.utils import speakers_info, returnAudioBuffer, sentences_split, audio_package


os.environ["PYTHONIOENCODING"] = "utf8"
os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"


# Load the pretrained weights from the checkpoint
checkpoint = "checkpoints/epoch=5155-step=524125.ckpt"
module = AcousticModule.load_from_checkpoint(checkpoint)

# Set the module to eval mode
module.eval()


# Load the speaker information
existed_speakers = speakers_info()

SAMPLING_RATE = 22050
AUDIO_FORMAT = "mpeg"


class TransformerParams(BaseModel):
    text: str
    speaker: str = Field(default="122") # Default speaker "carnright"


async def async_gen(text: str, speaker: str):
    """Audio streaming async generator function."""
    try:
        sentences = sentences_split(text)

        for text in sentences:
            if text.strip() == "": continue
            with torch.no_grad():
                wav_prediction = module(
                    text,
                    torch.tensor([
                        int(speaker)
                    ]),
                )

                buffer_ = returnAudioBuffer(
                    wav_prediction.detach().cpu(),
                    SAMPLING_RATE,
                    AUDIO_FORMAT
                )
                yield buffer_.read()
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

        return StreamingResponse(
            async_gen(params.text, params.speaker),
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
