import os

from pydantic import BaseModel, Field

from pydub import AudioSegment

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
checkpoint = "checkpoints/epoch=5482-step=601951.ckpt"

device_cpu = torch.device("cpu")
module_cpu = AcousticModule.load_from_checkpoint(checkpoint).to(device_cpu)
# Set the module to eval mode
module_cpu.eval()


device_gpu = torch.device("cuda")
module_gpu = AcousticModule.load_from_checkpoint(checkpoint).to(device_gpu)
# Set the module to eval mode
module_gpu.eval()


# Load the speaker information
existed_speakers = speakers_info()

BIT_RATE = 128  # kbps
SAMPLING_RATE = 22050
AUDIO_FORMAT = "mp3"

# FrameSize = 144 * BitRate / (SampleRate + Padding)
FRAME_SIZE = int((144 * BIT_RATE) / (SAMPLING_RATE * 0.001))


class TransformerParams(BaseModel):
    text: str
    speaker: str = Field(default="122") # Default speaker "carnright"


async def async_gen(text: str, speaker: torch.Tensor, module: AcousticModule):
    """Audio streaming async generator function."""
    try:
        sentences = sentences_split(text)

        for text in sentences:
            if text.strip() == "": continue
            with torch.no_grad():
                wav_prediction = module(
                    text,
                    speaker,
                )

                buffer_ = returnAudioBuffer(
                    wav_prediction.detach().cpu(),
                    SAMPLING_RATE,
                    AUDIO_FORMAT
                )

                audio = AudioSegment.from_file(buffer_, format=AUDIO_FORMAT)

                # Stream the audio in frame-sized chunks
                frame_size = FRAME_SIZE  # This is the frame size for MP3 at 22.05kHz and 128 kbps. Adjust as necessary.
                for i in range(0, len(audio), frame_size):
                    yield audio[i:i+frame_size].raw_data

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
            int(params.speaker)
        ], device=device_cpu)

        return StreamingResponse(
            async_gen(params.text, speaker, module_cpu),
            media_type=f"audio/{AUDIO_FORMAT}",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_cuda/")
def generate_cuda(params: TransformerParams):
    try:
        if params.speaker not in existed_speakers:
            raise HTTPException(status_code=400, detail="Speaker not found")

        speaker = torch.tensor([
            int(params.speaker)
        ], device=device_gpu)

        return StreamingResponse(
            async_gen(params.text, speaker, module_gpu),
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
