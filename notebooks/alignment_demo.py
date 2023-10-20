# %%
from datasets import load_dataset
import IPython
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# load pretrained model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


# %%
librispeech_samples_ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
librispeech_samples_ds

# %%
librispeech_samples_ds[1]["file"] # type: ignore

# %%
# load audio
audio_input, sample_rate = sf.read(librispeech_samples_ds[1]["file"]) # type: ignore

audio_input, sample_rate

# %%

IPython.display.Audio(audio_input, rate=sample_rate) # type: ignore

# %%
# pad input values and return pt tensor
input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values

input_values

# %%
# INFERENCE

# retrieve logits & take argmax
logits = model(input_values).logits # type: ignore
predicted_ids = torch.argmax(logits, dim=-1)

logits, predicted_ids

# %%
# transcribe
transcription = processor.decode(predicted_ids[0])
transcription


# %%
# FINE-TUNE

target_transcription = "NOR IS MISTER QUILTER'S MANNER LESS INTERESTING THAN HIS MATTER"
with processor.as_target_processor():
    labels = processor(target_transcription, return_tensors="pt")
labels

# %%

# encode labels
with processor.as_target_processor():
  labels = processor(target_transcription, return_tensors="pt").input_ids
labels


# %%
# encode labels
with processor.as_target_processor():
  labels = processor(target_transcription, return_tensors="pt").input_ids

# compute loss by passing labels
loss = model(input_values, labels=labels).loss  # type: ignore
loss.backward()

# %%
loss

# %%

# %%

# %%

# %%

# %%
