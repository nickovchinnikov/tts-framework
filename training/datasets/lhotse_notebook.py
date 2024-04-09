# %%
import os
from pathlib import Path
from pprint import pprint

from lhotse.recipes import (
    download_voxceleb1,
    download_voxceleb2,
    hifitts,
    libritts,
    prepare_voxceleb,
)
import pandas as pd

# %%
root_dir = Path("../../datasets_cache")
# root_dir = Path("datasets_cache")

voxceleb1_path = root_dir / "voxceleb1"
voxceleb2_path = root_dir / "voxceleb2"

hifitts_path = root_dir / "hifitts"
libritts_path = root_dir / "librittsr"

num_jobs = os.cpu_count() - 3  # type: ignore

num_jobs, hifitts_path

# %%
# voxceleb1_root = download_voxceleb1(voxceleb1_path)
# voxceleb1_root

# %%
# voxceleb2_root = download_voxceleb2(voxceleb2_path)
# voxceleb2_root

# %%
hifitts_root = hifitts.download_hifitts(hifitts_path)
hifitts_root


# %%
result = hifitts.prepare_hifitts(hifitts_root, num_jobs=num_jobs)
result


# %%
result.keys()

# %%
from lhotse import CutSet, Fbank, FbankConfig, Mfcc, MfccConfig, RecordingSet

cuts_train = CutSet.from_manifests(**result["6670_other_test"])  # type: ignore
cuts_train

# %%
pprint(cuts_train[0])

# %%
from lhotse.cut import Cut

# Filter the CutSet to only include cuts that are no more than the duration limit
duration_limit_min = 2.0
duration_limit_max = 2.5
# Duration limit in seconds
cuts_train = cuts_train.filter(
    lambda cut: isinstance(cut, Cut)
    and cut.duration >= duration_limit_min
    and cut.duration <= duration_limit_max,
)
cuts_train

# %%
cuts_train[0].supervisions[0]

# %%
# filter_length=2048,
# hop_length=512,  # NOTE: 441 ?? https://github.com/jik876/hifi-gan/issues/116#issuecomment-1436999858
# win_length=2048,
# n_mel_channels=128,
# mel_fmin=20,
# mel_fmax=11025,

fbank = Fbank(
    FbankConfig(
        sampling_rate=44100,
        num_filters=128,
    ),
)

cuts_train_fbank = cuts_train.compute_and_store_features(
    extractor=fbank,
    storage_path=hifitts_root / "features",
    num_jobs=1,
)

cuts_train_fbank

# %%
# cuts_train_fbank.to_file(hifitts_root / "cuts_train.json.gz")

# %%
cuts_train_fbank[0].plot_features()

# %%
cuts_train_fbank_item = cuts_train_fbank[0]
cuts_train_fbank_item

# %%
from lhotse.cut import MonoCut

if isinstance(cuts_train_fbank_item, MonoCut):
    print(cuts_train_fbank_item.features)

# %%
cuts_train_fbank_item.plot_audio()

# %%
cuts_train_fbank_item.play_audio()

# %%
from lhotse import CutSet
from lhotse.dataset import (
    SimpleCutSampler,
    UnsupervisedDataset,
    UnsupervisedWaveformDataset,
)
from torch.utils.data import DataLoader, Dataset

dataset = UnsupervisedDataset()
sampler = SimpleCutSampler(cuts_train_fbank, max_duration=300)

dataloader = DataLoader(dataset, sampler=sampler, batch_size=None)

batch = next(iter(dataloader))
batch


# %%
batch["cuts"][0].recording.sources[0].load_audio().shape

# %%
batch["cuts"][0].features

# %%
batch["features"][0].shape

# %%
batch["features"][0]

# %%
# Prepare the LibriTTS dataset
libritts_root = libritts.download_librittsr(
    libritts_path,
    dataset_parts=["train-clean-100"],
)
libritts_root, libritts_path

# %%
prepared_libri = libritts.prepare_librittsr(
    libritts_root / "LibriTTS_R",
    # dataset_parts=["dev-clean"],
    dataset_parts=["train-clean-100"],
    num_jobs=num_jobs,
)

# %%
prepared_libri

# %%
prepared_libri_100 = (
    pd.DataFrame(prepared_libri["train-clean-100"]["supervisions"])
    .groupby("speaker")["duration"]
    .sum()
    .sort_values(ascending=False)
)
prepared_libri_100

# %%
for k in prepared_libri:
    prepared_libri_ = (
        pd.DataFrame(prepared_libri[k]["supervisions"])
        .groupby("speaker")["duration"]
        .sum()
        .sort_values(ascending=False)
    )
    print(prepared_libri_.loc[prepared_libri_ >= 1800])

# %%
from lhotse import CutSet, SupervisionSet

supervisions_libri = SupervisionSet()

supervisions_libri.to_file(libritts_root / "supervisions_libri.json.gz")

# dev-clean
# Series([], Name: duration, dtype: float64)
# dev-other
# Series([], Name: duration, dtype: float64)
# test-clean
# speaker
# 3570    1865.052667
# Name: duration, dtype: float64
# test-other
# Series([], Name: duration, dtype: float64)
# train-clean-100
# speaker
# 40      2096.569333
# 6209    1926.765000
# 7447    1915.213333
# 1088    1900.926000
# Name: duration, dtype: float64
# train-clean-360
# speaker
# 3003    2385.213333
# 2204    2242.730333
# 3307    2086.246500
# 8080    2051.131500
# 5935    1959.650833
# 3922    1938.523500
# 7982    1893.050833
# 3638    1843.324000
# 3032    1812.692000
# Name: duration, dtype: float64
# train-other-500
# speaker
# 215     2385.047833
# 6594    2341.286667
# 3433    2206.806500
# 3867    2118.326167
# 5733    2097.689833
# 7649    2016.925500
# 2834    2008.083000
# 8291    1977.892000
# 483     1964.766000
# 5181    1959.280000
# 8799    1909.690500
# 7839    1888.650500
# 1665    1877.726833
# 8430    1872.845500
# 47      1861.966167
# 2361    1839.646333
# 1132    1838.686333
# 5439    1837.487000
# 3319    1821.083833
# 5445    1808.444667
# 2208    1804.525833
# 8346    1804.405500
# Name: duration, dtype: float64

selected_speakers_man = [
    # train-clean-100
    "40",
    "1088",
    # train-clean-360
    "3307",
    "5935",
    "3032",
    # train-other-500
    "215",
    "6594",
    "3867",
    "5733",
    "8291",
    "5181",
    "8799",
    "2361",
    "1132",
    "5439",
    "3319",
    "8346",
]


# %%
num_speakers_lib_100_over_1900_sec = prepared_libri_100.loc[prepared_libri_100 >= 1900]
num_speakers_lib_100_over_1900_sec

# %%
prepared_libri_360 = libritts.prepare_librittsr(
    libritts_root / "LibriTTS_R",
    # dataset_parts=["dev-clean"],
    dataset_parts=["train-clean-360"],
    num_jobs=num_jobs,
)

# %%
speaker_durations_360 = (
    pd.DataFrame(prepared_libri_360["train-clean-360"]["supervisions"])
    .groupby("speaker")["duration"]
    .sum()
    .sort_values(ascending=False)
)
speaker_durations_360

# %%
# Get the speaker IDs from both dataframes
speaker_ids_100 = prepared_libri_100.index
speaker_ids_360 = speaker_durations_360.index

# Find the intersection of the speaker IDs
common_speaker_ids = speaker_ids_100.intersection(speaker_ids_360)

# No intersection!
common_speaker_ids

# %%
num_speakers_lib_360_over_1900_sec = speaker_durations_360.loc[
    speaker_durations_360 > 1900
].count()
num_speakers_lib_360_over_1900_sec

# %%
from lhotse import CutSet, Fbank, FbankConfig

cuts_train = CutSet.from_manifests(**prepared_libri["train-clean-100"])  # type: ignore
cuts_train

# %%
# You can save the prepared CutSet to a file!
cuts_train.to_file("./libri_selected.json.gz")

cuts_train.to_file(root_dir / "./libri_selected.json.gz")

# %%
from lhotse import CutSet, SupervisionSet

libri_selected = CutSet.from_file(root_dir / "libri.json.gz")
libri_selected

# %%

pprint(libri_selected[0])

print(libri_selected[0].recording.sources[0].source)

# %%
libri_selected[0].play_audio()

# %%
import torchaudio

torchaudio.load(
    "datasets_cache/librittsr/LibriTTS_R/dev-clean/5694/64025/5694_64025_000017_000002.wav",
)

# %%
supervisions_libri = SupervisionSet.from_file(
    root_dir / "supervisions_libri.json.gz",
)
recordings_libri = RecordingSet.from_file(
    root_dir / "recordings_libri.json.gz",
)
supervisions_libri, recordings_libri


# %%
supervisions_libri[0]

# %%
speakers_dur = (
    pd.DataFrame(supervisions_libri)
    .groupby("speaker")["duration"]
    .sum()
    .sort_values(ascending=False)
)

# %%
speakers_dur_1900 = speakers_dur.loc[speakers_dur >= 1900]
speakers_dur_1900

# %%
# selected_1900_ids = set(
#     map(int, speakers_dur_1900.index.to_list()),
# )

selected_1900_ids = set(
    speakers_dur_1900.index.to_list(),
)
selected_1900_ids

# %%
duration_limit_min = 0.5
duration_limit_max = 35.0

libri_selected.filter(
    lambda cut: isinstance(cut, Cut)
    and cut.supervisions[0].speaker in selected_1900_ids
    and cut.duration >= duration_limit_min
    and cut.duration <= duration_limit_max,
)

# %%
libri_selected[0]

# %%
cuts_train_frame = pd.DataFrame(cuts_train)
cuts_train_frame

# %%
cuts_train[0].supervisions[0].speaker

# %%
# duration_limit_min = 2.0
# duration_limit_max = 2.5

cuts_train = cuts_train.filter(
    lambda cut: isinstance(cut, Cut) and cut.supervisions[0].speaker == "5338",
    # and cut.duration >= duration_limit_min
    # and cut.duration <= duration_limit_max,
)

cuts_train

# %%
# cuts_train.map(lambda cut: cut.supervisions[0].speaker)
# %%
cuts_train[0]

# %%
len(cuts_train)

# %%
selected_speakers_libri_ids = [
    # train-clean-100
    40,
    1088,
    # train-clean-360
    3307,
    5935,
    3032,
    # train-other-500
    215,
    6594,
    3867,
    5733,
    8291,
    5181,
    8799,
    2361,
    1132,
    5439,
    3319,
    8346,
]

# The selected speakers from the HiFiTTS dataset
selected_speakers_hi_fi_ids = [
    92,
    6670,
    6671,
    6097,
    8051,
    11614,
    11697,
    9017,
    12787,
    9136,
]

selected_speakers_ids = {
    v: k
    for k, v in enumerate(
        selected_speakers_libri_ids + selected_speakers_hi_fi_ids,
    )
}

selected_speakers_ids[1088]

# %%

selected_speakers_libri_ids = [
    # train-clean-100
    40,
    1088,
    # train-clean-360
    3307,
    5935,
    3032,
    # train-other-500
    215,
    6594,
    3867,
    5733,
    8291,
    5181,
    8799,
    2361,
    1132,
    5439,
    3319,
    8346,
]

# The selected speakers from the HiFiTTS dataset
selected_speakers_hi_fi_ids = [
    "Cori Samuel",  # 92,
    "Phil Benson",  # 6097,
    "Mike Pelton",  # 6670,
    "Tony Oliva",  # 6671,
    "Maria Kasper",  # 8051,
    "John Van Stan",  # 9017,
    "Helen Taylor",  # 9136,
    "Sylviamb",  # 11614,
    "Celine Major",  # 11697,
    "LikeManyWaters",  # 12787,
]

# Map the speaker ids to string and list of selected speaker ids to set
selected_speakers_ids = {
    v: k
    for k, v in enumerate(
        selected_speakers_libri_ids + selected_speakers_hi_fi_ids,
    )
}

selected_speakers_ids, len(selected_speakers_ids)

# %%
