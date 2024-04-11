from dataclasses import asdict, dataclass
import os
from pathlib import Path
import tempfile
from typing import Dict, List, Literal, Tuple

from lhotse import CutSet, RecordingSet, SupervisionSet
from lhotse.cut import MonoCut
from lhotse.recipes import hifitts, libritts
import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from voicefixer import VoiceFixer

from models.config import PreprocessingConfig, get_lang_map, lang2id
from training.preprocess import PreprocessLibriTTS
from training.tools import pad_1D, pad_2D, pad_3D

NUM_JOBS = (os.cpu_count() or 2) - 1


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

# The selected speakers from the LibriTTS dataset
selected_speakers_libri_ids = list(
    map(
        str,
        [
            # train-clean-100
            40,
            1088,
            # train-clean-360
            3307,
            5935,
            # train-other-500
            215,
            6594,
            3867,
            5733,
            5181,
        ],
    ),
)

# Map the speaker ids to string and list of selected speaker ids to set
selected_speakers_ids = {
    v: k
    for k, v in enumerate(
        selected_speakers_hi_fi_ids + selected_speakers_libri_ids,
    )
}


def prep_2_cutset(prep: Dict[str, Dict[str, RecordingSet | SupervisionSet]]) -> CutSet:
    r"""Prepare the dataset for the model. This function is used to convert the prepared dataset to a CutSet.

    Args:
        prep (Dict[str, Dict[str, RecordingSet | SupervisionSet]]): The prepared dataset.

    Returns:
        CutSet: The dataset prepared for the model.
    """
    recordings_hifi = RecordingSet()
    supervisions_hifi = SupervisionSet()

    for hifi_row in prep.values():
        record = hifi_row["recordings"]
        supervision = hifi_row["supervisions"]

        # Separate the recordings and supervisions
        if isinstance(record, RecordingSet):
            recordings_hifi += record

        if isinstance(supervision, SupervisionSet):
            supervisions_hifi += supervision

    # Add the recordings and supervisions to the CutSet
    return CutSet.from_manifests(
        recordings=recordings_hifi,
        supervisions=supervisions_hifi,
    )


DATASET_TYPES = Literal["hifitts", "libritts"]


@dataclass
class HifiLibriItem:
    """Dataset row for the HiFiTTS and LibriTTS datasets combined in this code.

    Args:
        id (str): The ID of the item.
        wav (Tensor): The waveform of the audio.
        mel (Tensor): The mel spectrogram.
        pitch (Tensor): The pitch.
        text (Tensor): The text.
        attn_prior (Tensor): The attention prior.
        energy (Tensor): The energy.
        raw_text (str): The raw text.
        normalized_text (str): The normalized text.
        speaker (int): The speaker ID.
        pitch_is_normalized (bool): Whether the pitch is normalized.
        lang (int): The language ID.
        dataset_type (DATASET_TYPES): The type of dataset.
    """

    id: str
    wav: Tensor
    mel: Tensor
    pitch: Tensor
    text: Tensor
    attn_prior: Tensor
    energy: Tensor
    raw_text: str
    normalized_text: str
    speaker: int
    pitch_is_normalized: bool
    lang: int
    dataset_type: DATASET_TYPES


class HifiLibriDataset(Dataset):
    r"""A PyTorch dataset for loading delightful TTS data."""

    def __init__(
        self,
        lang: str = "en",
        root: str = "datasets_cache",
        sampling_rate: int = 44100,
        hifitts_path: str = "hifitts",
        hifi_cutset_file_name: str = "hifi.json.gz",
        libritts_path: str = "librittsr",
        libritts_cutset_file_name: str = "libri.json.gz",
        libritts_subsets: List[str] | str = "all",
        cache: bool = False,
        cache_dir: str = "/dev/shm",
        num_jobs: int = NUM_JOBS,
    ):
        r"""Initializes the dataset.

        Args:
            lang (str, optional): The language of the dataset. Defaults to "en".
            root (str, optional): The root directory of the dataset. Defaults to "datasets_cache".
            sampling_rate (int, optional): The sampling rate of the audio. Defaults to 44100.
            hifitts_path (str, optional): The path to the HiFiTTS dataset. Defaults to "hifitts".
            hifi_cutset_file_name (str, optional): The file name of the HiFiTTS cutset. Defaults to "hifi.json.gz".
            libritts_path (str, optional): The path to the LibriTTS dataset. Defaults to "librittsr".
            libritts_cutset_file_name (str, optional): The file name of the LibriTTS cutset. Defaults to "libri.json.gz".
            libritts_subsets (Union[List[str], str], optional): The subsets of the LibriTTS dataset to use. Defaults to "all".
            cache (bool, optional): Whether to cache the dataset. Defaults to False.
            cache_dir (str, optional): The directory to cache the dataset in. Defaults to "/dev/shm".
            num_jobs (int, optional): The number of jobs to use for preparing the dataset. Defaults to NUM_JOBS.
        """
        lang_map = get_lang_map(lang)
        processing_lang_type = lang_map.processing_lang_type
        preprocess_config = PreprocessingConfig(
            processing_lang_type,
            sampling_rate=sampling_rate,
        )
        self.preprocess_libtts = PreprocessLibriTTS(
            lang,
            preprocess_config,
        )
        self.root_dir = Path(root)
        self.voicefixer = VoiceFixer()

        # Map the speaker ids to string and list of selected speaker ids to set
        self.selected_speakers_libri_ids_ = set(selected_speakers_libri_ids)

        self.cache = cache
        self.cache_dir = Path(cache_dir) / f"cache-{hifitts_path}-{libritts_path}"

        # Prepare the HiFiTTS dataset
        self.hifitts_path = self.root_dir / hifitts_path
        hifi_cutset_file_path = self.root_dir / hifi_cutset_file_name

        # Check if the HiFiTTS dataset has been prepared
        if hifi_cutset_file_path.exists():
            self.cutset_hifi = CutSet.from_file(hifi_cutset_file_path)
        else:
            hifitts_root = hifitts.download_hifitts(self.hifitts_path)
            prepared_hifi = hifitts.prepare_hifitts(
                hifitts_root,
                num_jobs=num_jobs,
            )

            # Add the recordings and supervisions to the CutSet
            self.cutset_hifi = prep_2_cutset(prepared_hifi)
            # Save the prepared HiFiTTS dataset cutset
            self.cutset_hifi.to_file(hifi_cutset_file_path)

        # Prepare the LibriTTS dataset
        self.libritts_path = self.root_dir / libritts_path
        libritts_cutset_file_path = self.root_dir / libritts_cutset_file_name

        # Check if the LibriTTS dataset has been prepared
        if libritts_cutset_file_path.exists():
            self.cutset_libri = CutSet.from_file(libritts_cutset_file_path)
        else:
            libritts_root = libritts.download_librittsr(
                self.libritts_path,
                dataset_parts=libritts_subsets,
            )
            prepared_libri = libritts.prepare_librittsr(
                libritts_root / "LibriTTS_R",
                dataset_parts=libritts_subsets,
                num_jobs=num_jobs,
            )

            # Add the recordings and supervisions to the CutSet
            self.cutset_libri = prep_2_cutset(prepared_libri)
            # Save the prepared cutset for LibriTTS
            self.cutset_libri.to_file(libritts_cutset_file_path)

        # Filter the libri cutset to only include the selected speakers
        self.cutset_libri = self.cutset_libri.filter(
            lambda cut: isinstance(cut, MonoCut)
            and str(cut.supervisions[0].speaker) in self.selected_speakers_libri_ids_,
        )

        # Final cutset for the dataset
        # to_eager() is used to evaluates all lazy operations on this manifest
        self.cutset = (
            (self.cutset_hifi + self.cutset_libri)
            .filter(
                lambda cut: isinstance(cut, MonoCut)
                and cut.duration >= preprocess_config.min_seconds
                and cut.duration <= preprocess_config.max_seconds,
            )
            .to_eager()
        )

    def get_cache_subdir_path(self, idx: int) -> Path:
        r"""Calculate the path to the cache subdirectory.

        Args:
            idx (int): The index of the cache subdirectory.

        Returns:
            Path: The path to the cache subdirectory.
        """
        return self.cache_dir / str(((idx // 1000) + 1) * 1000)

    def get_cache_file_path(self, idx: int) -> Path:
        r"""Calculate the path to the cache file.

        Args:
            idx (int): The index of the cache file.

        Returns:
            Path: The path to the cache file.
        """
        return self.get_cache_subdir_path(idx) / f"{idx}.pt"

    def __len__(self) -> int:
        r"""Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.cutset)

    def __getitem__(self, idx: int) -> HifiLibriItem:
        r"""Returns the item at the specified index.

        Args:
            idx (int): The index of the item.

        Returns:
            HifiLibriItem: The item at the specified index.
        """
        cache_file = self.get_cache_file_path(idx)

        if self.cache and cache_file.exists():
            cached_data: Dict = torch.load(cache_file)
            # Cast the cached data to the PreprocessForAcousticResult class
            result = HifiLibriItem(**cached_data)
            return result

        cutset = self.cutset[idx]

        if isinstance(cutset, MonoCut) and cutset.recording is not None:
            dataset_speaker_id = str(cutset.supervisions[0].speaker)

            # Map the dataset speaker id to the speaker id in the model
            speaker_id = selected_speakers_ids.get(
                dataset_speaker_id,
                len(selected_speakers_ids) + 1,
            )

            # Run voicefixer only for the libri speakers
            if str(dataset_speaker_id) in self.selected_speakers_libri_ids_:
                audio_path = cutset.recording.sources[0].source
                # Restore LibriTTS-R audio
                with tempfile.NamedTemporaryFile(
                    suffix=".wav",
                    delete=True,
                ) as out_file:
                    self.voicefixer.restore(
                        input=audio_path,  # low quality .wav/.flac file
                        output=out_file.name,  # save file path
                        cuda=True,  # GPU acceleration
                        mode=0,
                    )
                    audio, _ = sf.read(out_file.name)
                    # Convert the np audio to a tensor
                    audio = torch.from_numpy(audio).float().unsqueeze(0)
            else:
                # Load the audio from the cutset
                audio = torch.from_numpy(cutset.load_audio())

            text: str = str(cutset.supervisions[0].text)

            fileid = str(cutset.supervisions[0].recording_id)

            # _, chapter_id, _, utterance_id = fileid.split("_")
            split_fileid = fileid.split("_")
            chapter_id = split_fileid[1]
            utterance_id = split_fileid[-1]

            libri_row = (
                audio,
                cutset.sampling_rate,
                text,
                text,
                speaker_id,
                chapter_id,
                utterance_id,
            )
            data = self.preprocess_libtts.acoustic(libri_row)

            if data is None:
                rand_idx = int(
                    torch.randint(
                        0,
                        self.__len__(),
                        (1,),
                    ).item(),
                )
                return self.__getitem__(rand_idx)

            data.wav = data.wav.unsqueeze(0)

            result = HifiLibriItem(
                id=data.utterance_id,
                wav=data.wav,
                mel=data.mel,
                pitch=data.pitch,
                text=data.phones,
                attn_prior=data.attn_prior,
                energy=data.energy,
                raw_text=data.raw_text,
                normalized_text=data.normalized_text,
                speaker=speaker_id,
                pitch_is_normalized=data.pitch_is_normalized,
                lang=lang2id["en"],
                dataset_type="hifitts" if idx < len(self.cutset_hifi) else "libritts",
            )

            if self.cache:
                # Create the cache subdirectory if it doesn't exist
                Path.mkdir(
                    self.get_cache_subdir_path(idx),
                    parents=True,
                    exist_ok=True,
                )
                # Save the preprocessed data to the cache
                torch.save(asdict(result), cache_file)

            return result
        else:
            raise FileNotFoundError(f"Cut not found at index {idx}.")

    def __iter__(self):
        r"""Method makes the class iterable. It iterates over the `_walker` attribute
        and for each item, it gets the corresponding item from the dataset using the
        `__getitem__` method.

        Yields:
        The item from the dataset corresponding to the current item in `_walker`.
        """
        for item in range(self.__len__()):
            yield self.__getitem__(item)

    def collate_fn(self, data: List[HifiLibriItem]) -> List:
        r"""Collates a batch of data samples.

        Args:
            data (List[HifiLibriItem]): A list of data samples.

        Returns:
            List: A list of reprocessed data batches.
        """
        data_size = len(data)

        idxs = list(range(data_size))

        # Initialize empty lists to store extracted values
        empty_lists: List[List] = [[] for _ in range(12)]
        (
            ids,
            speakers,
            texts,
            raw_texts,
            mels,
            pitches,
            attn_priors,
            langs,
            src_lens,
            mel_lens,
            wavs,
            energy,
        ) = empty_lists

        # Extract fields from data dictionary and populate the lists
        for idx in idxs:
            data_entry = data[idx]
            ids.append(data_entry.id)
            speakers.append(data_entry.speaker)
            texts.append(data_entry.text)
            raw_texts.append(data_entry.raw_text)
            mels.append(data_entry.mel)
            pitches.append(data_entry.pitch)
            attn_priors.append(data_entry.attn_prior)
            langs.append(data_entry.lang)
            src_lens.append(data_entry.text.shape[0])
            mel_lens.append(data_entry.mel.shape[1])
            wavs.append(data_entry.wav)
            energy.append(data_entry.energy)

        # Convert langs, src_lens, and mel_lens to numpy arrays
        langs = np.array(langs)
        src_lens = np.array(src_lens)
        mel_lens = np.array(mel_lens)

        # NOTE: Instead of the pitches for the whole dataset, used stat for the batch
        # Take only min and max values for pitch
        pitches_stat = list(self.normalize_pitch(pitches)[:2])

        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        attn_priors = pad_3D(attn_priors, len(idxs), max(src_lens), max(mel_lens))

        speakers = np.repeat(
            np.expand_dims(np.array(speakers), axis=1),
            texts.shape[1],
            axis=1,
        )
        langs = np.repeat(
            np.expand_dims(np.array(langs), axis=1),
            texts.shape[1],
            axis=1,
        )

        wavs = pad_2D(wavs)
        energy = pad_2D(energy)

        return [
            ids,
            raw_texts,
            torch.from_numpy(speakers),
            texts.int(),
            torch.from_numpy(src_lens),
            mels,
            pitches,
            pitches_stat,
            torch.from_numpy(mel_lens),
            torch.from_numpy(langs),
            attn_priors,
            wavs,
            energy,
        ]

    def normalize_pitch(
        self,
        pitches: List[torch.Tensor],
    ) -> Tuple[float, float, float, float]:
        r"""Normalizes the pitch values.

        Args:
            pitches (List[torch.Tensor]): A list of pitch values.

        Returns:
            Tuple: A tuple containing the normalized pitch values.
        """
        pitches_t = torch.concatenate(pitches)

        min_value = torch.min(pitches_t).item()
        max_value = torch.max(pitches_t).item()

        mean = torch.mean(pitches_t).item()
        std = torch.std(pitches_t).item()

        return min_value, max_value, mean, std


def train_dataloader(
    batch_size: int = 6,
    num_workers: int = 5,
    shuffle: bool = False,
    lang: str = "en",
    root: str = "datasets_cache",
    hifitts_path: str = "hifitts",
    hifi_cutset_file_name: str = "hifi.json.gz",
    libritts_path: str = "librittsr",
    libritts_cutset_file_name: str = "libri.json.gz",
    libritts_subsets: List[str] | str = "all",
    cache: bool = False,
    cache_dir: str = "/dev/shm",
) -> DataLoader:
    r"""Returns the training dataloader, that is using the HifiLibriDataset dataset.

    Args:
        batch_size (int): The batch size.
        num_workers (int): The number of workers.
        shuffle (bool): Whether to shuffle the dataset.
        lang (str): The language of the dataset.
        root (str): The root directory of the dataset.
        hifitts_path (str): The path to the HiFiTTS dataset.
        hifi_cutset_file_name (str): The file name of the HiFiTTS cutset.
        libritts_path (str): The path to the LibriTTS dataset.
        libritts_cutset_file_name (str): The file name of the LibriTTS cutset.
        libritts_subsets (List[str] | str): The subsets of the LibriTTS dataset to use.
        cache (bool): Whether to cache the dataset.
        cache_dir (str): The directory to cache the dataset in.

    Returns:
        DataLoader: The training dataloader.
    """
    dataset = HifiLibriDataset(
        root=root,
        hifitts_path=hifitts_path,
        hifi_cutset_file_name=hifi_cutset_file_name,
        libritts_path=libritts_path,
        libritts_cutset_file_name=libritts_cutset_file_name,
        libritts_subsets=libritts_subsets,
        cache=cache,
        cache_dir=cache_dir,
        lang=lang,
    )

    train_loader = DataLoader(
        dataset,
        # 4x80Gb max 10 sec audio
        # batch_size=20, # self.train_config.batch_size,
        # 4*80Gb max ~20.4 sec audio
        batch_size=batch_size,
        # TODO: find the optimal num_workers
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
    )

    return train_loader
