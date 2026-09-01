# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


from transformers import Wav2Vec2Model
from typing import Dict, List

import librosa
import torchaudio
import torchaudio.transforms as T
from torchaudio.models.conformer import Conformer

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs


import contextlib
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import ast
from typing import Dict, List
import os
import random
import math
import time

# %%
# ? this is for local training

MODEL_PATH = "../../models/final_best_checkpoint.pth"
WORKING_MODEL_PATH = "../../models/checkpoint.pth"
WORKING_BEST_MODEL_PATH = "../../models/best_checkpoint.pth"

DATASET_PATH = "../../datasets/Quran_ds/Quran_ds/audio/audio/"
DATASET_PATH_1 = "../../datasets/Quran_ds/Quran_ds/audio/audio/"
TRAIN_DS_PATH = "../../datasets/Quran_ds/train_df.csv"
TEST_DS_PATH = "../../datasets/Quran_ds/val_df.csv"


# ? this is for Kaggle training

# MODEL_PATH = "/kaggle/input/datasets/muhammadbannan/quran-ds-v4/best_checkpoint.pth"
# WORKING_MODEL_PATH = "/kaggle/working/checkpoint.pth"
# WORKING_BEST_MODEL_PATH = "/kaggle/working/best_checkpoint.pth"

# DATASET_PATH = "/kaggle/input/datasets/omartariq612/quran-reciters/audio/audio/"
# DATASET_PATH_1 = "/kaggle/input/datasets/abdo3id/female-quran-recitation"

# TRAIN_DS_PATH = "/kaggle/input/datasets/mohammeddeebjalab1/quran-ds-csv-files/quran_train.csv"
# TEST_DS_PATH = "/kaggle/input/datasets/mohammeddeebjalab1/quran-ds-csv-files/quran_test.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1
SR = 16000 
NUM_EPOCHS = 8

DEVICE

# %% [markdown]
# # ============================================================
# # Tajweed-Aware Phoneme Vocabulary
# # ============================================================
# #
# # Defines the phoneme set used by the model:
# # - Arabic consonants
# # - short/long vowels and Tanween
# # - consonant-vowel combinations
# # - special Tajweed tokens for Ikhfaa, Qalqalah and Ghunnah
# #
# # Examples:
# #   مَ  -> ma
# #   مَا -> maa
# #   qK  -> Qalqalah form of ق
# #   nn  -> prolonged / Ghunnah-related ن
# #
# # PHONEME_BASE_TO_ARABIC is used only for readable Arabic display.
# #
# # Each phoneme is mapped to an integer ID using phoneme_to_id.
# # CTC reserves ID 0 for <blank>, so normal phoneme IDs are shifted
# # by +1 during CTC training and shifted back after decoding.
# # ============================================================

# %%
SILENT_TOKEN = "<sil>"

IKFAA_LETTERS = [
    "sˤ",
    "ð",
    "θ",
    "k",
    "j",
    "ʃ",
    "q",
    "s",
    "d",
    "tˤ",
    "z",
    "f",
    "t",
    "dˤ",
    "ðˤ",
]

QALQALAA_LETTERS = ["q", "tˤ", "b", "j", "d"]

SHORT_VOWELS: List[str] = ["a", "i", "u"]
LONG_VOWELS: List[str] = ["aa", "ii", "uu"]
TANWEEN: List[str] = ["an", "in", "un"]

SPECIAL_PHONEMES_FOR_TAJWEED = ["n" + l for l in IKFAA_LETTERS]
SPECIAL_PHONEMES_FOR_TAJWEED += ["an" + l for l in IKFAA_LETTERS]
SPECIAL_PHONEMES_FOR_TAJWEED += ["in" + l for l in IKFAA_LETTERS]
SPECIAL_PHONEMES_FOR_TAJWEED += ["un" + l for l in IKFAA_LETTERS]
SPECIAL_PHONEMES_FOR_TAJWEED += [l + "K" for l in QALQALAA_LETTERS]
SPECIAL_PHONEMES_FOR_TAJWEED += ["nn", "mm", "yy", "ww"]


BASE_CONSONANTS: List[str] = [
    "ʔ",
    "b",
    "t",
    "θ",
    "j",
    "ħ",
    "x",
    "d",
    "ð",
    "r",
    "rM",
    "z",
    "s",
    "ʃ",
    "sˤ",
    "dˤ",
    "tˤ",
    "ðˤ",
    "ʕ",
    "ɣ",
    "f",
    "q",
    "k",
    "l",
    "m",
    "n",
    "h",
    "w",
    "y",
    "T",
]

PHONEME_BASE_TO_ARABIC = {
    "ʔ": "ء",
    "b": "ب",
    "t": "ت",
    "θ": "ث",
    "j": "ج",
    "ħ": "ح",
    "x": "خ",
    "d": "د",
    "ð": "ذ",
    "r": "ر",
    "rM": "ر",
    "z": "ز",
    "s": "س",
    "ʃ": "ش",
    "sˤ": "ص",
    "dˤ": "ض",
    "tˤ": "ط",
    "ðˤ": "ظ",
    "ʕ": "ع",
    "ɣ": "غ",
    "f": "ف",
    "q": "ق",
    "k": "ك",
    "l": "ل",
    "m": "م",
    "n": "ن",
    "h": "ه",
    "w": "و",
    "y": "ي",
    "T": "ة",

    "<sil>": "صمت",
}



# Generate all CV combinations
CV_TOKENS = [
    c + v for c in BASE_CONSONANTS for v in SHORT_VOWELS + LONG_VOWELS + TANWEEN
]

# ================================
# All phonemes (for tokenization)
# ================================

PHONEMES: List[str] = (
    [
        SILENT_TOKEN,
    ]
    + SPECIAL_PHONEMES_FOR_TAJWEED
    + BASE_CONSONANTS
    + SHORT_VOWELS
    + LONG_VOWELS
    + TANWEEN
    + CV_TOKENS
)


PHONEMES_CTC: List[str] =  PHONEMES


phoneme_to_id: Dict[str, int] = {p: i for i, p in enumerate(PHONEMES_CTC)}
id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}

CTC_BLANK_ID = 0
CTC_VOCAB_SIZE = len(phoneme_to_id) + 1

# %%
len(phoneme_to_id)

# %% [markdown]
# # ============================================================
# # Audio Loading and Data Augmentation
# # ============================================================
# #
# # This block prepares raw audio before passing it to the model.
# #
# # Main steps:
# # - Load audio with torchaudio, with librosa as a fallback.
# # - Convert stereo audio to mono and resample to 16 kHz.
# # - Validate the waveform and reject empty/corrupted audio.
# #
# # During training only, random augmentation is applied:
# #   • Speed perturbation   -> simulates slightly faster/slower recitation.
# #   • Reverberation        -> simulates room, mosque, or distant microphone acoustics.
# #   • Gaussian noise       -> adds controlled noise using a selected SNR.
# #   • Random gain          -> changes recording loudness.
# #
# # Example:
# #
# #   Original audio
# #        ↓
# #   Speed change
# #        ↓
# #   Room reverb
# #        ↓
# #   Background noise
# #        ↓
# #   Normalize + random gain
# #        ↓
# #   Final waveform (16 kHz)
# #
# # These augmentations increase speaker/environment variation and help
# # the model generalize to recordings that were not seen during training.
# # ============================================================

# %%
@contextlib.contextmanager
def suppress_c_stderr():
    """Redirect C-level stderr to /dev/null — catches libmpg123 warnings."""
    with open(os.devnull, "w") as devnull:
        old_fd = os.dup(2)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_fd, 2)
            os.close(old_fd)


# Speed perturbation
SPEED_PROBABILITY = 0.25

SPEED_FACTORS = (
    0.95,  # slower and slightly longer
    1.05,  # faster and slightly shorter
)

NORMALIZED_PEAK = 0.8

# Random gain
GAIN_PROBABILITY = 0.50
MIN_GAIN = 0.8
MAX_GAIN = 1.2

# SNR-based noise
NOISE_PROBABILITY = 0.35
MIN_SNR_DB = 15.0
MAX_SNR_DB = 35.0

# Mild reverberation
REVERB_PROBABILITY = 0.25

REVERB_PROFILES = {
    "ordinary_room": {
        "rt60_range": (
            0.20,
            0.45,
        ),
        "wet_mix_range": (
            0.08,
            0.16,
        ),
        "reflection_delay_ms": (
            8.0,
            45.0,
        ),
        "reflection_count": (
            3,
            6,
        ),
    },

    "mosque": {
        "rt60_range": (
            0.80,
            1.40,
        ),
        "wet_mix_range": (
            0.08,
            0.18,
        ),
        "reflection_delay_ms": (
            20.0,
            120.0,
        ),
        "reflection_count": (
            6,
            12,
        ),
    },

    "distant_microphone": {
        "rt60_range": (
            0.35,
            0.80,
        ),
        "wet_mix_range": (
            0.12,
            0.22,
        ),
        "reflection_delay_ms": (
            12.0,
            80.0,
        ),
        "reflection_count": (
            5,
            9,
        ),
    },
}

def get_audio_duration(file_path):
    info = torchaudio.info(file_path)
    duration = info.num_frames / info.sample_rate
    return duration

@torch.no_grad()
def apply_speed_perturbation(
    signal,
    sample_rate,
    speed_factor=None,
):
    """
    Change waveform speed while keeping the waveform
    usable at the model sample rate.

    speed_factor > 1.0:
        Faster and shorter.

    speed_factor < 1.0:
        Slower and longer.
    """

    signal = np.asarray(
        signal,
        dtype=np.float32,
    )

    if signal.ndim != 1:
        raise ValueError(
            "Speed perturbation expects a mono "
            "one-dimensional waveform"
        )

    if len(signal) == 0:
        raise ValueError(
            "Cannot apply speed perturbation "
            "to an empty waveform"
        )

    if speed_factor is None:
        speed_factor = float(
            np.random.choice(
                SPEED_FACTORS
            )
        )

    speed_factor = float(
        speed_factor
    )

    if speed_factor <= 0:
        raise ValueError(
            "speed_factor must be greater "
            "than zero"
        )

    waveform = torch.from_numpy(
        signal
    ).unsqueeze(0)

    input_lengths = torch.tensor(
        [waveform.shape[-1]],
        dtype=torch.long,
    )

    (
        perturbed_waveform,
        perturbed_lengths,
    ) = torchaudio.functional.speed(
        waveform=waveform,
        orig_freq=sample_rate,
        factor=speed_factor,
        lengths=input_lengths,
    )

    valid_length = int(
        perturbed_lengths[0].item()
    )

    perturbed_waveform = (
        perturbed_waveform[
            0,
            :valid_length,
        ]
        .contiguous()
        .numpy()
        .astype(np.float32)
    )

    if len(perturbed_waveform) == 0:
        raise RuntimeError(
            "Speed perturbation produced "
            "an empty waveform"
        )

    if not np.all(
        np.isfinite(
            perturbed_waveform
        )
    ):
        raise RuntimeError(
            "Speed perturbation produced "
            "non-finite waveform values"
        )

    return perturbed_waveform

def add_noise_at_snr(
    signal,
    snr_db=None,
    min_snr_db=MIN_SNR_DB,
    max_snr_db=MAX_SNR_DB,
):
    """
    Add zero-mean Gaussian noise at a selected SNR.

    SNR is measured using RMS amplitudes:

        SNR(dB) = 20 * log10(
            signal_rms / noise_rms
        )
    """

    signal = np.asarray(
        signal,
        dtype=np.float32,
    )

    if snr_db is None:
        snr_db = np.random.uniform(
            min_snr_db,
            max_snr_db,
        )

    signal_rms = float(
        np.sqrt(
            np.mean(
                signal.astype(
                    np.float64
                ) ** 2
            )
            + 1e-12
        )
    )

    if signal_rms < 1e-8:
        return signal.copy()

    noise = np.random.randn(
        signal.shape[0]
    ).astype(
        np.float32
    )

    noise_rms = float(
        np.sqrt(
            np.mean(
                noise.astype(
                    np.float64
                ) ** 2
            )
            + 1e-12
        )
    )

    desired_noise_rms = (
        signal_rms
        / (
            10.0
            ** (
                float(snr_db)
                / 20.0
            )
        )
    )

    noise_scale = (
        desired_noise_rms
        / max(
            noise_rms,
            1e-8,
        )
    )

    noisy_signal = (
        signal
        + noise * noise_scale
    )

    return noisy_signal.astype(
        np.float32
    )


def random_gain(
    signal,
    min_gain=MIN_GAIN,
    max_gain=MAX_GAIN,
):
    gain = np.random.uniform(
        min_gain,
        max_gain,
    )

    return (
        signal * gain
    ).astype(
        np.float32
    )

def fft_convolve_same_length(
    signal,
    impulse_response,
):
    """
    FFT convolution while retaining the original
    waveform length.
    """

    signal = np.asarray(
        signal,
        dtype=np.float32,
    )

    impulse_response = np.asarray(
        impulse_response,
        dtype=np.float32,
    )

    full_length = (
        len(signal)
        + len(impulse_response)
        - 1
    )

    fft_length = (
        1
        << (
            full_length - 1
        ).bit_length()
    )

    signal_fft = np.fft.rfft(
        signal,
        n=fft_length,
    )

    impulse_fft = np.fft.rfft(
        impulse_response,
        n=fft_length,
    )

    convolved = np.fft.irfft(
        signal_fft * impulse_fft,
        n=fft_length,
    )

    convolved = convolved[
        :len(signal)
    ]

    return convolved.astype(
        np.float32
    )

def create_synthetic_rir(
    sample_rate,
    profile_name,
):
    """
    Build a mild synthetic room impulse response
    containing early reflections and a decaying tail.
    """

    if (
        profile_name
        not in REVERB_PROFILES
    ):
        raise ValueError(
            f"Unknown reverb profile: "
            f"{profile_name}"
        )

    profile = REVERB_PROFILES[
        profile_name
    ]

    rt60 = np.random.uniform(
        *profile[
            "rt60_range"
        ]
    )

    wet_mix = np.random.uniform(
        *profile[
            "wet_mix_range"
        ]
    )

    minimum_rir_length = int(
        0.05 * sample_rate
    )

    rir_length = max(
        int(
            rt60 * sample_rate
        ),
        minimum_rir_length,
    )

    time = (
        np.arange(
            rir_length,
            dtype=np.float32,
        )
        / float(sample_rate)
    )

    # Amplitude falls to approximately 0.001
    # at t = RT60, corresponding to -60 dB.
    decay = np.power(
        10.0,
        -3.0
        * time
        / max(
            rt60,
            1e-3,
        ),
    ).astype(
        np.float32
    )

    # Low-level diffuse late-reverberation tail.
    late_tail = np.random.randn(
        rir_length
    ).astype(
        np.float32
    )

    # Mild smoothing prevents an excessively
    # harsh white-noise-like reverb.
    smoothing_kernel = np.array(
        [
            0.20,
            0.60,
            0.20,
        ],
        dtype=np.float32,
    )

    late_tail = np.convolve(
        late_tail,
        smoothing_kernel,
        mode="same",
    ).astype(
        np.float32
    )

    impulse_response = (
        0.03
        * late_tail
        * decay
    )

    # Keep the first few milliseconds empty.
    # The dry signal is added separately.
    initial_silence = min(
        int(
            0.005
            * sample_rate
        ),
        rir_length,
    )

    impulse_response[
        :initial_silence
    ] = 0.0

    minimum_reflections, maximum_reflections = (
        profile[
            "reflection_count"
        ]
    )

    reflection_count = np.random.randint(
        minimum_reflections,
        maximum_reflections + 1,
    )

    minimum_delay, maximum_delay = (
        profile[
            "reflection_delay_ms"
        ]
    )

    for _ in range(
        reflection_count
    ):
        delay_ms = np.random.uniform(
            minimum_delay,
            maximum_delay,
        )

        delay_sample = int(
            delay_ms
            * sample_rate
            / 1000.0
        )

        delay_sample = min(
            delay_sample,
            rir_length - 1,
        )

        reflection_amplitude = (
            np.random.uniform(
                0.15,
                0.65,
            )
            * decay[
                delay_sample
            ]
        )

        # Reflections may undergo phase inversion.
        if np.random.rand() < 0.5:
            reflection_amplitude *= -1.0

        impulse_response[
            delay_sample
        ] += reflection_amplitude

    rir_energy = float(
        np.sqrt(
            np.sum(
                impulse_response.astype(
                    np.float64
                ) ** 2
            )
            + 1e-12
        )
    )

    if rir_energy < 1e-8:
        raise RuntimeError(
            "Generated impulse response "
            "has zero energy"
        )

    impulse_response = (
        impulse_response
        / rir_energy
    )

    return (
        impulse_response.astype(
            np.float32
        ),
        float(wet_mix),
    )

def apply_mild_reverb(
    signal,
    sample_rate,
    profile_name=None,
):
    signal = np.asarray(
        signal,
        dtype=np.float32,
    )

    if profile_name is None:
        profile_name = np.random.choice(
            [
                "ordinary_room",
                "mosque",
                "distant_microphone",
            ],
            p=[
                0.50,
                0.25,
                0.25,
            ],
        )

    (
        impulse_response,
        wet_mix,
    ) = create_synthetic_rir(
        sample_rate=sample_rate,
        profile_name=profile_name,
    )

    wet_signal = fft_convolve_same_length(
        signal=signal,
        impulse_response=(
            impulse_response
        ),
    )

    dry_rms = float(
        np.sqrt(
            np.mean(
                signal.astype(
                    np.float64
                ) ** 2
            )
            + 1e-12
        )
    )

    wet_rms = float(
        np.sqrt(
            np.mean(
                wet_signal.astype(
                    np.float64
                ) ** 2
            )
            + 1e-12
        )
    )

    if (
        dry_rms > 1e-8
        and wet_rms > 1e-8
    ):
        wet_signal = (
            wet_signal
            * (
                dry_rms
                / wet_rms
            )
        )

    reverberated_signal = (
        (
            1.0 - wet_mix
        )
        * signal
        + wet_mix
        * wet_signal
    )

    return reverberated_signal.astype(
        np.float32
    )

def load_waveform(
    audio_path,
    sr=16000,
    training=True,
):
    signal = None

    audio_path = audio_path.replace(".mp3", ".wav")


    # ============================================================
    # 1. Load using torchaudio
    # ============================================================

    try:
        waveform, original_sample_rate = (
            torchaudio.load(
                audio_path
            )
        )

        if waveform.shape[0] > 1:
            waveform = waveform.mean(
                dim=0,
                keepdim=True,
            )

        if original_sample_rate != sr:
            waveform = T.Resample(
                orig_freq=(
                    original_sample_rate
                ),
                new_freq=sr,
            )(waveform)

        signal = (
            waveform
            .squeeze(0)
            .numpy()
        )

    # ============================================================
    # 2. Fallback to librosa
    # ============================================================

    except Exception:
        try:
            with suppress_c_stderr():
                signal, _ = librosa.load(
                    audio_path,
                    sr=sr,
                    mono=True,
                )

        except Exception as error:
            raise RuntimeError(
                f"Both loaders failed for: "
                f"{audio_path} — {error}"
            )

    # ============================================================
    # 3. Validate waveform
    # ============================================================

    if signal is None or len(signal) == 0:
        raise RuntimeError(
            f"Audio is empty: "
            f"{audio_path}"
        )

    signal = np.asarray(
        signal,
        dtype=np.float32,
    )

    original_peak = float(
        np.max(
            np.abs(signal)
        )
    )

    if original_peak < 1e-6:
        raise RuntimeError(
            f"Audio appears silent or corrupt: "
            f"{audio_path}"
        )
    

    # ============================================================
    # 4. Mild speed perturbation
    # ============================================================
    
    if (
        training
        and np.random.rand()
        < SPEED_PROBABILITY
    ):
        signal = apply_speed_perturbation(
            signal=signal,
            sample_rate=sr,
        )
        

    # ============================================================
    # 4. Mild room reverberation
    # ============================================================

    if (
        training
        and np.random.rand()
        < REVERB_PROBABILITY
    ):
        signal = apply_mild_reverb(
            signal=signal,
            sample_rate=sr,
        )

    # ============================================================
    # 5. SNR-based Gaussian noise
    # ============================================================

    if (
        training
        and np.random.rand()
        < NOISE_PROBABILITY
    ):
        signal = add_noise_at_snr(
            signal=signal,
            min_snr_db=MIN_SNR_DB,
            max_snr_db=MAX_SNR_DB,
        )

    # ============================================================
    # 6. Normalize the complete augmented waveform
    # ============================================================
    #
    # Scaling the mixture does not change its SNR
    # or dry/reverberant relationship.
    # ============================================================

    augmented_peak = float(
        np.max(
            np.abs(signal)
        )
    )

    if augmented_peak < 1e-8:
        raise RuntimeError(
            f"Waveform became invalid after "
            f"augmentation: {audio_path}"
        )

    signal = (
        signal
        / augmented_peak
    ) * NORMALIZED_PEAK

    # ============================================================
    # 7. Random gain after normalization
    # ============================================================

    if (
        training
        and np.random.rand()
        < GAIN_PROBABILITY
    ):
        signal = random_gain(
            signal
        )

    # ============================================================
    # 8. Final safety check
    # ============================================================

    signal = np.nan_to_num(
        signal,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )

    signal = np.clip(
        signal,
        -1.0,
        1.0,
    )

    return torch.tensor(
        signal,
        dtype=torch.float32,
    )


# %% [markdown]
# # ============================================================
# # Dynamic Batch Sampler
# # ============================================================
# #
# # This sampler groups audio recordings with similar lengths into
# # dynamic batches while keeping the total padded audio size under
# # a fixed memory budget.
# #
# # For each recording:
# #   duration (sec) -> estimated number of samples
# #
# # The estimated length also considers possible speed perturbation,
# # so slower augmented audio is still handled safely.
# #
# # Example:
# #   lengths = [3s, 3.5s, 4s, 20s]
# #
# # Short recordings can be placed in the same batch, while a very
# # long recording may be placed alone.
# #
# # Batch size is therefore not fixed:
# #
# #   short audio -> larger batch
# #   long audio  -> smaller batch
# #
# # This reduces unnecessary padding, improves GPU utilization,
# # and helps prevent CUDA out-of-memory errors.
# # ============================================================

# %%
class DynamicBatchSampler(
    torch.utils.data.Sampler
):
    def __init__(
        self,
        dataset,
        max_padded_samples_per_batch,
        shuffle=True,
        minimum_speed_factor=1.0,
    ):
        self.dataset = dataset

        self.max_padded_samples_per_batch = int(
            max_padded_samples_per_batch
        )

        self.shuffle = shuffle

        self.minimum_speed_factor = float(
            minimum_speed_factor
        )

        if (
            self.max_padded_samples_per_batch
            <= 0
        ):
            raise ValueError(
                "max_padded_samples_per_batch "
                "must be greater than zero"
            )

        if self.minimum_speed_factor <= 0:
            raise ValueError(
                "minimum_speed_factor must "
                "be greater than zero"
            )

        print(
            "Building length index..."
        )

        self.original_lengths = []
        self.estimated_lengths = []

        for index in range(
            len(dataset)
        ):
            row = dataset.df.iloc[
                index
            ]

            duration_seconds = float(
                row["duration"]
            )

            if (
                not np.isfinite(
                    duration_seconds
                )
                or duration_seconds <= 0
            ):
                raise ValueError(
                    f"Invalid duration at "
                    f"index {index}: "
                    f"{duration_seconds}"
                )

            original_length = int(
                duration_seconds * SR
            )

            estimated_length = int(
                np.ceil(
                    original_length
                    / self.minimum_speed_factor
                )
            )

            self.original_lengths.append(
                original_length
            )

            self.estimated_lengths.append(
                estimated_length
            )

        longest_original_seconds = (
            max(
                self.original_lengths
            )
            / SR
        )

        longest_estimated_seconds = (
            max(
                self.estimated_lengths
            )
            / SR
        )

        over_budget_count = sum(
            length
            > self.max_padded_samples_per_batch
            for length in (
                self.estimated_lengths
            )
        )

        print(
            f"Length index built for "
            f"{len(self.estimated_lengths)} "
            f"samples"
        )

        print(
            f"Longest original audio: "
            f"{longest_original_seconds:.2f} "
            f"seconds"
        )

        print(
            f"Longest estimated audio: "
            f"{longest_estimated_seconds:.2f} "
            f"seconds"
        )

        print(
            f"Singleton recordings "
            f"over budget: "
            f"{over_budget_count}"
        )

    def _build_batches(
        self,
    ):
        indices = list(
            range(
                len(
                    self.estimated_lengths
                )
            )
        )

        indices.sort(
            key=lambda index:
            self.estimated_lengths[
                index
            ]
        )

        batches = []

        current_batch = []
        current_max_length = 0

        for index in indices:

            length = (
                self.estimated_lengths[
                    index
                ]
            )

            if (
                length
                > self.max_padded_samples_per_batch
            ):
                if current_batch:
                    batches.append(
                        current_batch
                    )

                    current_batch = []
                    current_max_length = 0

                batches.append(
                    [index]
                )

                continue

            new_max_length = max(
                current_max_length,
                length,
            )

            new_batch_size = (
                len(current_batch)
                + 1
            )

            padded_sample_count = (
                new_max_length
                * new_batch_size
            )

            if (
                current_batch
                and padded_sample_count
                > self.max_padded_samples_per_batch
            ):
                batches.append(
                    current_batch
                )

                current_batch = [
                    index
                ]

                current_max_length = (
                    length
                )

            else:
                current_batch.append(
                    index
                )

                current_max_length = (
                    new_max_length
                )

        if current_batch:
            batches.append(
                current_batch
            )

        if self.shuffle:
            random.shuffle(
                batches
            )

        return batches

    def __iter__(
        self,
    ):
        for batch in (
            self._build_batches()
        ):
            yield batch

    def __len__(
        self,
    ):
        return len(
            self._build_batches()
        )

# %% [markdown]
# # ============================================================
# # Dataset Preparation
# # ============================================================
# #
# # This dataset class connects each audio recording with its
# # corresponding phoneme sequence.
# #
# # For every sample, it:
# # - selects the correct audio folder using ds_index,
# # - loads and preprocesses the waveform,
# # - converts the phoneme sequence from text to phoneme IDs,
# # - checks that all phonemes exist in the vocabulary,
# # - returns the waveform and target sequence with their lengths.
# #
# # Example:
# #   phonemes = ["ma", "la", "ka"]
# #        ↓
# #   target IDs = [id(ma), id(la), id(ka)]
# #
# # Returned values:
# #   waveform, target_ids, audio_length, target_length
# # ============================================================

# %%
class TajweedCTCDataset(Dataset):

    def __init__(
        self, dataframe, dataset_path, dataset_path_1, phoneme_to_id, training=True
    ):
        self.df = dataframe
        self.dataset_path = dataset_path
        self.dataset_path_1 = dataset_path_1
        self.phoneme_to_id = phoneme_to_id
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        audio_path = ""
        if row["ds_index"] == 1:
            audio_path = os.path.join(self.dataset_path, row["path_of_audio"])
        else:
            audio_path = os.path.join(self.dataset_path_1, row["path_of_audio"])

        waveform = load_waveform(
            audio_path,
            training=self.training,
        )

        phoneme_seq = ast.literal_eval(row["phonemes"])
        
        unknown_phonemes = [
            p for p in phoneme_seq
            if p not in self.phoneme_to_id
        ]

        if unknown_phonemes:
            raise ValueError(
                f"Unknown phonemes {unknown_phonemes} "
                f"in sample index {idx}"
            )

        target_ids = [
            self.phoneme_to_id[p]
            for p in phoneme_seq
        ]

        if len(target_ids) == 0:
            raise ValueError(
                f"Empty target sequence at sample index {idx}"
            )

        return (
            waveform,
            torch.tensor(target_ids, dtype=torch.long),
            waveform.shape[0],
            len(target_ids),
        )

# %% [markdown]
# # ============================================================
# # Batch Padding and CTC Decoding
# # ============================================================
# #
# # ctc_collate() prepares variable-length samples for batching:
# # - audio waveforms are padded to the longest waveform,
# # - target phoneme sequences are padded with -100,
# # - original audio and target lengths are preserved.
# #
# # greedy_ctc_decode() converts frame-level CTC predictions into
# # the final phoneme sequence by:
# #   1. taking the highest-probability class at each frame,
# #   2. collapsing consecutive repeated predictions,
# #   3. removing the CTC <blank> token,
# #   4. shifting CTC IDs back to normal phoneme IDs.
# #
# # Example:
# #   CTC output: [blank, ma, ma, blank, la, la, blank]
# #        ↓
# #   Decoded sequence: [ma, la]
# # ============================================================

# %%
def ctc_collate(batch):
    waveforms, targets, input_lengths, target_lengths = zip(*batch)

    padded_waveforms = pad_sequence(
        waveforms,
        batch_first=True,
    )

    padded_targets = pad_sequence(
        targets,
        batch_first=True,
        padding_value=-100,
    )

    return (
        padded_waveforms,
        padded_targets,
        torch.tensor(input_lengths, dtype=torch.long),
        torch.tensor(target_lengths, dtype=torch.long),
    )

def greedy_ctc_decode(
    ctc_logits,
    hidden_lengths,
):
    """
    Convert frame-level CTC logits into
    phoneme sequences.

    CTC ID:
        0 = blank
        1 = original phoneme ID 0
        2 = original phoneme ID 1
        ...
    """

    frame_predictions = torch.argmax(
        ctc_logits,
        dim=-1,
    )

    decoded_batch = []

    for batch_index in range(
        frame_predictions.shape[0]
    ):
        valid_length = int(
            hidden_lengths[
                batch_index
            ].item()
        )

        frame_ids = (
            frame_predictions[
                batch_index,
                :valid_length,
            ]
            .detach()
            .cpu()
            .tolist()
        )

        decoded_ids = []

        previous_id = None

        for ctc_id in frame_ids:

            # CTC collapses consecutive
            # repeated predictions.
            if ctc_id == previous_id:
                continue

            previous_id = ctc_id

            # Remove blank.
            if ctc_id == CTC_BLANK_ID:
                continue

            # Convert CTC ID back to the
            # normal phoneme ID.
            phoneme_id = ctc_id - 1

            decoded_ids.append(
                phoneme_id
            )

        decoded_batch.append(
            decoded_ids
        )

    return decoded_batch


# %% [markdown]
# # ============================================================
# # SpecAugment for Tajweed-Aware Acoustic Features
# # ============================================================
# #
# # Applies mild augmentation directly to the acoustic feature map
# # during training by masking small time and frequency regions.
# #
# # Small masks are used intentionally to avoid destroying:
# # - Madd duration information,
# # - short-vowel timing,
# # - emphatic and nasal spectral characteristics.
# #
# # Input shape:
# #   (Batch, Time, Features)
# #
# # Example:
# #   original features
# #        ↓
# #   small time masks + small frequency masks
# #        ↓
# #   slightly corrupted features
# #
# # This improves robustness while preserving important Tajweed cues.
# # During evaluation, SpecAugment is disabled automatically.
# # ============================================================

# %%
class SpecAugment(nn.Module):
    """
    Conservative SpecAugment tuned for Tajweed:
    - Small time masks to preserve madd duration information
    - Small freq masks to preserve emphatic/nasal phoneme signatures
    - Multiple masks instead of one large one
    """

    def __init__(
        self,
        time_mask_max=8,  # max 80ms erased — safe for short vowels (~150ms)
        freq_mask_max=10,  # max 10/128 channels — preserves spectral shape
        num_time_masks=2,  # two small time masks instead of one big one
        num_freq_masks=2,  # two small freq masks
    ):
        super().__init__()
        self.time_mask_max = time_mask_max
        self.freq_mask_max = freq_mask_max
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks

    def forward(self, x):
        # x: (B, T, C)
        if not self.training:
            return x

        B, T, C = x.shape
        x = x.clone()

        # Multiple small time masks
        for _ in range(self.num_time_masks):
            t = np.random.randint(1, self.time_mask_max + 1)
            t0 = np.random.randint(0, max(1, T - t))
            x[:, t0 : t0 + t, :] = 0

        # Multiple small frequency masks
        for _ in range(self.num_freq_masks):
            f = np.random.randint(1, self.freq_mask_max + 1)
            f0 = np.random.randint(0, max(1, C - f))
            x[:, :, f0 : f0 + f] = 0

        return x

# %% [markdown]
# # ============================================================
# # Segment Classifier
# # ============================================================
# #
# # Classifies each segmented phoneme embedding into one class
# # from the phoneme vocabulary.
# #
# # Input:
# #   segment embedding -> (embedding_dim)
# #
# # Output:
# #   class scores -> (vocab_size)
# #
# # Example:
# #   segment embedding -> classifier -> "ma"
# #
# # This classifier is used as an auxiliary training signal to help
# # the segmentation module learn phoneme-discriminative segments.
# # ============================================================

# %%
class SegmentClassifier(nn.Module):
    def __init__(self, embedding_dim=64, vocab_size=len(phoneme_to_id)):
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        return self.classifier(x)

# %% [markdown]
# # ============================================================
# # Segmentation Head
# # ============================================================
# #
# # Converts each Conformer frame into a smaller segment embedding,
# # then predicts one positive weight for every time frame.
# #
# # Shapes:
# #   Input hidden features     : (B, T, 512)
# #   Segment embeddings       : (B, T, 64)
# #   Progress weights         : (B, T)
# #
# # The progress weights are forced to be positive using Softplus.
# # Their cumulative values are later used to divide the full audio
# # sequence into ordered phoneme segments.
# #
# # Example:
# #   Conformer frames
# #        ↓
# #   64-D segment embeddings
# #        ↓
# #   positive progress weights
# #        ↓
# #   phoneme boundaries
# # ============================================================

# %%
class SegmentationHead(nn.Module):
    def __init__(self, hidden_dim=512, embedding_dim=64,dropout=0.1):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
        )

        self.progress = nn.Linear(embedding_dim, 1)

    def forward(self, hidden):
        """
        hidden : (B,T,512)

        Returns
        -------
        segment_embedding : (B,T,64)

        progress_weights : (B,T)
            Positive weights whose cumulative sum
            defines the segmentation.
        """

        segment_embedding = self.embedding(hidden)

        # raw weights
        weights = self.progress(segment_embedding).squeeze(-1)

        # make them positive
        weights = F.softplus(weights)

        return segment_embedding, weights

# %% [markdown]
# # ============================================================
# # ASR Model: Recognition + Segmentation
# # ============================================================
# #
# # This is the complete model pipeline:
# #
# #   Audio
# #     ↓
# #   Wav2Vec2 feature extractor
# #     ↓
# #   Conformer
# #     ├── CTC head → phoneme sequence
# #     └── Segmentation head → phoneme boundaries
# #
# # During training:
# # - the true phoneme count is used to build the segments,
# # - the segment classifier helps train the segmentation branch.
# #
# # During inference:
# # - CTC predicts the phoneme sequence,
# # - its predicted length N is used to create N ordered segments.
# #
# # build_segments() converts learned progress weights into soft
# # frame-to-segment assignments, pools each segment embedding,
# # and extracts hard start/end boundaries.
# #
# # boundaries_to_seconds() converts encoder-frame boundaries into
# # real audio times in seconds.
# #
# # Final result:
# #   predicted phonemes + corresponding temporal boundaries.
# # ============================================================

# %%
class ASRModel(torch.nn.Module):

    def __init__(self, wav2vec2, spec_augment,dropout=0.1):
        super().__init__()
        self.wav2vec2 = wav2vec2.feature_extractor
        
        self.conv_kernel = tuple(
            wav2vec2.config.conv_kernel
        )

        self.conv_stride = tuple(
            wav2vec2.config.conv_stride
        )
        
        self.conformer = Conformer(
            input_dim=512,
            num_heads=8,
            ffn_dim=2048,
            num_layers=4,
            depthwise_conv_kernel_size=31,
            dropout=dropout,
        )

        self.segmentation = SegmentationHead(
            hidden_dim=512,
            embedding_dim=64,
            dropout=dropout,
        )

        self.segment_classifier = SegmentClassifier(
            embedding_dim=64,
            vocab_size=len(phoneme_to_id),
        )

        self.ctc_head = nn.Linear(
            512,
            CTC_VOCAB_SIZE,
        )

        self.spec_augment = spec_augment

    def get_feature_lengths(self, input_lengths):
        output_lengths = input_lengths

        for kernel_size, stride in zip(
            self.conv_kernel,
            self.conv_stride,
        ):
            output_lengths = torch.div(
                output_lengths - kernel_size,
                stride,
                rounding_mode="floor",
            ) + 1

        return output_lengths

    def get_hard_boundaries(self, weights):
        N = weights.shape[1]

        hard_segment_ids = torch.argmax(
            weights,
            dim=1,
        )

        hard_lengths = torch.bincount(
            hard_segment_ids,
            minlength=N,
        )

        end_frames = torch.cumsum(
            hard_lengths,
            dim=0,
        )

        start_frames = end_frames - hard_lengths

        hard_boundaries = torch.stack(
            [
                start_frames,
                end_frames,
            ],
            dim=1,
        )
        return hard_boundaries

    def build_segments(
        self,
        segment_embedding,
        progress_weights,
        target_lengths,
        hidden_lengths,
    ):
        B, T, D = segment_embedding.shape

        pooled_batch = []
        lengths_batch = []
        hard_boundaries_batch = []

        tau = 1.0

        for b in range(B):
            N = int(target_lengths[b].item())
            T_valid = int(hidden_lengths[b].item())

            if N <= 0:
                raise ValueError(
                    "Target length must be greater than zero"
                )

            if T_valid <= 0:
                raise ValueError(
                    "Hidden length must be greater than zero"
                )

            if N > T_valid:
                raise ValueError(
                    f"Target has {N} phonemes but only "
                    f"{T_valid} valid encoder frames"
                )

            embedding = segment_embedding[b, :T_valid]

            valid_progress_weights = progress_weights[
                b,
                :T_valid,
            ]

            progress = torch.cumsum(
                valid_progress_weights,
                dim=0,
            )

            progress = progress / progress[-1].clamp_min(
                1e-8
            )

            frame_positions = progress * N - 0.5

            centers = torch.arange(
                N,
                device=progress.device,
                dtype=progress.dtype,
            )

            distances = (
                frame_positions.unsqueeze(1)
                - centers.unsqueeze(0)
            ) ** 2

            weights = torch.softmax(
                -distances / tau,
                dim=1,
            )

            segment_mass = weights.sum(
                dim=0
            )

            pooled = (
                weights.transpose(0, 1)
                @ embedding
            )

            pooled = pooled / segment_mass.unsqueeze(
                1
            ).clamp_min(1e-8)

            hard_boundaries = self.get_hard_boundaries(
                weights
            )

            pooled_batch.append(
                pooled
            )

            lengths_batch.append(
                segment_mass
            )

            hard_boundaries_batch.append(
                hard_boundaries
            )

        return (
            pooled_batch,
            lengths_batch,
            hard_boundaries_batch,
        )
  
    def boundaries_to_seconds(
        self,
        boundaries,
        input_length,
        hidden_length,
        sample_rate=16000,
    ):
        total_stride = 1
        receptive_field = 1
    
        for kernel_size, stride in zip(
            self.conv_kernel,
            self.conv_stride,
        ):
            receptive_field += (
                kernel_size - 1
            ) * total_stride
    
            total_stride *= stride
    
        frame_center_offset = (
            receptive_field - 1
        ) / 2.0
    
        input_length = int(input_length)
        hidden_length = int(hidden_length)
    
        def frame_boundary_to_sample(frame_index):
            frame_index = int(frame_index)
    
            if frame_index <= 0:
                return 0
    
            if frame_index >= hidden_length:
                return input_length
    
            sample_index = round(
                (frame_index - 0.5) * total_stride
                + frame_center_offset
            )
    
            return max(
                0,
                min(sample_index, input_length),
            )
    
        output = []
    
        for start_frame, end_frame in (
            boundaries.detach().cpu().tolist()
        ):
            start_sample = frame_boundary_to_sample(
                start_frame
            )
    
            end_sample = frame_boundary_to_sample(
                end_frame
            )
    
            start_seconds = start_sample / sample_rate
            end_seconds = end_sample / sample_rate
    
            output.append(
                {
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "duration_seconds": (
                        end_seconds - start_seconds
                    ),
                }
            )
    
        return output

    def encode_audio(
        self,
        waveforms,
        input_lengths,
    ):
        features = self.wav2vec2(
            waveforms
        )

        features = features.transpose(
            1,
            2,
        )

        feature_lengths = (
            self.get_feature_lengths(
                input_lengths
            )
        )

        if self.training:
            features = self.spec_augment(
                features
            )

        hidden, hidden_lengths = (
            self.conformer(
                features,
                feature_lengths,
            )
        )

        return hidden, hidden_lengths

    def forward(
        self,
        waveforms,
        input_lengths,
        target_lengths=None,
    ):
        # ==========================================
        # 1. Shared acoustic encoder
        # ==========================================
        hidden, hidden_lengths = self.encode_audio(
            waveforms,
            input_lengths,
        )

        # ==========================================
        # 2. CTC phoneme recognition
        # ==========================================
        ctc_logits = self.ctc_head(
            hidden
        )

        # ==========================================
        # 3. Segmentation
        # ==========================================
        (
            segment_embedding,
            progress_weights,
        ) = self.segmentation(
            hidden
        )

        # During training we know the true number
        # of phonemes.
        if target_lengths is not None:

            (
                pooled_batch,
                segment_lengths,
                hard_boundaries_batch,
            ) = self.build_segments(
                segment_embedding,
                progress_weights,
                target_lengths,
                hidden_lengths,
            )

            segment_logits_batch = [
                self.segment_classifier(
                    pooled
                )
                for pooled in pooled_batch
            ]

            return (
                ctc_logits,
                segment_logits_batch,
                segment_lengths,
                hard_boundaries_batch,
                hidden_lengths,
            )

        # ==========================================
        # Real inference
        # ==========================================
        
        predicted_batch = greedy_ctc_decode(
            ctc_logits=ctc_logits,
            hidden_lengths=hidden_lengths,
        )
        
        predicted_lengths = torch.tensor(
            [
                max(
                    len(sequence),
                    1,
                )
                for sequence in predicted_batch
            ],
            dtype=torch.long,
            device=hidden.device,
        )
        
        predicted_lengths = torch.minimum(
            predicted_lengths,
            hidden_lengths.long(),
        )
        
        (
            pooled_batch,
            segment_lengths,
            hard_boundaries_batch,
        ) = self.build_segments(
            segment_embedding,
            progress_weights,
            predicted_lengths,
            hidden_lengths,
        )
        
        return (
            ctc_logits,
            predicted_batch,
            segment_lengths,
            hard_boundaries_batch,
            hidden_lengths,
            predicted_lengths,
        )
        

# %%
train_df = pd.read_csv(TRAIN_DS_PATH)
val_df = pd.read_csv(TEST_DS_PATH)

# %%
MAX_DIAGNOSTIC_DURATION = 45.0

train_df_full = train_df.copy()
val_df_full = val_df.copy()

train_df = train_df_full[
    train_df_full["duration"]
    <= MAX_DIAGNOSTIC_DURATION
].reset_index(
    drop=True
)

val_df = val_df_full[
    val_df_full["duration"]
    <= MAX_DIAGNOSTIC_DURATION
].reset_index(
    drop=True
)


# %%
train_dataset = TajweedCTCDataset(
    dataframe=train_df,
    training=True,
    dataset_path=DATASET_PATH,
    dataset_path_1=DATASET_PATH_1,
    phoneme_to_id=phoneme_to_id,
)
val_dataset = TajweedCTCDataset(
    dataframe=val_df,
    training=False,
    dataset_path=DATASET_PATH,
    dataset_path_1=DATASET_PATH_1,
    phoneme_to_id=phoneme_to_id,
)


MAX_PADDED_AUDIO_SECONDS = 40

MAX_PADDED_SAMPLES_PER_BATCH = int(
    SR
    * MAX_PADDED_AUDIO_SECONDS
)

MINIMUM_TRAIN_SPEED_FACTOR = min(
    SPEED_FACTORS
)


train_sampler = DynamicBatchSampler(
    dataset=train_dataset,
    max_padded_samples_per_batch=(
        MAX_PADDED_SAMPLES_PER_BATCH
    ),
    shuffle=True,
    minimum_speed_factor=(
        MINIMUM_TRAIN_SPEED_FACTOR
    ),
)


val_sampler = DynamicBatchSampler(
    dataset=val_dataset,
    max_padded_samples_per_batch=(
        MAX_PADDED_SAMPLES_PER_BATCH
    ),
    shuffle=False,
    minimum_speed_factor=1.0,
)

NUM_WORKERS = 4

train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    collate_fn=ctc_collate,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_sampler=val_sampler,
    collate_fn=ctc_collate,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
)


# %%
next(enumerate(train_loader))

# %%
def save_checkpoint(
    model,
    optimizer,
    warmup_scheduler,
    plateau_scheduler,
    accelerator,
    epoch,
    train_metrics,
    validation_metrics,
    best_validation_per,
    path,
):
    directory = os.path.dirname(
        path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    unwrapped_model = (
        accelerator.unwrap_model(model)
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state": (
                unwrapped_model.state_dict()
            ),
            "optimizer_state": (
                optimizer.state_dict()
            ),
            "warmup_scheduler_state": (
                warmup_scheduler.state_dict()
            ),
            "plateau_scheduler_state": (
                plateau_scheduler.state_dict()
            ),
            "train_metrics": (
                train_metrics
            ),
            "validation_metrics": (
                validation_metrics
            ),
            "best_validation_per": (
                best_validation_per
            ),
        },
        path,
    )
    
    checkpoint = torch.load(path)
    return checkpoint

def load_checkpoint(
    path,
    model,
    accelerator,
    optimizer=None,
    warmup_scheduler=None,
    plateau_scheduler=None,
    resume_training=True,
    strict=True,
):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Checkpoint was not found: {path}"
        )

    accelerator.wait_for_everyone()

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state'"
        )

    unwrapped_model = accelerator.unwrap_model(
        model
    )

    load_result = unwrapped_model.load_state_dict(
        checkpoint["model_state"],
        strict=strict,
    )

    if resume_training:
        if optimizer is None:
            raise ValueError(
                "optimizer must be provided when "
                "resume_training=True"
            )

        if "optimizer_state" not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain "
                "'optimizer_state'"
            )

        optimizer.load_state_dict(
            checkpoint["optimizer_state"]
        )

        if warmup_scheduler is not None:
            if (
                "warmup_scheduler_state"
                in checkpoint
            ):
                warmup_scheduler.load_state_dict(
                    checkpoint[
                        "warmup_scheduler_state"
                    ]
                )
            else:
                raise KeyError(
                    "Checkpoint does not contain "
                    "'warmup_scheduler_state'"
                )

        if plateau_scheduler is not None:
            if (
                "plateau_scheduler_state"
                in checkpoint
            ):
                plateau_scheduler.load_state_dict(
                    checkpoint[
                        "plateau_scheduler_state"
                    ]
                )
            else:
                raise KeyError(
                    "Checkpoint does not contain "
                    "'plateau_scheduler_state'"
                )

    start_epoch = int(
        checkpoint.get(
            "epoch",
            -1,
        )
    ) + 1

    best_validation_per = float(
        checkpoint.get(
            "best_validation_per",
            float("inf"),
        )
    )

    train_metrics = checkpoint.get(
        "train_metrics",
        {},
    )

    validation_metrics = checkpoint.get(
        "validation_metrics",
        {},
    )

    if accelerator.is_main_process:
        print()
        print(
            f"Checkpoint loaded: {path}"
        )

        print(
            f"Saved epoch: {start_epoch}"
        )

        print(
            f"Best validation PER: "
            f"{best_validation_per:.2%}"
        )

        if train_metrics:
            print(
                "Saved training metrics:",
                train_metrics,
            )

        if validation_metrics:
            print(
                "Saved validation metrics:",
                validation_metrics,
            )

        if not strict:
            print(
                "Missing model keys:",
                load_result.missing_keys,
            )

            print(
                "Unexpected model keys:",
                load_result.unexpected_keys,
            )

        if resume_training:
            print(
                "Optimizer and scheduler states "
                "were restored."
            )
        else:
            print(
                "Only model weights were restored."
            )

    accelerator.wait_for_everyone()

    return {
        "start_epoch": start_epoch,
        "best_validation_per": (
            best_validation_per
        ),
        "train_metrics": train_metrics,
        "validation_metrics": (
            validation_metrics
        ),
        "checkpoint": checkpoint,
    }

# %% [markdown]
# # ============================================================
# # Training Setup and Optimizer Configuration
# # ============================================================
# #
# # This block prepares the model for GPU training using Hugging Face
# # Accelerate with FP16 mixed precision.
# #
# # Different learning rates are assigned to each model component:
# #   Wav2Vec2          -> very small LR for careful fine-tuning
# #   Conformer         -> moderate LR
# #   Segmentation head -> larger LR
# #   Segment classifier-> larger LR
# #   CTC head          -> larger LR
# #
# # Training starts with a 5-epoch linear warmup, then
# # ReduceLROnPlateau lowers the learning rate when validation PER
# # stops improving.
# #
# # If a checkpoint exists, model/optimizer/scheduler states and the
# # previous epoch are restored; otherwise training starts from scratch.
# # ============================================================

# %%
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(mixed_precision="fp16", kwargs_handlers=[ddp_kwargs])

print(f"Using device: {DEVICE}")
print(f"Num processes: {accelerator.num_processes}")

wav2vec2_model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", ignore_mismatched_sizes=True,local_files_only=True
)

for param in wav2vec2_model.parameters():
    param.requires_grad = True


model = ASRModel(wav2vec2_model, SpecAugment(),dropout=0.1)

del wav2vec2_model


WARMUP_EPOCHS = 5

WAV2VEC2_LR = 3e-7
CONFORMER_LR = 3e-6
SEGMENTATION_LR = 1e-5
CLASSIFIER_LR = 2e-5
CTC_HEAD_LR = 2e-5

optimizer = torch.optim.AdamW(
    [
        {
            "params": model.wav2vec2.parameters(),
            "lr": WAV2VEC2_LR,
        },
        {
            "params": model.conformer.parameters(),
            "lr": CONFORMER_LR,
        },
        {
            "params": model.segmentation.parameters(),
            "lr": SEGMENTATION_LR,
        },
        {
            "params": model.segment_classifier.parameters(),
            "lr": CLASSIFIER_LR,
        },
        {
            "params": model.ctc_head.parameters(),
            "lr": CTC_HEAD_LR,
        },
    ],
    weight_decay=0.01,
)


# Linear warmup for first WARMUP_EPOCHS, then hand off to ReduceLROnPlateau
def warmup_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS  # 0.2, 0.4, 0.6, 0.8, 1.0
    return 1.0  # after warmup, LR stays at target — plateau scheduler takes over

model, optimizer, train_loader, val_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader
)

warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    min_lr=1e-9,
)


best_validation_per = float("inf")
start_epoch = 0

if os.path.exists(MODEL_PATH):
    print("Loading checkpoint...")
    resume_info = load_checkpoint(
        path=MODEL_PATH,
        model=model,
        accelerator=accelerator,
        optimizer=optimizer,
        warmup_scheduler=warmup_scheduler,
        plateau_scheduler=plateau_scheduler,
        resume_training=True,
        strict=True,
    )

    start_epoch = resume_info[
        "start_epoch"
    ]

    best_validation_per = resume_info[
        "best_validation_per"
    ]

    print("done loading")
else:
    print("Starting fresh training")
    best_validation_per = float("inf")
    start_epoch = 0

# %% [markdown]
# # ============================================================
# # Training Loss Functions
# # ============================================================
# #
# # segmentation_loss():
# # Measures how well each pooled segment is classified as its
# # correct phoneme using Cross-Entropy Loss with small label smoothing.
# #
# #   segment logits + target phoneme IDs
# #                 ↓
# #          classification loss
# #
# # compute_ctc_loss():
# # Measures how well the frame-level CTC output matches the target
# # phoneme sequence without requiring exact frame-to-phoneme alignment.
# #
# # CTC expects:
# #   logits shape: (T, B, C)
# #
# # Normal phoneme IDs are shifted by +1 because:
# #   CTC ID 0 = <blank>
# #
# # Example:
# #   phoneme ID 5 -> CTC target ID 6
# #
# # The two losses train different objectives:
# #   CTC loss          -> phoneme recognition
# #   Segmentation loss -> learning meaningful phoneme segments
# # ============================================================

# %%
def segmentation_loss(
    logits_batch,
    targets,
    target_lengths,
):
    all_logits = []
    all_targets = []

    for batch_index, logits in enumerate(
        logits_batch
    ):
        target_length = int(
            target_lengths[
                batch_index
            ].item()
        )

        target = targets[
            batch_index,
            :target_length,
        ].long().to(
            logits.device
        )

        all_logits.append(logits)
        all_targets.append(target)

    all_logits = torch.cat(
        all_logits,
        dim=0,
    )

    all_targets = torch.cat(
        all_targets,
        dim=0,
    )

    return F.cross_entropy(
        all_logits,
        all_targets,
        label_smoothing=0.05,
    )


def compute_ctc_loss(
    ctc_logits,
    targets,
    target_lengths,
    hidden_lengths,
):
    # ctc_logits:
    # (B, T, C)

    log_probs = F.log_softmax(
        ctc_logits,
        dim=-1,
    )

    # PyTorch CTC expects:
    # (T, B, C)
    log_probs = log_probs.transpose(
        0,
        1,
    )

    ctc_targets = []

    for batch_index in range(
        targets.shape[0]
    ):
        target_length = int(
            target_lengths[
                batch_index
            ].item()
        )

        target = targets[
            batch_index,
            :target_length,
        ].long()

        # Shift by +1 because:
        # CTC ID 0 = blank
        target = target + 1

        ctc_targets.append(
            target
        )

    ctc_targets = torch.cat(
        ctc_targets,
        dim=0,
    )

    return F.ctc_loss(
        log_probs=log_probs,
        targets=ctc_targets,
        input_lengths=hidden_lengths,
        target_lengths=target_lengths,
        blank=CTC_BLANK_ID,
        reduction="mean",
        zero_infinity=True,
    )
    

# %%
model

# %% [markdown]
# # ============================================================
# # Levenshtein Error Analysis
# # ============================================================
# #
# # Compares the predicted phoneme sequence with the reference
# # sequence using Levenshtein distance.
# #
# # It counts four possible alignment operations:
# #   match        -> correct phoneme
# #   substitute   -> wrong phoneme
# #   delete       -> missing phoneme
# #   insert       -> extra phoneme
# #
# # Example:
# #   Reference : [ma, la, ka]
# #   Prediction: [ma, na, ka, ta]
# #
# # Result:
# #   matches = 2
# #   substitutions = 1
# #   insertions = 1
# #   deletions = 0
# #
# # These counts are later used to calculate PER
# # (Phoneme Error Rate).
# # ============================================================

# %%
def levenshtein_counts(
    reference,
    hypothesis,
):
    reference = list(reference)
    hypothesis = list(hypothesis)

    reference_length = len(reference)
    hypothesis_length = len(hypothesis)

    costs = [
        [0] * (hypothesis_length + 1)
        for _ in range(
            reference_length + 1
        )
    ]

    operations = [
        [None] * (hypothesis_length + 1)
        for _ in range(
            reference_length + 1
        )
    ]

    for reference_index in range(
        1,
        reference_length + 1,
    ):
        costs[reference_index][0] = (
            reference_index
        )

        operations[
            reference_index
        ][0] = "delete"

    for hypothesis_index in range(
        1,
        hypothesis_length + 1,
    ):
        costs[0][hypothesis_index] = (
            hypothesis_index
        )

        operations[0][
            hypothesis_index
        ] = "insert"

    for reference_index in range(
        1,
        reference_length + 1,
    ):
        for hypothesis_index in range(
            1,
            hypothesis_length + 1,
        ):
            reference_token = reference[
                reference_index - 1
            ]

            hypothesis_token = hypothesis[
                hypothesis_index - 1
            ]

            if (
                reference_token
                == hypothesis_token
            ):
                costs[
                    reference_index
                ][
                    hypothesis_index
                ] = costs[
                    reference_index - 1
                ][
                    hypothesis_index - 1
                ]

                operations[
                    reference_index
                ][
                    hypothesis_index
                ] = "match"

                continue

            substitution_cost = (
                costs[
                    reference_index - 1
                ][
                    hypothesis_index - 1
                ]
                + 1
            )

            deletion_cost = (
                costs[
                    reference_index - 1
                ][
                    hypothesis_index
                ]
                + 1
            )

            insertion_cost = (
                costs[
                    reference_index
                ][
                    hypothesis_index - 1
                ]
                + 1
            )

            minimum_cost = min(
                substitution_cost,
                deletion_cost,
                insertion_cost,
            )

            costs[
                reference_index
            ][
                hypothesis_index
            ] = minimum_cost

            if (
                minimum_cost
                == substitution_cost
            ):
                operations[
                    reference_index
                ][
                    hypothesis_index
                ] = "substitute"

            elif (
                minimum_cost
                == deletion_cost
            ):
                operations[
                    reference_index
                ][
                    hypothesis_index
                ] = "delete"

            else:
                operations[
                    reference_index
                ][
                    hypothesis_index
                ] = "insert"

    substitutions = 0
    deletions = 0
    insertions = 0
    matches = 0

    reference_index = reference_length
    hypothesis_index = hypothesis_length

    while (
        reference_index > 0
        or hypothesis_index > 0
    ):
        operation = operations[
            reference_index
        ][
            hypothesis_index
        ]

        if operation == "match":
            matches += 1
            reference_index -= 1
            hypothesis_index -= 1

        elif operation == "substitute":
            substitutions += 1
            reference_index -= 1
            hypothesis_index -= 1

        elif operation == "delete":
            deletions += 1
            reference_index -= 1

        elif operation == "insert":
            insertions += 1
            hypothesis_index -= 1

        else:
            raise RuntimeError(
                "Invalid Levenshtein "
                f"operation at "
                f"({reference_index}, "
                f"{hypothesis_index})"
            )

    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "matches": matches,
    }

# %% [markdown]
# # ============================================================
# # Model Evaluation
# # ============================================================
# #
# # Evaluates the model using real inference, without providing
# # the true phoneme count to the model.
# #
# # For every sample:
# # - CTC predicts the phoneme sequence and its length,
# # - the prediction is compared with the reference sequence,
# # - Levenshtein operations are counted:
# #     substitution, deletion, insertion, match.
# #
# # Main metrics:
# #   PER = (substitutions + deletions + insertions)
# #         / number of reference phonemes
# #
# #   Aligned match rate -> percentage of correctly matched phonemes
# #   Exact sequence     -> percentage of fully correct sequences
# #   Mean count error   -> average difference between predicted
# #                         and true phoneme counts
# #   Exact count        -> percentage of samples with correct count
# #
# # Statistics are reduced across all processes when distributed
# # evaluation is used.
# # ============================================================

# %%
@torch.inference_mode()
def evaluate_model(
    model,
    val_loader,
    accelerator,
):
    model.eval()

    # ============================================================
    # Accumulators
    # ============================================================

    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0
    total_matches = 0

    total_reference_phonemes = 0
    total_predicted_phonemes = 0

    exact_sequences = 0
    total_sequences = 0

    total_count_error = 0
    exact_count_sequences = 0

    # ============================================================
    # Validation loop
    # ============================================================

    progress_bar = tqdm(
        val_loader,
        desc="CTC validation",
        disable=(
            not accelerator.is_local_main_process
        ),
    )

    for (
        waveforms,
        targets,
        input_lengths,
        target_lengths,
    ) in progress_bar:

        # --------------------------------------------------------
        # Real inference:
        # no target_lengths are given to the model
        # --------------------------------------------------------

        with accelerator.autocast():

            (
                ctc_logits,
                predicted_batch,
                segment_lengths,
                hard_boundaries_batch,
                hidden_lengths,
                predicted_lengths,
            ) = model(
                waveforms,
                input_lengths,
            )

        # ========================================================
        # Evaluate each sample
        # ========================================================

        for (
            batch_index,
            predicted_ids,
        ) in enumerate(
            predicted_batch
        ):

            # ----------------------------------------------------
            # True sequence length
            # ----------------------------------------------------

            reference_length = int(
                target_lengths[
                    batch_index
                ].item()
            )

            # ----------------------------------------------------
            # Predicted sequence length from CTC
            # ----------------------------------------------------

            predicted_length = int(
                predicted_lengths[
                    batch_index
                ].item()
            )

            # ----------------------------------------------------
            # Count metrics
            # ----------------------------------------------------

            count_error = abs(
                predicted_length
                - reference_length
            )

            total_count_error += (
                count_error
            )

            exact_count_sequences += int(
                count_error == 0
            )

            # ----------------------------------------------------
            # Get true phoneme IDs
            # ----------------------------------------------------

            reference_ids = (
                targets[
                    batch_index,
                    :reference_length,
                ]
                .detach()
                .cpu()
                .tolist()
            )

            # ----------------------------------------------------
            # Levenshtein comparison
            # ----------------------------------------------------

            alignment = levenshtein_counts(
                reference=reference_ids,
                hypothesis=predicted_ids,
            )

            substitutions = alignment[
                "substitutions"
            ]

            deletions = alignment[
                "deletions"
            ]

            insertions = alignment[
                "insertions"
            ]

            matches = alignment[
                "matches"
            ]

            # ----------------------------------------------------
            # Accumulate sequence metrics
            # ----------------------------------------------------

            total_substitutions += (
                substitutions
            )

            total_deletions += (
                deletions
            )

            total_insertions += (
                insertions
            )

            total_matches += (
                matches
            )

            total_reference_phonemes += (
                len(reference_ids)
            )

            total_predicted_phonemes += (
                len(predicted_ids)
            )

            edit_distance = (
                substitutions
                + deletions
                + insertions
            )

            exact_sequences += int(
                edit_distance == 0
            )

            total_sequences += 1

    # ============================================================
    # Combine statistics across processes
    # ============================================================

    statistics = torch.tensor(
        [
            total_substitutions,
            total_deletions,
            total_insertions,
            total_matches,
            total_reference_phonemes,
            total_predicted_phonemes,
            exact_sequences,
            total_sequences,
            total_count_error,
            exact_count_sequences,
        ],
        dtype=torch.float64,
        device=accelerator.device,
    )

    statistics = accelerator.reduce(
        statistics,
        reduction="sum",
    )

    # ============================================================
    # Extract reduced values
    # ============================================================

    substitutions = int(
        statistics[0].item()
    )

    deletions = int(
        statistics[1].item()
    )

    insertions = int(
        statistics[2].item()
    )

    matches = int(
        statistics[3].item()
    )

    reference_phoneme_count = max(
        int(
            statistics[4].item()
        ),
        1,
    )

    predicted_phoneme_count = int(
        statistics[5].item()
    )

    exact_sequence_count = int(
        statistics[6].item()
    )

    sequence_count = max(
        int(
            statistics[7].item()
        ),
        1,
    )

    total_count_error = int(
        statistics[8].item()
    )

    exact_count_sequences = int(
        statistics[9].item()
    )

    # ============================================================
    # Final metrics
    # ============================================================

    total_errors = (
        substitutions
        + deletions
        + insertions
    )

    phoneme_error_rate = (
        total_errors
        / reference_phoneme_count
    )

    aligned_match_rate = (
        matches
        / reference_phoneme_count
    )

    exact_sequence_accuracy = (
        exact_sequence_count
        / sequence_count
    )

    mean_count_error = (
        total_count_error
        / sequence_count
    )

    exact_count_accuracy = (
        exact_count_sequences
        / sequence_count
    )

    # ============================================================
    # Return metrics
    # ============================================================

    return {
        "phoneme_error_rate": (
            phoneme_error_rate
        ),
        "aligned_match_rate": (
            aligned_match_rate
        ),
        "exact_sequence_accuracy": (
            exact_sequence_accuracy
        ),

        "substitutions": (
            substitutions
        ),
        "deletions": (
            deletions
        ),
        "insertions": (
            insertions
        ),
        "matches": (
            matches
        ),

        "total_errors": (
            total_errors
        ),

        "reference_phonemes": (
            reference_phoneme_count
        ),
        "predicted_phonemes": (
            predicted_phoneme_count
        ),

        "sequence_count": (
            sequence_count
        ),

        "mean_count_error": (
            mean_count_error
        ),
        "exact_count_accuracy": (
            exact_count_accuracy
        ),
    }

# %%
CTC_LOSS_WEIGHT = 0.10
SEGMENT_LOSS_WEIGHT = 1.0

# %% [markdown]
# # ============================================================
# # Training Loop
# # ============================================================
# #
# # Supports three training modes:
# #   CTC          -> train phoneme recognition only
# #   Segmentation -> train segmentation/classifier only
# #   Joint        -> train both objectives together
# #
# # Frozen modules are kept in evaluation mode so dropout and
# # augmentation do not change their representations.
# #
# # For each batch:
# #   Audio -> Model -> CTC loss + Segmentation loss
# #                 -> selected training objective
# #                 -> Backpropagation -> Optimizer update
# #
# # After every epoch, the model is evaluated using PER and count
# # metrics, learning-rate schedulers are updated, timing/GPU usage
# # are reported, and the current/best checkpoints are saved.
# #
# # The validation stage always uses real inference:
# # the true phoneme count is not provided to the model.
# # ============================================================

# %%
def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    warmup_scheduler,
    plateau_scheduler,
    accelerator,
    epochs=5,
    start_epoch=0,
    warmup_epochs=2,
    best_validation_per=float("inf"),
    working_model_path=WORKING_MODEL_PATH,
    working_best_model_path=WORKING_BEST_MODEL_PATH,
    training_mode="joint",
):
    
    valid_training_modes = {
        "ctc",
        "segmentation",
        "joint",
    }

    if training_mode not in valid_training_modes:
        raise ValueError(
            f"Unknown training_mode: {training_mode}. "
            f"Expected one of {valid_training_modes}"
        )

    unwrapped_model = accelerator.unwrap_model(
        model
    )


    def set_requires_grad(
        module,
        requires_grad,
    ):
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad


    # ============================================================
    # Configure which parts receive gradients
    # ============================================================

    if training_mode == "ctc":

        set_requires_grad(
            unwrapped_model.wav2vec2,
            True,
        )

        set_requires_grad(
            unwrapped_model.conformer,
            True,
        )

        set_requires_grad(
            unwrapped_model.ctc_head,
            True,
        )

        set_requires_grad(
            unwrapped_model.segmentation,
            False,
        )

        set_requires_grad(
            unwrapped_model.segment_classifier,
            False,
        )


    elif training_mode == "segmentation":

        set_requires_grad(
            unwrapped_model.wav2vec2,
            False,
        )

        set_requires_grad(
            unwrapped_model.conformer,
            False,
        )

        set_requires_grad(
            unwrapped_model.ctc_head,
            False,
        )

        set_requires_grad(
            unwrapped_model.segmentation,
            True,
        )

        set_requires_grad(
            unwrapped_model.segment_classifier,
            True,
        )


    elif training_mode == "joint":

        set_requires_grad(
            unwrapped_model.wav2vec2,
            True,
        )

        set_requires_grad(
            unwrapped_model.conformer,
            True,
        )

        set_requires_grad(
            unwrapped_model.ctc_head,
            True,
        )

        set_requires_grad(
            unwrapped_model.segmentation,
            True,
        )

        set_requires_grad(
            unwrapped_model.segment_classifier,
            True,
        )


    if accelerator.is_main_process:

        print()
        print("=" * 78)
        print(
            f"Training mode: "
            f"{training_mode.upper()}"
        )
        print("=" * 78)

    
    end_epoch = start_epoch + epochs

    for epoch in range(
        start_epoch,
        end_epoch,
    ):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        epoch_start_time = time.perf_counter()

        # ============================================================
        # TRAINING
        # ============================================================

        model.train()

        # ============================================================
        # Keep frozen modules deterministic
        # ============================================================

        if training_mode == "ctc":
        
            unwrapped_model.wav2vec2.train()
            unwrapped_model.conformer.train()
            unwrapped_model.ctc_head.train()
            unwrapped_model.spec_augment.train()

            unwrapped_model.segmentation.eval()
            unwrapped_model.segment_classifier.eval()


        elif training_mode == "segmentation":
        
            # Frozen acoustic encoder.
            # Keep it in eval mode so dropout does not change
            # the representation seen by segmentation.
            unwrapped_model.wav2vec2.eval()
            unwrapped_model.conformer.eval()
            unwrapped_model.ctc_head.eval()
            unwrapped_model.spec_augment.eval()

            unwrapped_model.segmentation.train()
            unwrapped_model.segment_classifier.train()


        elif training_mode == "joint":
        
            unwrapped_model.wav2vec2.train()
            unwrapped_model.conformer.train()
            unwrapped_model.ctc_head.train()
            unwrapped_model.spec_augment.train()

            unwrapped_model.segmentation.train()
            unwrapped_model.segment_classifier.train()




        train_ctc_loss_sum = 0.0
        train_segment_loss_sum = 0.0

        train_correct_segments = 0
        train_phoneme_count = 0

        total_data_wait_time = 0.0
        total_step_time = 0.0
        measured_batch_count = 0

        slowest_batch_time = 0.0
        slowest_batch_audio_seconds = 0.0

        total_processed_audio_seconds = 0.0

        training_start_time = time.perf_counter()


        desc=(
            f"Training {epoch + 1}/"
            f"{end_epoch}"
        )
        
        train_progress = tqdm(
            train_loader,
            desc=desc,
            disable=not accelerator.is_local_main_process,
        )

        previous_batch_end_time = time.perf_counter()

        for (
            waveforms,
            targets,
            input_lengths,
            target_lengths,
        ) in train_progress:

            # ========================================================
            # Timing: DataLoader wait
            # ========================================================

            batch_ready_time = time.perf_counter()

            current_data_wait_time = (
                batch_ready_time
                - previous_batch_end_time
            )

            total_data_wait_time += (
                current_data_wait_time
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            step_start_time = time.perf_counter()

            # ========================================================
            # Forward pass
            # ========================================================

            optimizer.zero_grad(
                set_to_none=True
            )

            with accelerator.autocast():

                (
                    ctc_logits,
                    segment_logits_batch,
                    segment_lengths,
                    hard_boundaries_batch,
                    hidden_lengths,
                ) = model(
                    waveforms,
                    input_lengths,
                    target_lengths=target_lengths,
                )

                # ----------------------------------------------------
                # CTC loss
                # Main phoneme-recognition objective
                # ----------------------------------------------------

                ctc_loss = compute_ctc_loss(
                    ctc_logits=ctc_logits,
                    targets=targets,
                    target_lengths=target_lengths,
                    hidden_lengths=hidden_lengths,
                )

                # ----------------------------------------------------
                # Segmentation classifier loss
                # Auxiliary objective for learning boundaries
                # ----------------------------------------------------

                segment_loss = segmentation_loss(
                    logits_batch=segment_logits_batch,
                    targets=targets,
                    target_lengths=target_lengths,
                )

                # ----------------------------------------------------
                # Combined loss
                # ----------------------------------------------------

                # ============================================================
                # Select optimization objective
                # ============================================================

                if training_mode == "ctc":
                
                    total_loss = ctc_loss


                elif training_mode == "segmentation":
                
                    total_loss = segment_loss


                elif training_mode == "joint":
                
                    total_loss = (
                        CTC_LOSS_WEIGHT
                        * ctc_loss
                        +
                        SEGMENT_LOSS_WEIGHT
                        * segment_loss
                    )


            # ========================================================
            # Backward pass
            # ========================================================

            accelerator.backward(
                total_loss
            )

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

            optimizer.step()

            # ========================================================
            # Timing
            # ========================================================

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            step_end_time = time.perf_counter()

            current_step_time = (
                step_end_time
                - step_start_time
            )

            total_step_time += (
                current_step_time
            )

            measured_batch_count += 1

            batch_audio_seconds = float(
                input_lengths.sum().item()
                / SR
            )

            total_processed_audio_seconds += (
                batch_audio_seconds
            )

            if (
                current_step_time
                > slowest_batch_time
            ):
                slowest_batch_time = (
                    current_step_time
                )

                slowest_batch_audio_seconds = (
                    batch_audio_seconds
                )

            previous_batch_end_time = (
                step_end_time
            )

            # ========================================================
            # Training statistics
            # ========================================================

            batch_phoneme_count = int(
                target_lengths.sum().item()
            )

            train_ctc_loss_sum += (
                float(
                    ctc_loss
                    .detach()
                    .item()
                )
                * batch_phoneme_count
            )

            train_segment_loss_sum += (
                float(
                    segment_loss
                    .detach()
                    .item()
                )
                * batch_phoneme_count
            )

            train_phoneme_count += (
                batch_phoneme_count
            )

            # ========================================================
            # Auxiliary segment-classifier accuracy
            # This is NOT the final CTC accuracy.
            # ========================================================

            for (
                batch_index,
                segment_logits,
            ) in enumerate(
                segment_logits_batch
            ):

                target_length = int(
                    target_lengths[
                        batch_index
                    ].item()
                )

                predictions = torch.argmax(
                    segment_logits,
                    dim=-1,
                )

                target = targets[
                    batch_index,
                    :target_length,
                ].to(
                    predictions.device
                )

                train_correct_segments += int(
                    (
                        predictions
                        == target
                    )
                    .sum()
                    .item()
                )

            # ========================================================
            # Progress display
            # ========================================================

            train_progress.set_postfix(
                mode=training_mode,
                total=(
                    f"{total_loss.detach().item():.4f}"
                ),
                ctc=(
                    f"{ctc_loss.detach().item():.4f}"
                ),
                segment=(
                    f"{segment_loss.detach().item():.4f}"
                ),
                wait=(
                    f"{current_data_wait_time:.2f}s"
                ),
                step=(
                    f"{current_step_time:.2f}s"
                ),
            )

        # ============================================================
        # END TRAINING
        # ============================================================

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        training_time = (
            time.perf_counter()
            - training_start_time
        )

        # ============================================================
        # Combine training statistics across processes
        # ============================================================

        train_statistics = torch.tensor(
            [
                train_ctc_loss_sum,
                train_segment_loss_sum,
                train_correct_segments,
                train_phoneme_count,
            ],
            dtype=torch.float64,
            device=accelerator.device,
        )

        train_statistics = accelerator.reduce(
            train_statistics,
            reduction="sum",
        )

        global_train_phoneme_count = max(
            int(
                train_statistics[
                    3
                ].item()
            ),
            1,
        )

        average_train_ctc_loss = (
            train_statistics[
                0
            ].item()
            / global_train_phoneme_count
        )

        average_train_segment_loss = (
            train_statistics[
                1
            ].item()
            / global_train_phoneme_count
        )

        train_segment_accuracy = (
            train_statistics[
                2
            ].item()
            / global_train_phoneme_count
        )

        if training_mode == "ctc":

            average_train_total_loss = (
                average_train_ctc_loss
            )


        elif training_mode == "segmentation":
        
            average_train_total_loss = (
                average_train_segment_loss
            )


        elif training_mode == "joint":
        
            average_train_total_loss = (
                CTC_LOSS_WEIGHT
                * average_train_ctc_loss
                +
                SEGMENT_LOSS_WEIGHT
                * average_train_segment_loss
            )


        train_metrics = {
            "total_loss": (
                average_train_total_loss
            ),
            "ctc_loss": (
                average_train_ctc_loss
            ),
            "segment_loss": (
                average_train_segment_loss
            ),
            "segment_accuracy": (
                train_segment_accuracy
            ),
        }

        # ============================================================
        # VALIDATION
        # ============================================================

        validation_start_time = (
            time.perf_counter()
        )

        validation_metrics = evaluate_model(
            model=model,
            val_loader=val_loader,
            accelerator=accelerator,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        validation_time = (
            time.perf_counter()
            - validation_start_time
        )

        validation_per = (
            validation_metrics[
                "phoneme_error_rate"
            ]
        )

        # ============================================================
        # Scheduler
        # ============================================================

        if epoch < warmup_epochs:

            warmup_scheduler.step()
        
            active_scheduler = "warmup"
        
        
        elif training_mode in {
            "ctc",
            "joint",
        }:
        
            plateau_scheduler.step(
                validation_per
            )
        
            active_scheduler = "plateau"
        
        
        else:
        
            # Segmentation-only:
            # CTC PER is not a valid scheduler metric
            # because the CTC system is frozen.
            active_scheduler = "none"
        
        

        # ============================================================
        # Best model
        # ============================================================

        improved = (
            validation_per
            < best_validation_per
        )

        if improved:
            best_validation_per = (
                validation_per
            )

        # ============================================================
        # Timing statistics
        # ============================================================

        average_data_wait_time = (
            total_data_wait_time
            / max(
                measured_batch_count,
                1,
            )
        )

        average_step_time = (
            total_step_time
            / max(
                measured_batch_count,
                1,
            )
        )

        measured_training_time = (
            total_data_wait_time
            + total_step_time
        )

        data_wait_percentage = (
            100.0
            * total_data_wait_time
            / max(
                measured_training_time,
                1e-8,
            )
        )

        model_work_percentage = (
            100.0
            * total_step_time
            / max(
                measured_training_time,
                1e-8,
            )
        )

        audio_throughput = (
            total_processed_audio_seconds
            / max(
                training_time,
                1e-8,
            )
        )

        epoch_time_before_saving = (
            time.perf_counter()
            - epoch_start_time
        )

        timing_metrics = {
            "epoch_minutes_before_saving": (
                epoch_time_before_saving
                / 60.0
            ),
            "training_minutes": (
                training_time
                / 60.0
            ),
            "validation_minutes": (
                validation_time
                / 60.0
            ),
            "average_data_wait_seconds": (
                average_data_wait_time
            ),
            "average_step_seconds": (
                average_step_time
            ),
            "data_wait_percentage": (
                data_wait_percentage
            ),
            "model_work_percentage": (
                model_work_percentage
            ),
            "slowest_batch_seconds": (
                slowest_batch_time
            ),
            "slowest_batch_audio_seconds": (
                slowest_batch_audio_seconds
            ),
            "audio_throughput": (
                audio_throughput
            ),
            "training_batches": (
                measured_batch_count
            ),
        }

        train_metrics[
            "timing"
        ] = timing_metrics

        # ============================================================
        # PRINT RESULTS
        # ============================================================

        if accelerator.is_main_process:

            print()
            print("=" * 78)
            print(
                f"Epoch {epoch + 1}/{end_epoch}"
            )
            print("=" * 78)

            print()
            print("Training")
            print("-" * 40)

            print(
                f"Total loss          : "
                f"{average_train_total_loss:.4f}"
            )

            print(
                f"CTC loss            : "
                f"{average_train_ctc_loss:.4f}"
            )

            print(
                f"Segment loss        : "
                f"{average_train_segment_loss:.4f}"
            )

            print(
                f"Segment accuracy    : "
                f"{train_segment_accuracy:.2%}"
            )

            print()
            print("CTC validation")
            print("-" * 40)

            print(
                f"Phoneme error rate  : "
                f"{validation_per:.2%}"
            )

            print(
                f"Aligned match rate  : "
                f"{validation_metrics['aligned_match_rate']:.2%}"
            )

            print(
                f"Exact sequences     : "
                f"{validation_metrics['exact_sequence_accuracy']:.2%}"
            )

            print(
                f"Substitutions       : "
                f"{validation_metrics['substitutions']}"
            )

            print(
                f"Deletions           : "
                f"{validation_metrics['deletions']}"
            )

            print(
                f"Insertions          : "
                f"{validation_metrics['insertions']}"
            )

            print(
                f"Reference phonemes  : "
                f"{validation_metrics['reference_phonemes']}"
            )

            print(
                f"Predicted phonemes  : "
                f"{validation_metrics['predicted_phonemes']}"
            )

            # These exist after our Step 8.
            if (
                "exact_count_accuracy"
                in validation_metrics
            ):
                print(
                    f"Exact count         : "
                    f"{validation_metrics['exact_count_accuracy']:.2%}"
                )

            if (
                "mean_count_error"
                in validation_metrics
            ):
                print(
                    f"Mean count error    : "
                    f"{validation_metrics['mean_count_error']:.4f}"
                )

            print(
                f"Best validation PER : "
                f"{best_validation_per:.2%}"
            )

            print(
                f"Scheduler           : "
                f"{active_scheduler}"
            )

            print()
            print("Timing")
            print("-" * 40)

            print(
                f"Training time       : "
                f"{training_time / 60:.2f} min"
            )

            print(
                f"Validation time     : "
                f"{validation_time / 60:.2f} min"
            )

            print(
                f"Mean data wait      : "
                f"{average_data_wait_time:.4f} sec/batch"
            )

            print(
                f"Mean model step     : "
                f"{average_step_time:.4f} sec/batch"
            )

            print(
                f"Data wait share     : "
                f"{data_wait_percentage:.2f}%"
            )

            print(
                f"Model work share    : "
                f"{model_work_percentage:.2f}%"
            )

            print(
                f"Slowest step        : "
                f"{slowest_batch_time:.2f} sec"
            )

            print(
                f"Slowest batch audio : "
                f"{slowest_batch_audio_seconds:.2f} sec"
            )

            print(
                f"Audio throughput    : "
                f"{audio_throughput:.2f} "
                f"audio-sec/wall-sec"
            )

            print()
            print("Learning rates")
            print("-" * 40)

            module_names = [
                "Wav2Vec2",
                "Conformer",
                "Segmentation",
                "Classifier",
                "CTC head",
            ]

            for (
                group_index,
                group,
            ) in enumerate(
                optimizer.param_groups
            ):

                if (
                    group_index
                    < len(module_names)
                ):
                    group_name = (
                        module_names[
                            group_index
                        ]
                    )
                else:
                    group_name = (
                        f"Group {group_index}"
                    )

                print(
                    f"{group_name:<14}: "
                    f"{group['lr']:.8g}"
                )


            print()
            print("GPU memory usage")
            print("-" * 40)

            if torch.cuda.is_available():
                peak_allocated_gb = (
                    torch.cuda.max_memory_allocated()
                    / 1024**3
                )

                peak_reserved_gb = (
                    torch.cuda.max_memory_reserved()
                    / 1024**3
                )

                total_memory_gb = (
                    torch.cuda.get_device_properties(0).total_memory
                    / 1024**3
                )

                print(
                    f"GPU memory allocated : "
                    f"{peak_allocated_gb:.2f} GB"
                )

                print(
                    f"GPU memory reserved  : "
                    f"{peak_reserved_gb:.2f} GB"
                )

                print(
                    f"GPU total memory     : "
                    f"{total_memory_gb:.2f} GB"
                )


            print("=" * 78)
            print()

        # ============================================================
        # SAVE CHECKPOINT
        # ============================================================

        accelerator.wait_for_everyone()

        checkpoint_start_time = (
            time.perf_counter()
        )

        if accelerator.is_main_process:

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                warmup_scheduler=warmup_scheduler,
                plateau_scheduler=plateau_scheduler,
                accelerator=accelerator,
                epoch=epoch,
                train_metrics=train_metrics,
                validation_metrics=(
                    validation_metrics
                ),
                best_validation_per=(
                    best_validation_per
                ),
                path=working_model_path,
            )

            if improved:

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    warmup_scheduler=warmup_scheduler,
                    plateau_scheduler=plateau_scheduler,
                    accelerator=accelerator,
                    epoch=epoch,
                    train_metrics=(
                        train_metrics
                    ),
                    validation_metrics=(
                        validation_metrics
                    ),
                    best_validation_per=(
                        best_validation_per
                    ),
                    path=(
                        working_best_model_path
                    ),
                )

                print(
                    "Best model saved"
                )

        accelerator.wait_for_everyone()

        checkpoint_time = (
            time.perf_counter()
            - checkpoint_start_time
        )

        total_epoch_time = (
            time.perf_counter()
            - epoch_start_time
        )

        if accelerator.is_main_process:

            print(
                f"Checkpoint time     : "
                f"{checkpoint_time:.2f} sec"
            )

            print(
                f"Complete epoch time : "
                f"{total_epoch_time / 60:.2f} min"
            )

            print("-" * 78)

    return best_validation_per

# %%
print("training is starting .......")

# %%
# best_validation_per = train_model(
#     model=model,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     optimizer=optimizer,
#     warmup_scheduler=warmup_scheduler,
#     plateau_scheduler=plateau_scheduler,
#     accelerator=accelerator,

#     epochs=NUM_EPOCHS,
#     start_epoch=start_epoch,

#     warmup_epochs=WARMUP_EPOCHS,
#     best_validation_per=best_validation_per,

#     working_model_path=WORKING_MODEL_PATH,
#     working_best_model_path=WORKING_BEST_MODEL_PATH,

#     training_mode="ctc",
# )

# %% [markdown]
# # ============================================================
# # Detailed Phoneme Sequence Alignment
# # ============================================================
# #
# # Aligns the reference and predicted phoneme sequences using
# # Levenshtein dynamic programming.
# #
# # Each aligned position is labeled as:
# #   match        -> correct phoneme
# #   substitute   -> wrong phoneme
# #   delete       -> missing phoneme
# #   insert       -> extra phoneme
# #
# # Example:
# #   Reference : [ma, la, ka]
# #   Prediction: [ma, na, ka, ta]
# #
# #   Alignment :
# #   ma -> ma   (match)
# #   la -> na   (substitute)
# #   ka -> ka   (match)
# #   -  -> ta   (insert)
# #
# # The function also keeps the reference/prediction indices so the
# # aligned phonemes can later be connected with their time boundaries.
# # ============================================================

# %%
def align_phoneme_sequences(
    reference,
    hypothesis,
):
    """
    Levenshtein alignment.

    Returns an aligned sequence of operations:
        match
        substitute
        delete
        insert
    """

    reference = list(reference)
    hypothesis = list(hypothesis)

    reference_length = len(reference)
    hypothesis_length = len(hypothesis)

    costs = [
        [0] * (hypothesis_length + 1)
        for _ in range(reference_length + 1)
    ]

    operations = [
        [None] * (hypothesis_length + 1)
        for _ in range(reference_length + 1)
    ]

    for reference_index in range(
        1,
        reference_length + 1,
    ):
        costs[reference_index][0] = (
            reference_index
        )

        operations[reference_index][0] = (
            "delete"
        )

    for hypothesis_index in range(
        1,
        hypothesis_length + 1,
    ):
        costs[0][hypothesis_index] = (
            hypothesis_index
        )

        operations[0][hypothesis_index] = (
            "insert"
        )

    for reference_index in range(
        1,
        reference_length + 1,
    ):
        for hypothesis_index in range(
            1,
            hypothesis_length + 1,
        ):
            reference_phoneme = reference[
                reference_index - 1
            ]

            hypothesis_phoneme = hypothesis[
                hypothesis_index - 1
            ]

            if (
                reference_phoneme
                == hypothesis_phoneme
            ):
                costs[
                    reference_index
                ][
                    hypothesis_index
                ] = costs[
                    reference_index - 1
                ][
                    hypothesis_index - 1
                ]

                operations[
                    reference_index
                ][
                    hypothesis_index
                ] = "match"

                continue

            substitution_cost = (
                costs[
                    reference_index - 1
                ][
                    hypothesis_index - 1
                ]
                + 1
            )

            deletion_cost = (
                costs[
                    reference_index - 1
                ][
                    hypothesis_index
                ]
                + 1
            )

            insertion_cost = (
                costs[
                    reference_index
                ][
                    hypothesis_index - 1
                ]
                + 1
            )

            minimum_cost = min(
                substitution_cost,
                deletion_cost,
                insertion_cost,
            )

            costs[
                reference_index
            ][
                hypothesis_index
            ] = minimum_cost

            if (
                minimum_cost
                == substitution_cost
            ):
                operations[
                    reference_index
                ][
                    hypothesis_index
                ] = "substitute"

            elif (
                minimum_cost
                == deletion_cost
            ):
                operations[
                    reference_index
                ][
                    hypothesis_index
                ] = "delete"

            else:
                operations[
                    reference_index
                ][
                    hypothesis_index
                ] = "insert"

    alignment = []

    reference_index = reference_length
    hypothesis_index = hypothesis_length

    while (
        reference_index > 0
        or hypothesis_index > 0
    ):
        operation = operations[
            reference_index
        ][
            hypothesis_index
        ]

        if operation == "match":
            alignment.append(
                {
                    "operation": "match",
                    "reference_index": (
                        reference_index - 1
                    ),
                    "hypothesis_index": (
                        hypothesis_index - 1
                    ),
                    "target_phoneme": reference[
                        reference_index - 1
                    ],
                    "predicted_phoneme": hypothesis[
                        hypothesis_index - 1
                    ],
                }
            )

            reference_index -= 1
            hypothesis_index -= 1

        elif operation == "substitute":
            alignment.append(
                {
                    "operation": "substitute",
                    "reference_index": (
                        reference_index - 1
                    ),
                    "hypothesis_index": (
                        hypothesis_index - 1
                    ),
                    "target_phoneme": reference[
                        reference_index - 1
                    ],
                    "predicted_phoneme": hypothesis[
                        hypothesis_index - 1
                    ],
                }
            )

            reference_index -= 1
            hypothesis_index -= 1

        elif operation == "delete":
            alignment.append(
                {
                    "operation": "delete",
                    "reference_index": (
                        reference_index - 1
                    ),
                    "hypothesis_index": None,
                    "target_phoneme": reference[
                        reference_index - 1
                    ],
                    "predicted_phoneme": None,
                }
            )

            reference_index -= 1

        elif operation == "insert":
            alignment.append(
                {
                    "operation": "insert",
                    "reference_index": None,
                    "hypothesis_index": (
                        hypothesis_index - 1
                    ),
                    "target_phoneme": None,
                    "predicted_phoneme": hypothesis[
                        hypothesis_index - 1
                    ],
                }
            )

            hypothesis_index -= 1

        else:
            raise RuntimeError(
                "Invalid Levenshtein operation "
                f"at ({reference_index}, "
                f"{hypothesis_index})"
            )

    alignment.reverse()

    substitutions = sum(
        item["operation"] == "substitute"
        for item in alignment
    )

    deletions = sum(
        item["operation"] == "delete"
        for item in alignment
    )

    insertions = sum(
        item["operation"] == "insert"
        for item in alignment
    )

    matches = sum(
        item["operation"] == "match"
        for item in alignment
    )

    return {
        "alignment": alignment,
        "matches": matches,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "total_errors": (
            substitutions
            + deletions
            + insertions
        ),
    }

# %%
def phoneme_to_arabic_text(phoneme):
    """
    Convert one model phoneme token to a readable
    Arabic representation.

    This is for display only.
    """

    if phoneme is None:
        return "-"

    if phoneme == "<sil>":
        return "صمت"

    # ============================================================
    # Special Tajweed tokens
    # ============================================================

    if phoneme == "nn":
        return "نّ"

    if phoneme == "mm":
        return "مّ"

    if phoneme == "yy":
        return "يّ"

    if phoneme == "ww":
        return "وّ"

    # ============================================================
    # Find the consonant/base part.
    #
    # Longer tokens must be checked first:
    # rM before r
    # sˤ before s
    # etc.
    # ============================================================

    base_tokens = sorted(
        PHONEME_BASE_TO_ARABIC.keys(),
        key=len,
        reverse=True,
    )

    base = None
    remainder = phoneme

    for candidate in base_tokens:

        if candidate == "<sil>":
            continue

        if phoneme.startswith(candidate):

            base = candidate

            remainder = phoneme[
                len(candidate):
            ]

            break

    # Pure vowel
    if base is None:

        vowel_map = {
            "a": "َ",
            "i": "ِ",
            "u": "ُ",

            "aa": "ا",
            "ii": "ي",
            "uu": "و",

            "an": "ً",
            "in": "ٍ",
            "un": "ٌ",
        }

        return vowel_map.get(
            phoneme,
            phoneme,
        )

    arabic_letter = (
        PHONEME_BASE_TO_ARABIC[
            base
        ]
    )

    # ============================================================
    # Vowels
    # ============================================================

    suffix_map = {
        "": "",

        "a": "َ",
        "i": "ِ",
        "u": "ُ",

        "aa": "َا",
        "ii": "ِي",
        "uu": "ُو",

        "an": "ً",
        "in": "ٍ",
        "un": "ٌ",

        # Qalqalah marker
        "K": "ْ",
    }

    if remainder in suffix_map:

        return (
            arabic_letter
            + suffix_map[
                remainder
            ]
        )

    # Tajweed-special tokens such as
    # n + ikhfa letter.
    return arabic_letter

# %% [markdown]
# # ============================================================
# # Single-Audio Testing and Detailed Inference
# # ============================================================
# #
# # Evaluates one external audio file using the final model pipeline:
# #
# #   Audio
# #     ↓
# #   Wav2Vec2 + Conformer
# #     ↓
# #   CTC phoneme prediction
# #     ↓
# #   predicted phoneme count
# #     ↓
# #   segmentation boundaries
# #
# # If a reference phoneme sequence is provided, the function:
# # - aligns reference and prediction using Levenshtein alignment,
# # - calculates PER, matches, substitutions, deletions and insertions,
# # - reports missing/extra phonemes,
# # - connects each predicted phoneme with its start/end time.
# #
# # It also converts encoder-frame boundaries into seconds and
# # prints a detailed table containing:
# #   target phoneme, predicted phoneme, Arabic display,
# #   alignment result, start time, end time and duration.
# # ============================================================

# %%
@torch.inference_mode()
def test_external_audio(
    audio_path,
    model,
    accelerator,
    id_to_phoneme,
    phoneme_to_id=None,
    correct_target=None,
    sample_rate=16000,
    print_result=True,
):
    """
    Evaluate one audio file using the FINAL architecture:

        Audio
          ↓
        Wav2Vec2 + Conformer
          ↓
        CTC
          ↓
        predicted phoneme sequence
          ↓
        N = number of CTC phonemes
          ↓
        Segmentation head
          ↓
        N temporal boundaries

    Final phoneme recognition comes ONLY from CTC.

    The segment classifier is NOT used for final
    phoneme prediction.

    If correct_target is supplied, CTC recognition
    is evaluated using Levenshtein alignment.
    """

    import ast
    import torch

    # ============================================================
    # 1. Parse optional reference
    # ============================================================

    if correct_target is not None:

        if isinstance(correct_target, str):

            target_text = correct_target.strip()

            if target_text.startswith("["):
                correct_target = ast.literal_eval(
                    target_text
                )
            else:
                correct_target = (
                    target_text.split()
                )

        if not isinstance(
            correct_target,
            (list, tuple),
        ):
            raise TypeError(
                "correct_target must be None, "
                "a list/tuple of phonemes, "
                "or a string."
            )

        correct_target = list(
            correct_target
        )

        if len(correct_target) == 0:
            raise ValueError(
                "correct_target cannot be empty."
            )

        if phoneme_to_id is not None:

            unknown_phonemes = [
                phoneme
                for phoneme in correct_target
                if phoneme not in phoneme_to_id
            ]

            if unknown_phonemes:
                raise ValueError(
                    f"Unknown reference phonemes: "
                    f"{unknown_phonemes}"
                )

    # ============================================================
    # 2. Load audio WITHOUT augmentation
    # ============================================================

    waveform = load_waveform(
        audio_path=audio_path,
        sr=sample_rate,
        training=False,
    )

    original_num_samples = int(
        waveform.shape[0]
    )

    audio_duration_seconds = (
        original_num_samples
        / float(sample_rate)
    )

    waveform_batch = (
        waveform
        .unsqueeze(0)
        .to(accelerator.device)
    )

    input_lengths = torch.tensor(
        [original_num_samples],
        dtype=torch.long,
        device=accelerator.device,
    )

    # ============================================================
    # 3. REAL autonomous inference
    #
    # IMPORTANT:
    # target_lengths are NOT passed.
    # ============================================================

    model.eval()

    with accelerator.autocast():

        (
            ctc_logits,
            predicted_batch,
            segment_lengths_batch,
            hard_boundaries_batch,
            hidden_lengths,
            predicted_lengths,
        ) = model(
            waveform_batch,
            input_lengths,
        )

    # ============================================================
    # 4. CTC output
    # ============================================================

    predicted_ids = list(
        predicted_batch[0]
    )

    # This is the REAL CTC count.
    ctc_predicted_count = len(
        predicted_ids
    )

    predicted_phonemes = [
        id_to_phoneme[
            int(phoneme_id)
        ]
        for phoneme_id in predicted_ids
    ]

    hidden_length = int(
        hidden_lengths[0].item()
    )

    # ============================================================
    # 5. Segmentation output
    # ============================================================

    # Current model still uses max(N, 1) internally
    # when CTC gives an empty sequence.
    #
    # Therefore, if CTC predicts N=0, ignore the
    # artificial segmentation result.
    # ============================================================

    if ctc_predicted_count == 0:

        hard_boundaries = torch.empty(
            (0, 2),
            dtype=torch.long,
        )

        soft_segment_masses = torch.empty(
            (0,),
            dtype=torch.float32,
        )

    else:

        hard_boundaries = (
            hard_boundaries_batch[0]
            .detach()
            .cpu()
        )

        soft_segment_masses = (
            segment_lengths_batch[0]
            .detach()
            .float()
            .cpu()
        )

        # Safety check.
        if (
            hard_boundaries.shape[0]
            != ctc_predicted_count
        ):
            raise RuntimeError(
                "Segmentation count does not match "
                "CTC decoded count. "
                f"CTC={ctc_predicted_count}, "
                f"segments="
                f"{hard_boundaries.shape[0]}"
            )

    # ============================================================
    # 6. Convert boundaries from frames to seconds
    # ============================================================

    unwrapped_model = (
        accelerator.unwrap_model(model)
    )

    if ctc_predicted_count > 0:

        boundary_data = (
            unwrapped_model.boundaries_to_seconds(
                boundaries=hard_boundaries,
                input_length=original_num_samples,
                hidden_length=hidden_length,
                sample_rate=sample_rate,
            )
        )

    else:

        boundary_data = []

    # Encoder nominal frame stride.
    total_stride = 1

    for stride in (
        unwrapped_model.conv_stride
    ):
        total_stride *= stride

    nominal_frame_duration_seconds = (
        total_stride
        / float(sample_rate)
    )

    # ============================================================
    # 7. Build phoneme + boundary results
    # ============================================================

    segments = []

    for index in range(
        ctc_predicted_count
    ):

        boundary = boundary_data[
            index
        ]

        start_frame = int(
            boundary["start_frame"]
        )

        end_frame = int(
            boundary["end_frame"]
        )

        frame_count = (
            end_frame
            - start_frame
        )

        duration_seconds = float(
            boundary[
                "duration_seconds"
            ]
        )

        segment = {
            "index": index,

            # Final phoneme comes from CTC.
            "predicted_id": int(
                predicted_ids[index]
            ),

            "predicted_phoneme": (
                predicted_phonemes[index]
            ),

            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_count": frame_count,

            "start_sample": int(
                boundary["start_sample"]
            ),

            "end_sample": int(
                boundary["end_sample"]
            ),

            "start_seconds": float(
                boundary["start_seconds"]
            ),

            "end_seconds": float(
                boundary["end_seconds"]
            ),

            "duration_seconds": (
                duration_seconds
            ),

            "soft_segment_mass": float(
                soft_segment_masses[
                    index
                ].item()
            ),

            # Filled later if reference exists.
            "target_phoneme": None,
            "target_index": None,
            "alignment_operation": None,
            "correct": None,
        }

        segments.append(
            segment
        )

    # ============================================================
    # 8. Optional CTC reference evaluation
    # ============================================================

    evaluation = None
    alignment = None

    if correct_target is not None:

        alignment_result = (
            align_phoneme_sequences(
                reference=correct_target,
                hypothesis=predicted_phonemes,
            )
        )

        alignment = alignment_result[
            "alignment"
        ]

        matches = int(
            alignment_result["matches"]
        )

        substitutions = int(
            alignment_result[
                "substitutions"
            ]
        )

        deletions = int(
            alignment_result["deletions"]
        )

        insertions = int(
            alignment_result["insertions"]
        )

        total_errors = int(
            alignment_result[
                "total_errors"
            ]
        )

        reference_count = len(
            correct_target
        )

        phoneme_error_rate = (
            total_errors
            / max(reference_count, 1)
        )

        reference_match_rate = (
            matches
            / max(reference_count, 1)
        )

        count_error = abs(
            ctc_predicted_count
            - reference_count
        )

        exact_sequence = (
            total_errors == 0
        )

        evaluation = {
            "reference_count": (
                reference_count
            ),

            "predicted_count": (
                ctc_predicted_count
            ),

            "count_error": (
                count_error
            ),

            "count_is_exact": (
                count_error == 0
            ),

            "matches": matches,

            "substitutions": (
                substitutions
            ),

            "deletions": (
                deletions
            ),

            "insertions": (
                insertions
            ),

            "total_errors": (
                total_errors
            ),

            "phoneme_error_rate": (
                phoneme_error_rate
            ),

            "reference_match_rate": (
                reference_match_rate
            ),

            "exact_sequence": (
                exact_sequence
            ),
        }

        # --------------------------------------------------------
        # Attach Levenshtein alignment to corresponding
        # predicted segment.
        # --------------------------------------------------------

        for item in alignment:

            hypothesis_index = item[
                "hypothesis_index"
            ]

            if hypothesis_index is None:
                # Deleted target has no predicted
                # phoneme and therefore no boundary.
                continue

            if (
                hypothesis_index
                >= len(segments)
            ):
                raise RuntimeError(
                    "Alignment hypothesis index "
                    "is outside segment range."
                )

            segments[
                hypothesis_index
            ][
                "target_phoneme"
            ] = item[
                "target_phoneme"
            ]

            segments[
                hypothesis_index
            ][
                "target_index"
            ] = item[
                "reference_index"
            ]

            segments[
                hypothesis_index
            ][
                "alignment_operation"
            ] = item[
                "operation"
            ]

            segments[
                hypothesis_index
            ][
                "correct"
            ] = (
                item["operation"]
                == "match"
            )

    # ============================================================
    # 9. Useful segmentation sanity statistics
    # ============================================================

    durations = [
        segment["duration_seconds"]
        for segment in segments
    ]

    if durations:

        mean_duration = float(
            np.mean(durations)
        )

        median_duration = float(
            np.median(durations)
        )

        minimum_duration = float(
            np.min(durations)
        )

        maximum_duration = float(
            np.max(durations)
        )

        zero_duration_segments = int(
            sum(
                duration <= 0.0
                for duration in durations
            )
        )

    else:

        mean_duration = 0.0
        median_duration = 0.0
        minimum_duration = 0.0
        maximum_duration = 0.0
        zero_duration_segments = 0

    segmentation_summary = {
        "segment_count": (
            len(segments)
        ),

        "mean_duration_seconds": (
            mean_duration
        ),

        "median_duration_seconds": (
            median_duration
        ),

        "minimum_duration_seconds": (
            minimum_duration
        ),

        "maximum_duration_seconds": (
            maximum_duration
        ),

        "zero_duration_segments": (
            zero_duration_segments
        ),
    }

    # ============================================================
    # 10. Final output
    # ============================================================

    output = {
        "audio_path": audio_path,

        "sample_rate": sample_rate,

        "audio_samples": (
            original_num_samples
        ),

        "audio_duration_seconds": (
            audio_duration_seconds
        ),

        "encoder_frame_count": (
            hidden_length
        ),

        "nominal_frame_duration_seconds": (
            nominal_frame_duration_seconds
        ),

        "predicted_count": (
            ctc_predicted_count
        ),

        "predicted_phoneme_ids": (
            predicted_ids
        ),

        "predicted_phonemes": (
            predicted_phonemes
        ),

        "correct_target": (
            correct_target
        ),

        "evaluation": evaluation,

        "alignment": alignment,

        "segmentation_summary": (
            segmentation_summary
        ),

        "segments": segments,
    }

    # ============================================================
    # 11. Print results
    # ============================================================

    if print_result:

        print()
        print("=" * 100)

        print(
            f"Audio: {audio_path}"
        )

        print(
            f"Duration: "
            f"{audio_duration_seconds:.3f} s"
        )

        print(
            f"Encoder frames: "
            f"{hidden_length}"
        )

        print(
            f"CTC predicted count: "
            f"{ctc_predicted_count}"
        )

        print(
            f"Nominal encoder frame: "
            f"{nominal_frame_duration_seconds * 1000:.2f} ms"
        )

        # --------------------------------------------------------
        # Recognition metrics
        # --------------------------------------------------------

        if evaluation is not None:

            print()
            print(
                "CTC recognition evaluation"
            )
            print("-" * 50)

            print(
                f"Reference count : "
                f"{evaluation['reference_count']}"
            )

            print(
                f"Predicted count : "
                f"{evaluation['predicted_count']}"
            )

            print(
                f"Count error     : "
                f"{evaluation['count_error']}"
            )

            print(
                f"Matches         : "
                f"{evaluation['matches']}"
            )

            print(
                f"Substitutions   : "
                f"{evaluation['substitutions']}"
            )

            print(
                f"Deletions       : "
                f"{evaluation['deletions']}"
            )

            print(
                f"Insertions      : "
                f"{evaluation['insertions']}"
            )

            print(
                f"PER             : "
                f"{evaluation['phoneme_error_rate']:.2%}"
            )

            print(
                f"Match rate      : "
                f"{evaluation['reference_match_rate']:.2%}"
            )

            print(
                f"Exact sequence  : "
                f"{evaluation['exact_sequence']}"
            )

        # --------------------------------------------------------
        # Segmentation summary
        # --------------------------------------------------------

        print()
        print(
            "Segmentation summary"
        )
        print("-" * 50)

        print(
            f"Segments        : "
            f"{segmentation_summary['segment_count']}"
        )

        print(
            f"Mean duration   : "
            f"{mean_duration:.4f} s"
        )

        print(
            f"Median duration : "
            f"{median_duration:.4f} s"
        )

        print(
            f"Min duration    : "
            f"{minimum_duration:.4f} s"
        )

        print(
            f"Max duration    : "
            f"{maximum_duration:.4f} s"
        )

        print(
            f"Zero durations  : "
            f"{zero_duration_segments}"
        )

        # --------------------------------------------------------
        # Individual phonemes / boundaries
        # --------------------------------------------------------

        print()
        print(
            "Predicted phonemes and boundaries"
        )
        
        print("=" * 155)
        
        # ============================================================
        # If reference exists, print using the alignment.
        #
        # This is important because deletions have no predicted
        # segment and therefore would otherwise disappear.
        # ============================================================
        
        if alignment is not None:
        
            row_index = 0
        
            for item in alignment:
            
                operation = item[
                    "operation"
                ]
        
                target_phoneme = item[
                    "target_phoneme"
                ]
        
                predicted_phoneme = item[
                    "predicted_phoneme"
                ]
        
                target_arabic = (
                    phoneme_to_arabic_text(
                        target_phoneme
                    )
                )
        
                predicted_arabic = (
                    phoneme_to_arabic_text(
                        predicted_phoneme
                    )
                )
        
                hypothesis_index = item[
                    "hypothesis_index"
                ]
        
                # ========================================================
                # DELETION / MISSING PHONEME
                # ========================================================
        
                if hypothesis_index is None:
                
                    print(
                        f"{row_index:03d} | "
                        f"target="
                        f"{str(target_phoneme):<10} "
                        f"({target_arabic:<5}) | "
                        f"pred="
                        f"{'<MISSING>':<10} "
                        f"({'-':<5}) | "
                        f"{'MISSING':<11} | "
                        f"time=[   -----,    -----) | "
                        f"duration=  ----- | "
                        f"frames= ---"
                    )
        
                    row_index += 1
        
                    continue
                
                # ========================================================
                # MATCH / SUBSTITUTE / INSERT
                # ========================================================
        
                segment = segments[
                    hypothesis_index
                ]
        
                # For insertion there is no reference phoneme.
                if target_phoneme is None:
                
                    target_display = "-"
                    target_arabic = "-"
        
                else:
                
                    target_display = (
                        target_phoneme
                    )
        
                predicted_display = (
                    predicted_phoneme
                )
        
                print(
                    f"{row_index:03d} | "
                    f"target="
                    f"{str(target_display):<10} "
                    f"({target_arabic:<5}) | "
                    f"pred="
                    f"{str(predicted_display):<10} "
                    f"({predicted_arabic:<5}) | "
                    f"{operation:<11} | "
                    f"time=["
                    f"{segment['start_seconds']:7.3f}, "
                    f"{segment['end_seconds']:7.3f}) | "
                    f"duration="
                    f"{segment['duration_seconds']:7.3f}s | "
                    f"frames="
                    f"{segment['frame_count']:3d}"
                )
        
                row_index += 1
        
        
        # ============================================================
        # No reference target:
        # just print autonomous predictions.
        # ============================================================
        
        else:
        
            for segment in segments:
            
                predicted_phoneme = (
                    segment[
                        "predicted_phoneme"
                    ]
                )
        
                predicted_arabic = (
                    phoneme_to_arabic_text(
                        predicted_phoneme
                    )
                )
        
                print(
                    f"{segment['index']:03d} | "
                    f"target="
                    f"{'-':<10} "
                    f"({'-':<5}) | "
                    f"pred="
                    f"{predicted_phoneme:<10} "
                    f"({predicted_arabic:<5}) | "
                    f"{'-':<11} | "
                    f"time=["
                    f"{segment['start_seconds']:7.3f}, "
                    f"{segment['end_seconds']:7.3f}) | "
                    f"duration="
                    f"{segment['duration_seconds']:7.3f}s | "
                    f"frames="
                    f"{segment['frame_count']:3d}"
                )
        
        print("=" * 155)
        

        # --------------------------------------------------------
        # Show deletions separately because deletions
        # have no predicted boundary.
        # --------------------------------------------------------

        if (
            alignment is not None
            and evaluation["deletions"] > 0
        ):

            print()
            print(
                "Deleted reference phonemes"
            )
            print("-" * 50)

            for item in alignment:

                if (
                    item["operation"]
                    == "delete"
                ):
                    print(
                        f"reference index "
                        f"{item['reference_index']:03d} | "
                        f"{item['target_phoneme']}"
                    )

        print()
        print(
            "Reference phonemes:"
        )
        print(
            correct_target
            if correct_target is not None
            else "-"
        )

        print()
        print(
            "Predicted phonemes:"
        )
        print(
            predicted_phonemes
        )

        print("=" * 100)

    return output

# %%
# import numpy as np
# import sounddevice as sd
# import soundfile as sf

# rs = 16000
# recording = []

# audio_path = "../../datasets/test_audios/1.wav"

# def callback(indata, frames, time, status):
#     if status:
#         print(status)
#     recording.append(indata.copy())

# input("Press Enter to start recording...")

# print("Recording... Press Enter again to stop.")

# with sd.InputStream(samplerate=rs, channels=1, callback=callback):
#     input()

# audio = np.concatenate(recording, axis=0)
# sf.write(audio_path, audio, rs)

# print(f"Saved {audio_path}")

# %%
item = val_df.iloc[1]

audio_path = (
    "../../datasets/Quran_ds/"
    "Quran_ds/audio/audio/"
    + item["path_of_audio"]
)

target_phonemes = ast.literal_eval(
    item["phonemes"]
)


# audio_path =  "../../datasets/test_audios/002004.wav"


# audio_path = "../../datasets/QDAT_Quran_DS/FINAL SOUND/FINAL SOUND/S0_1.wav"
# target_phonemes = ['qaa', 'luu', 'laa', 'ʕi', 'l', 'ma', 'la', 'naa', 'ʔi', 'nn', 'na', 'ka', 'ʔa', 'nt', 'ta', 'ʕa', 'l', 'laa', 'mu', 'l', 'ɣu', 'yuu', 'bK']

print(f"Audio path: {audio_path}")
print(f"Target phonemes: {target_phonemes}")

# %%
output = test_external_audio(
    audio_path=audio_path,
    model=model,
    accelerator=accelerator,
    id_to_phoneme=id_to_phoneme,
    phoneme_to_id=phoneme_to_id,
    # correct_target=target_phonemes,
)

# %%
@torch.inference_mode()
def evaluate_external_dataset(
    csv_path,
    audio_path,
    model,
    accelerator,
    phoneme_to_id,
    audio_path_1=None,
    max_duration=45.0,
    max_padded_audio_seconds=20.0,
    print_results=True,
):
    """
    Evaluate the trained model on an external dataset.

    Expected CSV columns:
        path_of_audio
        phonemes

    Optional columns:
        duration
        ds_index

    Returns:
        CTC loss
        PER
        phoneme accuracy
        aligned match rate
        exact sequence accuracy
        substitutions
        deletions
        insertions
        matches
        reference phoneme count
        predicted phoneme count
        mean count error
        exact count accuracy
        segment loss
        segment accuracy

    Notes:
        - No augmentation is used.
        - Segment accuracy is an auxiliary classifier metric,
          NOT true boundary accuracy.
        - True timing accuracy requires reference boundaries.
    """

    # ============================================================
    # 1. Load dataframe
    # ============================================================

    dataframe = pd.read_csv(
        csv_path
    )

    required_columns = {
        "path_of_audio",
        "phonemes",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required CSV columns: "
            f"{sorted(missing_columns)}"
        )

    original_dataset_size = len(
        dataframe
    )

    if original_dataset_size == 0:
        raise ValueError(
            "The dataset is empty."
        )

    # ============================================================
    # 2. Print duration information
    # ============================================================

    if (
        "duration"
        in dataframe.columns
    ):

        if (
            print_results
            and accelerator.is_main_process
        ):

            print()
            print(
                "External dataset duration statistics"
            )
            print("-" * 50)

            print(
                f"Samples before filtering : "
                f"{len(dataframe)}"
            )

            print(
                f"Minimum duration         : "
                f"{dataframe['duration'].min():.2f}s"
            )

            print(
                f"Mean duration            : "
                f"{dataframe['duration'].mean():.2f}s"
            )

            print(
                f"Maximum duration         : "
                f"{dataframe['duration'].max():.2f}s"
            )

    # ============================================================
    # 3. Filter very long recordings
    # ============================================================

    if (
        max_duration is not None
        and "duration" in dataframe.columns
    ):

        dataframe = dataframe[
            dataframe["duration"]
            <= max_duration
        ].reset_index(
            drop=True
        )

        removed_count = (
            original_dataset_size
            - len(dataframe)
        )

        if (
            print_results
            and accelerator.is_main_process
        ):

            print(
                f"Removed recordings > "
                f"{max_duration:.1f}s : "
                f"{removed_count}"
            )

            print(
                f"Samples after filtering  : "
                f"{len(dataframe)}"
            )

    if len(dataframe) == 0:
        raise ValueError(
            "No samples remain after duration filtering."
        )

    # ============================================================
    # 4. Audio path fallback
    # ============================================================

    if audio_path_1 is None:
        audio_path_1 = audio_path

    # ============================================================
    # 5. Dataset
    # ============================================================

    dataset = TajweedCTCDataset(
        dataframe=dataframe,
        dataset_path=audio_path,
        dataset_path_1=audio_path_1,
        phoneme_to_id=phoneme_to_id,
        training=False,
    )

    # ============================================================
    # 6. Safe evaluation batch budget
    # ============================================================

    evaluation_max_padded_samples = int(
        SR
        * max_padded_audio_seconds
    )

    sampler = DynamicBatchSampler(
        dataset=dataset,
        max_padded_samples_per_batch=(
            evaluation_max_padded_samples
        ),
        shuffle=False,
        minimum_speed_factor=1.0,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=ctc_collate,
        num_workers=NUM_WORKERS,
        pin_memory=(
            torch.cuda.is_available()
        ),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    loader = accelerator.prepare(
        loader
    )

    # ============================================================
    # 7. Evaluation mode
    # ============================================================

    model.eval()

    # ============================================================
    # 8. Accumulators
    # ============================================================

    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0
    total_matches = 0

    total_reference_phonemes = 0
    total_predicted_phonemes = 0

    exact_sequences = 0
    total_sequences = 0

    total_count_error = 0
    exact_count_sequences = 0

    total_ctc_loss = 0.0

    total_segment_loss = 0.0
    total_segment_correct = 0
    total_segment_phonemes = 0

    # ============================================================
    # 9. Evaluation loop
    # ============================================================

    progress = tqdm(
        loader,
        desc="Evaluating external dataset",
        disable=(
            not accelerator.is_local_main_process
        ),
    )

    for (
        waveforms,
        targets,
        input_lengths,
        target_lengths,
    ) in progress:

        # ========================================================
        # Forward
        #
        # target_lengths are supplied here ONLY so we can
        # measure segmentation loss and auxiliary accuracy.
        #
        # CTC recognition remains autonomous.
        # ========================================================

        with accelerator.autocast():

            (
                ctc_logits,
                segment_logits_batch,
                segment_lengths,
                hard_boundaries_batch,
                hidden_lengths,
            ) = model(
                waveforms,
                input_lengths,
                target_lengths=target_lengths,
            )

            # ====================================================
            # CTC loss
            # ====================================================

            ctc_loss = compute_ctc_loss(
                ctc_logits=ctc_logits,
                targets=targets,
                target_lengths=target_lengths,
                hidden_lengths=hidden_lengths,
            )

            # ====================================================
            # Segment loss
            # ====================================================

            segment_loss = segmentation_loss(
                logits_batch=(
                    segment_logits_batch
                ),
                targets=targets,
                target_lengths=target_lengths,
            )

        # ========================================================
        # 10. Autonomous CTC decode
        # ========================================================

        predicted_batch = greedy_ctc_decode(
            ctc_logits=ctc_logits,
            hidden_lengths=hidden_lengths,
        )

        # ========================================================
        # 11. Loss accumulation
        # ========================================================

        batch_phoneme_count = int(
            target_lengths.sum().item()
        )

        total_ctc_loss += (
            float(
                ctc_loss
                .detach()
                .item()
            )
            * batch_phoneme_count
        )

        total_segment_loss += (
            float(
                segment_loss
                .detach()
                .item()
            )
            * batch_phoneme_count
        )

        # ========================================================
        # 12. Segment classifier accuracy
        # ========================================================

        for (
            batch_index,
            segment_logits,
        ) in enumerate(
            segment_logits_batch
        ):

            reference_length = int(
                target_lengths[
                    batch_index
                ].item()
            )

            segment_predictions = (
                torch.argmax(
                    segment_logits,
                    dim=-1,
                )
            )

            segment_targets = (
                targets[
                    batch_index,
                    :reference_length,
                ]
                .to(
                    segment_predictions.device
                )
            )

            total_segment_correct += int(
                (
                    segment_predictions
                    == segment_targets
                )
                .sum()
                .item()
            )

            total_segment_phonemes += (
                reference_length
            )

        # ========================================================
        # 13. CTC sequence metrics
        # ========================================================

        for (
            batch_index,
            predicted_ids,
        ) in enumerate(
            predicted_batch
        ):

            reference_length = int(
                target_lengths[
                    batch_index
                ].item()
            )

            reference_ids = (
                targets[
                    batch_index,
                    :reference_length,
                ]
                .detach()
                .cpu()
                .tolist()
            )

            predicted_ids = list(
                predicted_ids
            )

            alignment = levenshtein_counts(
                reference=reference_ids,
                hypothesis=predicted_ids,
            )

            substitutions = int(
                alignment[
                    "substitutions"
                ]
            )

            deletions = int(
                alignment[
                    "deletions"
                ]
            )

            insertions = int(
                alignment[
                    "insertions"
                ]
            )

            matches = int(
                alignment[
                    "matches"
                ]
            )

            total_substitutions += (
                substitutions
            )

            total_deletions += (
                deletions
            )

            total_insertions += (
                insertions
            )

            total_matches += (
                matches
            )

            total_reference_phonemes += (
                len(reference_ids)
            )

            total_predicted_phonemes += (
                len(predicted_ids)
            )

            errors = (
                substitutions
                + deletions
                + insertions
            )

            exact_sequences += int(
                errors == 0
            )

            total_sequences += 1

            reference_count = len(
                reference_ids
            )

            predicted_count = len(
                predicted_ids
            )

            count_error = abs(
                predicted_count
                - reference_count
            )

            total_count_error += (
                count_error
            )

            exact_count_sequences += int(
                count_error == 0
            )

        # ========================================================
        # 14. Explicit GPU cleanup
        # ========================================================

        del ctc_logits
        del segment_logits_batch
        del segment_lengths
        del hard_boundaries_batch
        del hidden_lengths
        del ctc_loss
        del segment_loss
        del predicted_batch

        del waveforms
        del targets
        del input_lengths
        del target_lengths

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ============================================================
    # 15. Reduce across processes
    # ============================================================

    statistics = torch.tensor(
        [
            total_substitutions,
            total_deletions,
            total_insertions,
            total_matches,

            total_reference_phonemes,
            total_predicted_phonemes,

            exact_sequences,
            total_sequences,

            total_count_error,
            exact_count_sequences,

            total_ctc_loss,
            total_segment_loss,

            total_segment_correct,
            total_segment_phonemes,
        ],
        dtype=torch.float64,
        device=accelerator.device,
    )

    statistics = accelerator.reduce(
        statistics,
        reduction="sum",
    )

    # ============================================================
    # 16. Extract values
    # ============================================================

    substitutions = int(
        statistics[0].item()
    )

    deletions = int(
        statistics[1].item()
    )

    insertions = int(
        statistics[2].item()
    )

    matches = int(
        statistics[3].item()
    )

    reference_phonemes = max(
        int(
            statistics[4].item()
        ),
        1,
    )

    predicted_phonemes = int(
        statistics[5].item()
    )

    exact_sequence_count = int(
        statistics[6].item()
    )

    sequence_count = max(
        int(
            statistics[7].item()
        ),
        1,
    )

    count_error_sum = int(
        statistics[8].item()
    )

    exact_count_count = int(
        statistics[9].item()
    )

    ctc_loss_sum = float(
        statistics[10].item()
    )

    segment_loss_sum = float(
        statistics[11].item()
    )

    segment_correct = int(
        statistics[12].item()
    )

    segment_phoneme_count = max(
        int(
            statistics[13].item()
        ),
        1,
    )

    # ============================================================
    # 17. Final metrics
    # ============================================================

    total_errors = (
        substitutions
        + deletions
        + insertions
    )

    phoneme_error_rate = (
        total_errors
        / reference_phonemes
    )

    phoneme_accuracy = max(
        0.0,
        1.0 - phoneme_error_rate,
    )

    aligned_match_rate = (
        matches
        / reference_phonemes
    )

    exact_sequence_accuracy = (
        exact_sequence_count
        / sequence_count
    )

    mean_count_error = (
        count_error_sum
        / sequence_count
    )

    exact_count_accuracy = (
        exact_count_count
        / sequence_count
    )

    average_ctc_loss = (
        ctc_loss_sum
        / reference_phonemes
    )

    average_segment_loss = (
        segment_loss_sum
        / segment_phoneme_count
    )

    segment_accuracy = (
        segment_correct
        / segment_phoneme_count
    )

    # ============================================================
    # 18. Results dictionary
    # ============================================================

    results = {
        "dataset_samples": (
            sequence_count
        ),

        "ctc_loss": (
            average_ctc_loss
        ),

        "phoneme_error_rate": (
            phoneme_error_rate
        ),

        "phoneme_accuracy": (
            phoneme_accuracy
        ),

        "aligned_match_rate": (
            aligned_match_rate
        ),

        "exact_sequence_accuracy": (
            exact_sequence_accuracy
        ),

        "substitutions": (
            substitutions
        ),

        "deletions": (
            deletions
        ),

        "insertions": (
            insertions
        ),

        "matches": (
            matches
        ),

        "total_errors": (
            total_errors
        ),

        "reference_phonemes": (
            reference_phonemes
        ),

        "predicted_phonemes": (
            predicted_phonemes
        ),

        "mean_count_error": (
            mean_count_error
        ),

        "exact_count_accuracy": (
            exact_count_accuracy
        ),

        "segment_loss": (
            average_segment_loss
        ),

        "segment_accuracy": (
            segment_accuracy
        ),
    }

    # ============================================================
    # 19. Print
    # ============================================================

    if (
        print_results
        and accelerator.is_main_process
    ):

        print()
        print("=" * 70)
        print(
            "EXTERNAL DATASET EVALUATION"
        )
        print("=" * 70)

        print(
            f"Samples                : "
            f"{sequence_count}"
        )

        print()

        print(
            "CTC recognition"
        )
        print("-" * 50)

        print(
            f"CTC loss               : "
            f"{average_ctc_loss:.4f}"
        )

        print(
            f"PER                    : "
            f"{phoneme_error_rate:.2%}"
        )

        print(
            f"Phoneme accuracy       : "
            f"{phoneme_accuracy:.2%}"
        )

        print(
            f"Aligned match rate     : "
            f"{aligned_match_rate:.2%}"
        )

        print(
            f"Exact sequence accuracy: "
            f"{exact_sequence_accuracy:.2%}"
        )

        print(
            f"Substitutions          : "
            f"{substitutions}"
        )

        print(
            f"Deletions              : "
            f"{deletions}"
        )

        print(
            f"Insertions             : "
            f"{insertions}"
        )

        print(
            f"Matches                : "
            f"{matches}"
        )

        print(
            f"Reference phonemes     : "
            f"{reference_phonemes}"
        )

        print(
            f"Predicted phonemes     : "
            f"{predicted_phonemes}"
        )

        print(
            f"Mean count error       : "
            f"{mean_count_error:.4f}"
        )

        print(
            f"Exact count accuracy   : "
            f"{exact_count_accuracy:.2%}"
        )

        print()

        print(
            "Segmentation auxiliary evaluation"
        )
        print("-" * 50)

        print(
            f"Segment loss           : "
            f"{average_segment_loss:.4f}"
        )

        print(
            f"Segment accuracy       : "
            f"{segment_accuracy:.2%}"
        )

        print("=" * 70)

    return results

# %%
# metrics = evaluate_external_dataset(
#     csv_path= '../../datasets/Quran_ds/validation_df_v1.csv',
#     audio_path=DATASET_PATH,
#     model=model,
#     accelerator=accelerator,
#     phoneme_to_id=phoneme_to_id,

#     max_duration=45,
#     max_padded_audio_seconds=20,
# )


