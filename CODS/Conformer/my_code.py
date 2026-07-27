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

# %%
# ? this is for local training

MODEL_PATH = "../../models/count_checkpoint.pth"
WORKING_MODEL_PATH = "../../models/checkpoint.pth"
WORKING_BEST_MODEL_PATH = "../../models/best_checkpoint.pth"
COUNT_MODEL_PATH = "../../models/best_count_checkpoint.pth"

DATASET_PATH = "../../datasets/Quran_ds/Quran_ds/audio/audio/"
DATASET_PATH_1 = "../../datasets/Quran_ds/Quran_ds/audio/audio/"
TRAIN_DS_PATH = "../../datasets/Quran_ds/quran_train.csv"
TEST_DS_PATH = "../../datasets/Quran_ds/quran_test.csv"


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

# %%
len(phoneme_to_id)

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


def add_noise(signal, noise_level=0.003):
    return signal + noise_level * np.random.randn(len(signal))


def random_gain(signal):
    return signal * np.random.uniform(0.8, 1.2)


def load_waveform(audio_path, sr=16000, training=True):

    # audio_path = audio_path.replace(".wav", ".mp3")

    signal = None

    # ── Step 1: try torchaudio — fast, no temp files ──────────────────────
    try:

        waveform, orig_sr = torchaudio.load(audio_path)

        # Stereo → mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if orig_sr != sr:
            waveform = T.Resample(orig_freq=orig_sr, new_freq=sr)(waveform)

        signal = waveform.squeeze(0).numpy()

    # ── Step 2: fallback to librosa for corrupt MP3 frames ────────────────
    except Exception:
        try:
            with suppress_c_stderr():
                signal, _ = librosa.load(audio_path, sr=sr, mono=True)
        except Exception as e:
            raise RuntimeError(f"Both loaders failed for: {audio_path} — {e}")

    # ── Step 3: guard against silent/empty output ─────────────────────────
    if signal is None or len(signal) == 0:
        raise RuntimeError(f"Audio is empty: {audio_path}")

    if np.max(np.abs(signal)) < 1e-6:
        raise RuntimeError(f"Audio appears silent or corrupt: {audio_path}")

    # ── Step 4: augmentation (training only) ──────────────────────────────
    if training and np.random.rand() < 0.5:
        signal = add_noise(signal)

    if training and np.random.rand() < 0.5:
        signal = random_gain(signal)

    # ── Step 5: normalize to [-1, 1] ──────────────────────────────────────
    max_val = np.max(np.abs(signal)) + 1e-8
    signal = signal / max_val

    return torch.tensor(signal, dtype=torch.float32)

# %%
class DynamicBatchSampler(torch.utils.data.Sampler):
    """
    Groups samples by audio length so each batch contains
    similarly-lengthed ayahs — minimizes padding waste and OOM risk.
    """

    def __init__(self, dataset, max_samples_per_batch, shuffle=True):
        self.dataset = dataset
        self.max_samples_per_batch = max_samples_per_batch
        self.shuffle = shuffle

        # Pre-read waveform lengths from the dataframe
        # (avoids loading audio just to know the length)
        print("Building length index...")
        self.lengths = []
        for idx in range(len(dataset)):
            row = dataset.df.iloc[idx]

            audio_path = ""
            if row["ds_index"] == 1:
                audio_path = os.path.join(dataset.dataset_path, row["path_of_audio"])
            else:
                audio_path = os.path.join(dataset.dataset_path_1, row["path_of_audio"])

            try:
                info = torchaudio.info(audio_path)
                # Resample length if needed
                length = int(info.num_frames * SR / info.sample_rate)
            except Exception:
                length = SR * 10  # fallback: assume 10 seconds
            self.lengths.append(length)
        print(f"Length index built for {len(self.lengths)} samples")

    def __iter__(self):
        batches = self._build_batches()
        for batch in batches:
            yield batch

    def _build_batches(self):
        import random

        indices = list(range(len(self.lengths)))
        if self.shuffle:
            random.shuffle(indices)
        indices.sort(key=lambda i: self.lengths[i])

        batches = []
        current_batch = []
        current_max_len = 0

        for idx in indices:
            length = self.lengths[idx]
            new_max = max(current_max_len, length)
            if (
                current_batch
                and (len(current_batch) + 1) * new_max > self.max_samples_per_batch
            ):
                batches.append(current_batch)
                current_batch = [idx]
                current_max_len = length
            else:
                current_batch.append(idx)
                current_max_len = new_max

        if current_batch:
            batches.append(current_batch)

        if self.shuffle:
            random.shuffle(batches)

        return batches

    def __len__(self):
        # Build once with no shuffle to get stable count
        indices = list(range(len(self.lengths)))
        indices.sort(key=lambda i: self.lengths[i])
        batches = []
        current_batch = []
        current_max_len = 0
        for idx in indices:
            length = self.lengths[idx]
            new_max = max(current_max_len, length)
            if (
                current_batch
                and (len(current_batch) + 1) * new_max > self.max_samples_per_batch
            ):
                batches.append(current_batch)
                current_batch = [idx]
                current_max_len = length
            else:
                current_batch.append(idx)
                current_max_len = new_max
        if current_batch:
            batches.append(current_batch)
        return len(batches)

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

# %%
class SegmentClassifier(nn.Module):
    def __init__(self, embedding_dim=64, vocab_size=len(phoneme_to_id)):
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        return self.classifier(x)

# %%
class SegmentationHead(nn.Module):
    def __init__(self, hidden_dim=512, embedding_dim=64):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
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

# %%
wav2vec2_model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", ignore_mismatched_sizes=True,local_files_only=True
)

for param in wav2vec2_model.parameters():
    param.requires_grad = True

# %%
class ASRModel(torch.nn.Module):

    def __init__(self, wav2vec2, spec_augment):
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
            dropout=0.1,
        )

        self.segmentation = SegmentationHead(
            hidden_dim=512,
            embedding_dim=64,
        )

        self.segment_classifier = SegmentClassifier(
            embedding_dim=64,
            vocab_size=len(phoneme_to_id),
        )

        self.count_head = nn.Sequential(
            nn.LayerNorm(1025),
            nn.Linear(1025, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
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

    def predict_phoneme_count(
        self,
        hidden,
        hidden_lengths,
    ):
        B, T, H = hidden.shape

        hidden_float = hidden.float()

        frame_indices = torch.arange(
            T,
            device=hidden.device,
        ).unsqueeze(0)

        valid_mask = (
            frame_indices
            < hidden_lengths.unsqueeze(1)
        )

        valid_mask = valid_mask.unsqueeze(
            -1
        ).to(hidden_float.dtype)

        valid_lengths = hidden_lengths.clamp_min(
            1
        ).to(hidden_float.dtype).unsqueeze(1)

        hidden_mean = (
            hidden_float * valid_mask
        ).sum(dim=1) / valid_lengths

        centered_hidden = (
            hidden_float
            - hidden_mean.unsqueeze(1)
        )

        hidden_variance = (
            centered_hidden.pow(2)
            * valid_mask
        ).sum(dim=1) / valid_lengths

        hidden_std = torch.sqrt(
            hidden_variance + 1e-5
        )

        log_hidden_length = torch.log1p(
            hidden_lengths.to(
                hidden_float.dtype
            )
        ).unsqueeze(1)

        count_features = torch.cat(
            [
                hidden_mean,
                hidden_std,
                log_hidden_length,
            ],
            dim=1,
        )

        predicted_counts = self.count_head(
            count_features
        ).squeeze(-1)

        return predicted_counts

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
        count_only=False,
    ):
        hidden, hidden_lengths = (
            self.encode_audio(
                waveforms,
                input_lengths,
            )
        )
    
        predicted_counts = (
            self.predict_phoneme_count(
                hidden,
                hidden_lengths,
            )
        )
    
        if count_only:
            return (
                predicted_counts,
                hidden_lengths,
            )
    
        (
            segment_embedding,
            progress_weights,
        ) = self.segmentation(
            hidden
        )
    
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
    
            logits_batch = [
                self.segment_classifier(
                    pooled
                )
                for pooled in pooled_batch
            ]
    
            return (
                logits_batch,
                segment_lengths,
                hard_boundaries_batch,
                hidden_lengths,
            )
    
        predicted_lengths = torch.round(
            predicted_counts
        ).long()
    
        predicted_lengths = torch.clamp(
            predicted_lengths,
            min=1,
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
    
        logits_batch = [
            self.segment_classifier(
                pooled
            )
            for pooled in pooled_batch
        ]
    
        return (
            logits_batch,
            segment_lengths,
            hard_boundaries_batch,
            hidden_lengths,
            predicted_counts,
            predicted_lengths,
        )
    

# %%
train_df = pd.read_csv(TRAIN_DS_PATH)
val_df = pd.read_csv(TEST_DS_PATH)

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


MAX_TOKENS =  BATCH_SIZE * 16000 * 20

train_sampler = DynamicBatchSampler(
    train_dataset,
    max_samples_per_batch=MAX_TOKENS,
    shuffle=True,
)

val_sampler = DynamicBatchSampler(
    val_dataset,
    max_samples_per_batch=MAX_TOKENS,
    shuffle=False,
)

train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    # batch_size=BATCH_SIZE,
    collate_fn=ctc_collate,
    num_workers=1,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_sampler=val_sampler,
    # batch_size=BATCH_SIZE,
    collate_fn=ctc_collate,
    num_workers=1,
    pin_memory=True,
)

# %%
# next(enumerate(train_loader))

# %%
def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_loss,
    best_val_loss,
    epochs_no_improve,
    warmup_scheduler,
    plateau_scheduler,
    path,
):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "warmup_scheduler_state": warmup_scheduler.state_dict(),
            "plateau_scheduler_state": plateau_scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "epochs_no_improve": epochs_no_improve,
        },
        path,
    )

def load_checkpoint(path):
    checkpoint = torch.load(path)
    return checkpoint

# %%
model = ASRModel(
    wav2vec2_model,
    SpecAugment(),
)

del wav2vec2_model

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
)

load_result = model.load_state_dict(
    checkpoint["model_state"],
    strict=False,
)

print(
    "Missing keys:",
    load_result.missing_keys,
)

print(
    "Unexpected keys:",
    load_result.unexpected_keys,
)

invalid_missing_keys = [
    key
    for key in load_result.missing_keys
    if not key.startswith(
        "count_head."
    )
]

if invalid_missing_keys:
    raise RuntimeError(
        f"Unexpected missing keys: "
        f"{invalid_missing_keys}"
    )

if load_result.unexpected_keys:
    raise RuntimeError(
        f"Unexpected checkpoint keys: "
        f"{load_result.unexpected_keys}"
    )

for parameter in model.parameters():
    parameter.requires_grad = False

for parameter in (
    model.count_head.parameters()
):
    parameter.requires_grad = True

trainable_parameters = [
    name
    for name, parameter
    in model.named_parameters()
    if parameter.requires_grad
]

print(
    "Trainable parameters:"
)

for name in trainable_parameters:
    print(name)

count_optimizer = torch.optim.AdamW(
    model.count_head.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)

accelerator = Accelerator(
    mixed_precision="fp16"
)


(
    model,
    count_optimizer,
    train_loader,
    val_loader,
) = accelerator.prepare(
    model,
    count_optimizer,
    train_loader,
    val_loader,
)



# %%
def train_count_head(
    model,
    train_loader,
    val_loader,
    optimizer,
    accelerator,
    checkpoint_path,
    epochs=20,
    patience=5,
):
    best_val_mae = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.eval()

        accelerator.unwrap_model(
            model
        ).count_head.train()

        train_loss_sum = 0.0
        train_absolute_error_sum = 0.0
        train_sample_count = 0

        train_progress = tqdm(
            train_loader,
            desc=(
                f"Count training "
                f"{epoch + 1}/{epochs}"
            ),
            disable=(
                not accelerator
                .is_local_main_process
            ),
        )

        for (
            waveforms,
            targets,
            input_lengths,
            target_lengths,
        ) in train_progress:
            optimizer.zero_grad(
                set_to_none=True
            )

            (
                predicted_counts,
                hidden_lengths,
            ) = model(
                waveforms,
                input_lengths,
                count_only=True,
            )

            target_counts = (
                target_lengths.float()
            )

            batch_loss_sum = (
                F.smooth_l1_loss(
                    predicted_counts,
                    target_counts,
                    reduction="sum",
                )
            )

            batch_size = (
                target_counts.numel()
            )

            loss = (
                batch_loss_sum
                / batch_size
            )

            accelerator.backward(
                loss
            )

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

            optimizer.step()

            with torch.no_grad():
                absolute_errors = torch.abs(
                    predicted_counts
                    - target_counts
                )

            train_loss_sum += float(
                batch_loss_sum
                .detach()
                .item()
            )

            train_absolute_error_sum += float(
                absolute_errors
                .sum()
                .item()
            )

            train_sample_count += (
                batch_size
            )

            train_progress.set_postfix(
                loss=(
                    f"{loss.item():.4f}"
                )
            )

        train_stats = torch.tensor(
            [
                train_loss_sum,
                train_absolute_error_sum,
                train_sample_count,
            ],
            dtype=torch.float64,
            device=accelerator.device,
        )

        train_stats = accelerator.reduce(
            train_stats,
            reduction="sum",
        )

        global_train_loss = (
            train_stats[0].item()
            / max(
                train_stats[2].item(),
                1,
            )
        )

        global_train_mae = (
            train_stats[1].item()
            / max(
                train_stats[2].item(),
                1,
            )
        )

        model.eval()

        val_loss_sum = 0.0
        val_absolute_error_sum = 0.0
        val_rounded_error_sum = 0.0
        val_bias_sum = 0.0
        val_exact_count = 0
        val_within_one_count = 0
        val_within_two_count = 0
        val_sample_count = 0

        val_progress = tqdm(
            val_loader,
            desc=(
                f"Count validation "
                f"{epoch + 1}/{epochs}"
            ),
            disable=(
                not accelerator
                .is_local_main_process
            ),
        )

        with torch.no_grad():
            for (
                waveforms,
                targets,
                input_lengths,
                target_lengths,
            ) in val_progress:
                (
                    predicted_counts,
                    hidden_lengths,
                ) = model(
                    waveforms,
                    input_lengths,
                    count_only=True,
                )

                target_counts = (
                    target_lengths.float()
                )

                batch_loss_sum = (
                    F.smooth_l1_loss(
                        predicted_counts,
                        target_counts,
                        reduction="sum",
                    )
                )

                rounded_counts = torch.round(
                    predicted_counts
                ).long()

                rounded_counts = torch.clamp(
                    rounded_counts,
                    min=1,
                )

                rounded_counts = torch.minimum(
                    rounded_counts,
                    hidden_lengths.long(),
                )

                count_errors = (
                    rounded_counts
                    - target_lengths.long()
                )

                absolute_errors = torch.abs(
                    predicted_counts
                    - target_counts
                )

                rounded_absolute_errors = (
                    torch.abs(
                        count_errors
                    )
                )

                batch_size = (
                    target_counts.numel()
                )

                val_loss_sum += float(
                    batch_loss_sum.item()
                )

                val_absolute_error_sum += float(
                    absolute_errors
                    .sum()
                    .item()
                )

                val_rounded_error_sum += float(
                    rounded_absolute_errors
                    .sum()
                    .item()
                )

                val_bias_sum += float(
                    (
                        predicted_counts
                        - target_counts
                    )
                    .sum()
                    .item()
                )

                val_exact_count += int(
                    (
                        rounded_absolute_errors
                        == 0
                    )
                    .sum()
                    .item()
                )

                val_within_one_count += int(
                    (
                        rounded_absolute_errors
                        <= 1
                    )
                    .sum()
                    .item()
                )

                val_within_two_count += int(
                    (
                        rounded_absolute_errors
                        <= 2
                    )
                    .sum()
                    .item()
                )

                val_sample_count += (
                    batch_size
                )

        val_stats = torch.tensor(
            [
                val_loss_sum,
                val_absolute_error_sum,
                val_rounded_error_sum,
                val_bias_sum,
                val_exact_count,
                val_within_one_count,
                val_within_two_count,
                val_sample_count,
            ],
            dtype=torch.float64,
            device=accelerator.device,
        )

        val_stats = accelerator.reduce(
            val_stats,
            reduction="sum",
        )

        total_samples = max(
            int(
                val_stats[7].item()
            ),
            1,
        )

        avg_val_loss = (
            val_stats[0].item()
            / total_samples
        )

        val_mae = (
            val_stats[1].item()
            / total_samples
        )

        val_rounded_mae = (
            val_stats[2].item()
            / total_samples
        )

        val_bias = (
            val_stats[3].item()
            / total_samples
        )

        exact_accuracy = (
            val_stats[4].item()
            / total_samples
        )

        within_one_accuracy = (
            val_stats[5].item()
            / total_samples
        )

        within_two_accuracy = (
            val_stats[6].item()
            / total_samples
        )

        if accelerator.is_main_process:
            print()
            print(
                f"Epoch {epoch + 1}/{epochs}"
            )
            print(
                f"Train count loss : "
                f"{global_train_loss:.4f}"
            )
            print(
                f"Train count MAE  : "
                f"{global_train_mae:.3f}"
            )
            print(
                f"Val count loss   : "
                f"{avg_val_loss:.4f}"
            )
            print(
                f"Val raw MAE      : "
                f"{val_mae:.3f}"
            )
            print(
                f"Val rounded MAE  : "
                f"{val_rounded_mae:.3f}"
            )
            print(
                f"Val bias         : "
                f"{val_bias:+.3f}"
            )
            print(
                f"Exact count      : "
                f"{exact_accuracy:.2%}"
            )
            print(
                f"Within ±1        : "
                f"{within_one_accuracy:.2%}"
            )
            print(
                f"Within ±2        : "
                f"{within_two_accuracy:.2%}"
            )

        improved = (
            val_mae < best_val_mae
        )

        if improved:
            best_val_mae = val_mae
            epochs_without_improvement = 0

            if accelerator.is_main_process:
                unwrapped_model = (
                    accelerator
                    .unwrap_model(model)
                )

                torch.save(
                    {
                        "model_state": (
                            unwrapped_model
                            .state_dict()
                        ),
                        "optimizer_state": (
                            optimizer
                            .state_dict()
                        ),
                        "epoch": epoch,
                        "best_val_mae": (
                            best_val_mae
                        ),
                        "val_exact_accuracy": (
                            exact_accuracy
                        ),
                        "val_within_one": (
                            within_one_accuracy
                        ),
                        "val_within_two": (
                            within_two_accuracy
                        ),
                    },
                    checkpoint_path,
                )

                print(
                    "Best count model saved"
                )
        else:
            epochs_without_improvement += 1

        accelerator.wait_for_everyone()

        if (
            epochs_without_improvement
            >= patience
        ):
            if accelerator.is_main_process:
                print(
                    "Count training stopped "
                    "early"
                )

            break

    return best_val_mae

# %%
best_count_mae = train_count_head(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=count_optimizer,
    accelerator=accelerator,
    checkpoint_path=COUNT_MODEL_PATH,
    epochs=20,
    patience=5,
)

# %%
count_checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
)

accelerator.unwrap_model(
    model
).load_state_dict(
    count_checkpoint[
        "model_state"
    ]
)

model.eval()

# %%
batch = next(
    iter(val_loader)
)

(
    waveforms,
    targets,
    input_lengths,
    target_lengths,
) = batch

with torch.no_grad():
    (
        logits_batch,
        segment_lengths,
        hard_boundaries_batch,
        hidden_lengths,
        predicted_counts,
        predicted_lengths,
    ) = model(
        waveforms,
        input_lengths,
    )

print(
    "True lengths:",
    target_lengths.detach().cpu().tolist(),
)

print(
    "Raw predicted counts:",
    predicted_counts.detach().cpu().tolist(),
)

print(
    "Rounded predicted lengths:",
    predicted_lengths.detach().cpu().tolist(),
)

for batch_index, logits in enumerate(
    logits_batch
):
    predicted_phonemes = [
        id_to_phoneme[
            int(token_id)
        ]
        for token_id in torch.argmax(
            logits,
            dim=-1,
        )
        .detach()
        .cpu()
        .tolist()
    ]

    print()
    print(
        f"Sample {batch_index}"
    )
    print(
        "Predicted phonemes:",
        predicted_phonemes,
    )

# %%
# ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
# accelerator = Accelerator(mixed_precision="fp16", kwargs_handlers=[ddp_kwargs])

# print(f"Using device: {DEVICE}")
# print(f"Num processes: {accelerator.num_processes}")


# model = ASRModel(wav2vec2_model, SpecAugment())

# del wav2vec2_model


# WARMUP_EPOCHS = 5
# WAV2VEC2_LR = 1e-6
# CONFORMER_LR = 1e-4
# SEGMENTATION_LR = 2e-4
# CLASSIFIER_LR = 3e-4

# optimizer = torch.optim.AdamW(
#     [
#         {
#             "params": model.wav2vec2.parameters(),
#             "lr": WAV2VEC2_LR,
#         },
#         {
#             "params": model.conformer.parameters(),
#             "lr": CONFORMER_LR,
#         },
#         {
#             "params": model.segmentation.parameters(),
#             "lr": SEGMENTATION_LR,
#         },
#         {
#             "params": model.segment_classifier.parameters(),
#             "lr": CLASSIFIER_LR,
#         },
#     ],
#     weight_decay=0.01,
# )


# # Linear warmup for first WARMUP_EPOCHS, then hand off to ReduceLROnPlateau
# def warmup_lambda(epoch):
#     if epoch < WARMUP_EPOCHS:
#         return (epoch + 1) / WARMUP_EPOCHS  # 0.2, 0.4, 0.6, 0.8, 1.0
#     return 1.0  # after warmup, LR stays at target — plateau scheduler takes over


# warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

# plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     mode="min",
#     factor=0.5,
#     patience=3,
#     min_lr=1e-9,
# )

# model, optimizer, train_loader, val_loader = accelerator.prepare(
#     model, optimizer, train_loader, val_loader
# )


# best_val_loss = float("inf")
# epochs_no_improve = 0
# start_epoch = 0

# if os.path.exists(MODEL_PATH):
#     print("Loading checkpoint...")
#     ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
#     accelerator.unwrap_model(model).load_state_dict(ckpt["model_state"])

#     optimizer.load_state_dict(ckpt["optimizer_state"])

#     # Safe load — handles both old and new checkpoint formats
#     if "warmup_scheduler_state" in ckpt:
#         warmup_scheduler.load_state_dict(ckpt["warmup_scheduler_state"])
#         print("Warmup scheduler state restored ✅")
#     else:
#         print("⚠️ No warmup scheduler state found — starting fresh (old checkpoint)")

#     if "plateau_scheduler_state" in ckpt:
#         plateau_scheduler.load_state_dict(ckpt["plateau_scheduler_state"])
#         print("Plateau scheduler state restored ✅")
#     elif "scheduler_state" in ckpt:
#         # Old checkpoint had a single scheduler — load into plateau scheduler
#         plateau_scheduler.load_state_dict(ckpt["scheduler_state"])
#         print("Plateau scheduler restored from old checkpoint format ✅")
#     else:
#         print("⚠️ No plateau scheduler state found — starting fresh")

#     best_val_loss = ckpt["best_val_loss"]
#     epochs_no_improve = ckpt["epochs_no_improve"]
#     start_epoch = ckpt["epoch"] + 1
#     print(f"Resumed from epoch {start_epoch}")
#     print(f"Best val loss so far: {best_val_loss:.4f}")
#     print("done loading")
# else:
#     print("Starting fresh training")
#     best_val_loss = float("inf")
#     epochs_no_improve = 0
#     start_epoch = 0

# %%
# def segmentation_loss(
#     logits_batch,
#     targets,
#     target_lengths,
# ):
#     all_logits = []
#     all_targets = []

#     for b, logits in enumerate(logits_batch):
#         N = int(target_lengths[b].item())

#         target = targets[b, :N].long()
#         target = target.to(logits.device)

#         all_logits.append(logits)
#         all_targets.append(target)

#     all_logits = torch.cat(all_logits, dim=0)
#     all_targets = torch.cat(all_targets, dim=0)

#     return F.cross_entropy(
#         all_logits,
#         all_targets,
#     )

# %%
# import gc

# model_wights = torch.load("best_model.pth",weights_only=False)

# model.load_state_dict(model_wights)

# print(gc.collect())


# for waveforms, targets, input_lengths, target_lengths in tqdm(train_loader):

#     print("*" * 40)
#     print("Waveforms shape:", waveforms.shape)

#     audio_path = (
#         f"../../datasets/Quran_ds/Quran_ds/audio/audio/{train_df.iloc[1]['path_of_audio']}"
#     )
    
#     # waveforms, sr = torchaudio.load(audio_path)
#     # input_lengths = torch.tensor([waveforms.shape[1]], dtype=torch.long, device=DEVICE)

#     # target_lengths = torch.tensor(
#     #     [len(ast.literal_eval(train_df.iloc[1]["phonemes"]))], dtype=torch.long
#     # ).to(DEVICE)

#     # targets = torch.tensor(
#     #     [[phoneme_to_id[p] for p in ast.literal_eval(train_df.iloc[1]["phonemes"])]],
#     #     dtype=torch.long,
#     # ).to(DEVICE)

#     # target_lengths = torch.tensor([targets.shape[1]], dtype=torch.long).to(DEVICE)


#     logits_batch, segment_lengths, hidden_lengths = model(
#         waveforms.to(DEVICE), input_lengths, target_lengths
#     )


#     print("Waveforms:", waveforms.shape)
#     print("Input lengths:", input_lengths)
#     print("target_lengths:", target_lengths)
#     print("Targets shape:", targets.shape)
#     print("Targets:", targets)
#     print()
#     print("hidden_lengths  :", hidden_lengths)
#     print("Logits:", logits_batch[0].shape)
#     print("Segment lengths:", segment_lengths[0].shape)

#     print("***" * 40)

#     total_frames = sum([sl.sum().item() for sl in segment_lengths])

#     print(f"Total frames assigned: {total_frames}")

#     model_output = [
#         (pred, segment_lengths[0][i].item())
#         for i, pred in enumerate(
#             [
#                 id_to_phoneme[p.item()]
#                 for p in torch.stack(
#                     [torch.argmax(logits, dim=-1).cpu() for logits in logits_batch[0]]
#                 )
#             ]
#         )
#     ]

#     print("preds", model_output)

#     if targets.dim() == 1:
#         targets_batch = targets.unsqueeze(0)
#     else:
#         targets_batch = targets

#     loss = segmentation_loss(
#         logits_batch=logits_batch,
#         targets=targets_batch,
#         target_lengths=target_lengths,
#         segment_lengths=segment_lengths,
#         hidden_lengths=hidden_lengths,
#     )

#     print("Loss:", loss.item())

# loss.backward()

# print("Backward OK")

# %%
model

# %%
print("training is starting .......")

# %%
# def train_model(
#     model,
#     train_loader,
#     val_loader,
#     optimizer,
#     warmup_scheduler,
#     plateau_scheduler,
#     accelerator,
#     epochs=30,
#     patience=6,
#     best_val_loss=float("inf"),
#     epochs_no_improve=0,
#     start_epoch=0,
#     warmup_epochs=WARMUP_EPOCHS,
#     print_boundaries_every=1,
# ):
#     end_epoch = start_epoch + epochs

#     for epoch in range(start_epoch, end_epoch):

#         model.train()

#         train_loss_sum = 0.0
#         train_segment_count = 0

#         train_progress = tqdm(
#             train_loader,
#             desc=f"Training epoch {epoch + 1}",
#             disable=not accelerator.is_local_main_process,
#         )

#         for (
#             waveforms,
#             targets,
#             input_lengths,
#             target_lengths,
#         ) in train_progress:

#             optimizer.zero_grad(set_to_none=True)

#             (
#                 logits_batch,
#                 segment_lengths,
#                 hard_boundaries_batch,
#                 hidden_lengths,
#             ) = model(
#                 waveforms,
#                 input_lengths,
#                 target_lengths,
#             )

#             loss = segmentation_loss(
#                 logits_batch=logits_batch,
#                 targets=targets,
#                 target_lengths=target_lengths,
#             )

#             accelerator.backward(loss)

#             if accelerator.sync_gradients:
#                 accelerator.clip_grad_norm_(
#                     model.parameters(),
#                     max_norm=5.0,
#                 )

#             optimizer.step()

#             batch_segment_count = int(
#                 target_lengths.sum().item()
#             )

#             train_loss_sum += (
#                 loss.detach().item()
#                 * batch_segment_count
#             )

#             train_segment_count += batch_segment_count

#             train_progress.set_postfix(
#                 loss=f"{loss.detach().item():.4f}"
#             )

#         train_stats = torch.tensor(
#             [
#                 train_loss_sum,
#                 train_segment_count,
#             ],
#             dtype=torch.float64,
#             device=accelerator.device,
#         )

#         train_stats = accelerator.reduce(
#             train_stats,
#             reduction="sum",
#         )

#         global_train_loss_sum = train_stats[0].item()
#         global_train_segment_count = int(
#             train_stats[1].item()
#         )

#         if global_train_segment_count == 0:
#             raise RuntimeError(
#                 "Training loader produced zero target segments"
#             )

#         avg_train_loss = (
#             global_train_loss_sum
#             / global_train_segment_count
#         )

#         model.eval()

#         val_loss_sum = 0.0
#         val_correct_segments = 0
#         val_segment_count = 0

#         boundary_debug = None

#         val_progress = tqdm(
#             val_loader,
#             desc=f"Validation epoch {epoch + 1}",
#             disable=not accelerator.is_local_main_process,
#         )

#         with torch.no_grad():
#             for (
#                 waveforms,
#                 targets,
#                 input_lengths,
#                 target_lengths,
#             ) in val_progress:

#                 (
#                     logits_batch,
#                     segment_lengths,
#                     hard_boundaries_batch,
#                     hidden_lengths,
#                 ) = model(
#                     waveforms,
#                     input_lengths,
#                     target_lengths,
#                 )

#                 loss = segmentation_loss(
#                     logits_batch=logits_batch,
#                     targets=targets,
#                     target_lengths=target_lengths,
#                 )

#                 batch_segment_count = int(
#                     target_lengths.sum().item()
#                 )

#                 val_loss_sum += (
#                     loss.detach().item()
#                     * batch_segment_count
#                 )

#                 val_segment_count += batch_segment_count

#                 for b, logits in enumerate(logits_batch):

#                     N = int(
#                         target_lengths[b].item()
#                     )

#                     predictions = torch.argmax(
#                         logits,
#                         dim=-1,
#                     )

#                     target = targets[b, :N].to(
#                         predictions.device
#                     )

#                     val_correct_segments += int(
#                         (
#                             predictions == target
#                         ).sum().item()
#                     )

#                 if (
#                     boundary_debug is None
#                     and accelerator.is_main_process
#                 ):
#                     sample_index = 0

#                     N = int(
#                         target_lengths[
#                             sample_index
#                         ].item()
#                     )

#                     boundary_debug = {
#                         "predictions": torch.argmax(
#                             logits_batch[sample_index],
#                             dim=-1,
#                         ).detach().cpu(),
#                         "targets": targets[
#                             sample_index,
#                             :N,
#                         ].detach().cpu(),
#                         "boundaries": hard_boundaries_batch[
#                             sample_index
#                         ].detach().cpu(),
#                         "segment_masses": segment_lengths[
#                             sample_index
#                         ].detach().cpu(),
#                         "input_length": int(
#                             input_lengths[
#                                 sample_index
#                             ].item()
#                         ),
#                         "hidden_length": int(
#                             hidden_lengths[
#                                 sample_index
#                             ].item()
#                         ),
#                         "target_length": N,
#                     }

#                 val_progress.set_postfix(
#                     loss=f"{loss.detach().item():.4f}"
#                 )

#         validation_stats = torch.tensor(
#             [
#                 val_loss_sum,
#                 val_correct_segments,
#                 val_segment_count,
#             ],
#             dtype=torch.float64,
#             device=accelerator.device,
#         )

#         validation_stats = accelerator.reduce(
#             validation_stats,
#             reduction="sum",
#         )

#         global_val_loss_sum = (
#             validation_stats[0].item()
#         )

#         global_val_correct_segments = int(
#             validation_stats[1].item()
#         )

#         global_val_segment_count = int(
#             validation_stats[2].item()
#         )

#         if global_val_segment_count == 0:
#             raise RuntimeError(
#                 "Validation loader produced zero target segments"
#             )

#         avg_val_loss = (
#             global_val_loss_sum
#             / global_val_segment_count
#         )

#         val_accuracy = (
#             global_val_correct_segments
#             / global_val_segment_count
#         )

#         if epoch < warmup_epochs:
#             warmup_scheduler.step()
#         else:
#             plateau_scheduler.step(
#                 avg_val_loss
#             )

#         if accelerator.is_main_process:

#             print()
#             print(
#                 f"Epoch {epoch + 1}/{end_epoch}"
#             )
#             print(
#                 f"  Train Loss   : "
#                 f"{avg_train_loss:.4f}"
#             )
#             print(
#                 f"  Val Loss     : "
#                 f"{avg_val_loss:.4f}"
#             )
#             print(
#                 f"  Val Accuracy : "
#                 f"{val_accuracy:.2%}"
#             )
#             print(
#                 f"  Correct      : "
#                 f"{global_val_correct_segments}"
#                 f"/{global_val_segment_count}"
#             )

#             for group_index, group in enumerate(
#                 optimizer.param_groups
#             ):
#                 print(
#                     f"  LR group {group_index}: "
#                     f"{group['lr']:.8g}"
#                 )

#             print("-" * 60)

#         should_print_boundaries = (
#             print_boundaries_every is not None
#             and print_boundaries_every > 0
#             and (epoch + 1) % print_boundaries_every == 0
#         )

#         if (
#             accelerator.is_main_process
#             and should_print_boundaries
#             and boundary_debug is not None
#         ):
#             unwrapped_model = (
#                 accelerator.unwrap_model(model)
#             )

#             boundary_data = (
#                 unwrapped_model.boundaries_to_seconds(
#                     boundaries=boundary_debug[
#                         "boundaries"
#                     ],
#                     input_length=boundary_debug[
#                         "input_length"
#                     ],
#                     hidden_length=boundary_debug[
#                         "hidden_length"
#                     ],
#                     sample_rate=SR,
#                 )
#             )

#             predictions = boundary_debug[
#                 "predictions"
#             ]

#             target_ids = boundary_debug[
#                 "targets"
#             ]

#             segment_masses = boundary_debug[
#                 "segment_masses"
#             ]

#             N = boundary_debug[
#                 "target_length"
#             ]

#             print()
#             print(
#                 "Predicted phoneme boundaries"
#             )
#             print(
#                 "=" * 100
#             )

#             for segment_index in range(N):

#                 prediction_id = int(
#                     predictions[
#                         segment_index
#                     ].item()
#                 )

#                 target_id = int(
#                     target_ids[
#                         segment_index
#                     ].item()
#                 )

#                 predicted_phoneme = (
#                     id_to_phoneme[
#                         prediction_id
#                     ]
#                 )

#                 target_phoneme = (
#                     id_to_phoneme[
#                         target_id
#                     ]
#                 )

#                 boundary = boundary_data[
#                     segment_index
#                 ]

#                 segment_mass = float(
#                     segment_masses[
#                         segment_index
#                     ].item()
#                 )

#                 is_correct = (
#                     prediction_id == target_id
#                 )

#                 status = (
#                     "correct"
#                     if is_correct
#                     else "wrong"
#                 )

#                 print(
#                     f"{segment_index:03d} | "
#                     f"target={target_phoneme:<8} | "
#                     f"pred={predicted_phoneme:<8} | "
#                     f"{status:<7} | "
#                     f"frames=["
#                     f"{boundary['start_frame']:4d}, "
#                     f"{boundary['end_frame']:4d}) | "
#                     f"time=["
#                     f"{boundary['start_seconds']:7.3f}, "
#                     f"{boundary['end_seconds']:7.3f}) s | "
#                     f"duration="
#                     f"{boundary['duration_seconds']:6.3f} s | "
#                     f"soft_mass="
#                     f"{segment_mass:6.2f}"
#                 )

#             print(
#                 "=" * 100
#             )
#             print()

#         improved = (
#             avg_val_loss < best_val_loss
#         )

#         if improved:
#             best_val_loss = avg_val_loss
#             epochs_no_improve = 0
#         else:
#             epochs_no_improve += 1

#         accelerator.wait_for_everyone()

#         if accelerator.is_main_process:

#             unwrapped_model = (
#                 accelerator.unwrap_model(model)
#             )

#             if improved:
#                 save_checkpoint(
#                     model=unwrapped_model,
#                     optimizer=optimizer,
#                     epoch=epoch,
#                     val_loss=avg_val_loss,
#                     best_val_loss=best_val_loss,
#                     epochs_no_improve=epochs_no_improve,
#                     warmup_scheduler=warmup_scheduler,
#                     plateau_scheduler=plateau_scheduler,
#                     path=WORKING_BEST_MODEL_PATH,
#                 )

#                 print(
#                     "Best model saved"
#                 )

#             save_checkpoint(
#                 model=unwrapped_model,
#                 optimizer=optimizer,
#                 epoch=epoch,
#                 val_loss=avg_val_loss,
#                 best_val_loss=best_val_loss,
#                 epochs_no_improve=epochs_no_improve,
#                 warmup_scheduler=warmup_scheduler,
#                 plateau_scheduler=plateau_scheduler,
#                 path=WORKING_MODEL_PATH,
#             )

#         accelerator.wait_for_everyone()

#         if epochs_no_improve >= patience:

#             if accelerator.is_main_process:
#                 print(
#                     f"Early stopping at epoch "
#                     f"{epoch + 1}"
#                 )

#             break

#     return (
#         best_val_loss,
#         epochs_no_improve,
#     )

# %%
# best_val_loss, epochs_no_improve = train_model(
#     model=model,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     optimizer=optimizer,
#     warmup_scheduler=warmup_scheduler,
#     plateau_scheduler=plateau_scheduler,
#     accelerator=accelerator,
#     epochs=NUM_EPOCHS,
#     patience=6,
#     best_val_loss=best_val_loss,
#     epochs_no_improve=epochs_no_improve,
#     start_epoch=start_epoch,
#     warmup_epochs=WARMUP_EPOCHS,
#     print_boundaries_every=1,
# )

# %%
# def test_external_audio(
#     audio_path,
#     correct_target,
#     model,
#     accelerator,
#     phoneme_to_id,
#     id_to_phoneme,
#     sample_rate=16000,
#     print_result=True,
# ):
#     import ast
#     import torch

#     if isinstance(correct_target, str):
#         target_text = correct_target.strip()

#         if target_text.startswith("["):
#             correct_target = ast.literal_eval(
#                 target_text
#             )
#         else:
#             correct_target = target_text.split()

#     if not isinstance(
#         correct_target,
#         (list, tuple),
#     ):
#         raise TypeError(
#             "correct_target must be a list of phonemes "
#             "or a string representation of a list"
#         )

#     correct_target = list(correct_target)

#     if len(correct_target) == 0:
#         raise ValueError(
#             "correct_target cannot be empty"
#         )

#     unknown_phonemes = [
#         phoneme
#         for phoneme in correct_target
#         if phoneme not in phoneme_to_id
#     ]

#     if unknown_phonemes:
#         raise ValueError(
#             f"Unknown phonemes: {unknown_phonemes}"
#         )

#     waveform = load_waveform(
#         audio_path=audio_path,
#         sr=sample_rate,
#         training=False,
#     )

#     original_num_samples = waveform.shape[0]
#     audio_duration_seconds = (
#         original_num_samples / sample_rate
#     )

#     waveform = waveform.unsqueeze(0)

#     input_lengths = torch.tensor(
#         [original_num_samples],
#         dtype=torch.long,
#     )

#     target_ids = torch.tensor(
#         [
#             phoneme_to_id[phoneme]
#             for phoneme in correct_target
#         ],
#         dtype=torch.long,
#     )

#     target_lengths = torch.tensor(
#         [len(correct_target)],
#         dtype=torch.long,
#     )

#     waveform = waveform.to(
#         accelerator.device
#     )

#     input_lengths = input_lengths.to(
#         accelerator.device
#     )

#     target_lengths = target_lengths.to(
#         accelerator.device
#     )

#     target_ids = target_ids.to(
#         accelerator.device
#     )

#     model.eval()

#     with torch.no_grad():
#         (
#             logits_batch,
#             segment_lengths,
#             hard_boundaries_batch,
#             hidden_lengths,
#         ) = model(
#             waveform,
#             input_lengths,
#             target_lengths,
#         )

#     logits = logits_batch[0]

#     predicted_ids = torch.argmax(
#         logits,
#         dim=-1,
#     )

#     probabilities = torch.softmax(
#         logits,
#         dim=-1,
#     )

#     predicted_confidences = probabilities.max(
#         dim=-1
#     ).values

#     hard_boundaries = (
#         hard_boundaries_batch[0]
#     )

#     soft_segment_masses = (
#         segment_lengths[0]
#     )

#     hidden_length = int(
#         hidden_lengths[0].item()
#     )

#     unwrapped_model = (
#         accelerator.unwrap_model(model)
#     )

#     boundary_data = (
#         unwrapped_model.boundaries_to_seconds(
#             boundaries=hard_boundaries,
#             input_length=original_num_samples,
#             hidden_length=hidden_length,
#             sample_rate=sample_rate,
#         )
#     )

#     total_stride = 1

#     for stride in unwrapped_model.conv_stride:
#         total_stride *= stride

#     nominal_frame_duration_seconds = (
#         total_stride / sample_rate
#     )

#     results = []
#     correct_count = 0

#     for index in range(
#         len(correct_target)
#     ):
#         predicted_id = int(
#             predicted_ids[index].item()
#         )

#         target_id = int(
#             target_ids[index].item()
#         )

#         predicted_phoneme = (
#             id_to_phoneme[predicted_id]
#         )

#         target_phoneme = (
#             id_to_phoneme[target_id]
#         )

#         is_correct = (
#             predicted_id == target_id
#         )

#         if is_correct:
#             correct_count += 1

#         boundary = boundary_data[index]

#         start_frame = int(
#             boundary["start_frame"]
#         )

#         end_frame = int(
#             boundary["end_frame"]
#         )

#         frame_count = (
#             end_frame - start_frame
#         )

#         segment_duration = float(
#             boundary["duration_seconds"]
#         )

#         if frame_count > 0:
#             average_frame_duration = (
#                 segment_duration / frame_count
#             )
#         else:
#             average_frame_duration = 0.0

#         result = {
#             "index": index,
#             "target_phoneme": target_phoneme,
#             "predicted_phoneme": predicted_phoneme,
#             "correct": is_correct,
#             "confidence": float(
#                 predicted_confidences[
#                     index
#                 ].item()
#             ),
#             "start_frame": start_frame,
#             "end_frame": end_frame,
#             "frame_count": frame_count,
#             "start_sample": int(
#                 boundary["start_sample"]
#             ),
#             "end_sample": int(
#                 boundary["end_sample"]
#             ),
#             "start_seconds": float(
#                 boundary["start_seconds"]
#             ),
#             "end_seconds": float(
#                 boundary["end_seconds"]
#             ),
#             "duration_seconds": segment_duration,
#             "average_frame_duration_seconds": (
#                 average_frame_duration
#             ),
#             "nominal_frame_duration_seconds": (
#                 nominal_frame_duration_seconds
#             ),
#             "soft_segment_mass": float(
#                 soft_segment_masses[
#                     index
#                 ].item()
#             ),
#         }

#         results.append(result)

#     total_segments = len(correct_target)

#     segment_accuracy = (
#         correct_count / total_segments
#     )

#     predicted_phonemes = [
#         result["predicted_phoneme"]
#         for result in results
#     ]

#     output = {
#         "audio_path": audio_path,
#         "sample_rate": sample_rate,
#         "audio_samples": original_num_samples,
#         "audio_duration_seconds": (
#             audio_duration_seconds
#         ),
#         "encoder_frame_count": hidden_length,
#         "nominal_frame_duration_seconds": (
#             nominal_frame_duration_seconds
#         ),
#         "correct_target": correct_target,
#         "predicted_phonemes": (
#             predicted_phonemes
#         ),
#         "correct_segments": correct_count,
#         "total_segments": total_segments,
#         "segment_accuracy": segment_accuracy,
#         "segments": results,
#     }

#     if print_result:
#         print()
#         print(
#             f"Audio: {audio_path}"
#         )
#         print(
#             f"Audio duration: "
#             f"{audio_duration_seconds:.3f} seconds"
#         )
#         print(
#             f"Encoder frames: {hidden_length}"
#         )
#         print(
#             f"Nominal frame duration: "
#             f"{nominal_frame_duration_seconds * 1000:.2f} ms"
#         )
#         print(
#             f"Correct segments: "
#             f"{correct_count}/{total_segments}"
#         )
#         print(
#             f"Segment accuracy: "
#             f"{segment_accuracy:.2%}"
#         )
#         print()
#         print(
#             "Predicted phoneme boundaries"
#         )
#         print("=" * 130)

#         for result in results:
#             status = (
#                 "correct"
#                 if result["correct"]
#                 else "wrong"
#             )

#             print(
#                 f"{result['index']:03d} | "
#                 f"target={result['target_phoneme']:<9} | "
#                 f"pred={result['predicted_phoneme']:<9} | "
#                 f"{status:<7} | "
#                 f"confidence={result['confidence']:.3f} | "
#                 f"frames=["
#                 f"{result['start_frame']:4d}, "
#                 f"{result['end_frame']:4d}) | "
#                 f"count={result['frame_count']:3d} | "
#                 f"time=["
#                 f"{result['start_seconds']:7.3f}, "
#                 f"{result['end_seconds']:7.3f}) s | "
#                 f"duration="
#                 f"{result['duration_seconds']:6.3f} s | "
#                 f"frame_duration="
#                 f"{result['average_frame_duration_seconds'] * 1000:6.2f} ms | "
#                 f"soft_mass="
#                 f"{result['soft_segment_mass']:6.2f}"
#             )

#         print("=" * 130)

#         print()
#         print(
#             "Correct target:"
#         )
#         print(correct_target)

#         print()
#         print(
#             "Predicted phonemes:"
#         )
#         print(predicted_phonemes)

#     return output

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
# item = val_df.iloc[0]
# audio_path =  f"../../datasets/Quran_ds/Quran_ds/audio/audio/{item['path_of_audio']}"
# target_phonemes = ast.literal_eval(item['phonemes'])


# # audio_path = "../../datasets/test_audios/1.wav"

# print(f"Audio path: {audio_path}")
# print(f"Target phonemes: {target_phonemes}")

# %%
# result = test_external_audio(
#     audio_path=audio_path,
#     correct_target=target_phonemes,
#     model=model,
#     accelerator=accelerator, 
#     phoneme_to_id=phoneme_to_id,
#     id_to_phoneme=id_to_phoneme,
#     sample_rate=SR,
# )

# %%
# from collections import Counter
# import ast

# counter = Counter()

# for phonemes in train_df["phonemes"]:
#     seq = ast.literal_eval(phonemes)
#     counter.update(seq)

# print(len(counter))

# print(counter["<sil>"])
# print(counter.most_common())

# %%
# import os
# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from tqdm.auto import tqdm
# from sklearn.metrics import confusion_matrix


# @torch.no_grad()
# def collect_validation_predictions(
#     model,
#     val_loader,
#     accelerator,
# ):
#     model.eval()

#     all_target_ids = []
#     all_prediction_ids = []

#     progress_bar = tqdm(
#         val_loader,
#         desc="Collecting validation predictions",
#         disable=not accelerator.is_local_main_process,
#     )

#     for (
#         waveforms,
#         targets,
#         input_lengths,
#         target_lengths,
#     ) in progress_bar:

#         (
#             logits_batch,
#             segment_lengths,
#             hard_boundaries_batch,
#             hidden_lengths,
#         ) = model(
#             waveforms,
#             input_lengths,
#             target_lengths,
#         )

#         for batch_index, logits in enumerate(
#             logits_batch
#         ):
#             target_length = int(
#                 target_lengths[
#                     batch_index
#                 ].item()
#             )

#             predictions = torch.argmax(
#                 logits,
#                 dim=-1,
#             )

#             targets_for_sample = targets[
#                 batch_index,
#                 :target_length,
#             ]

#             predictions = (
#                 predictions.detach()
#                 .cpu()
#                 .long()
#             )

#             targets_for_sample = (
#                 targets_for_sample.detach()
#                 .cpu()
#                 .long()
#             )

#             if predictions.numel() != target_length:
#                 raise RuntimeError(
#                     f"Prediction length "
#                     f"{predictions.numel()} does not match "
#                     f"target length {target_length}"
#                 )

#             valid_mask = (
#                 targets_for_sample != -100
#             )

#             all_prediction_ids.append(
#                 predictions[valid_mask]
#             )

#             all_target_ids.append(
#                 targets_for_sample[valid_mask]
#             )

#     if len(all_target_ids) == 0:
#         raise RuntimeError(
#             "No validation predictions were collected"
#         )

#     all_target_ids = torch.cat(
#         all_target_ids
#     ).numpy()

#     all_prediction_ids = torch.cat(
#         all_prediction_ids
#     ).numpy()

#     return (
#         all_target_ids,
#         all_prediction_ids,
#     )


# def safe_row_normalize(matrix):
#     matrix = matrix.astype(
#         np.float64
#     )

#     row_sums = matrix.sum(
#         axis=1,
#         keepdims=True,
#     )

#     normalized = np.divide(
#         matrix,
#         row_sums,
#         out=np.zeros_like(
#             matrix,
#             dtype=np.float64,
#         ),
#         where=row_sums != 0,
#     )

#     return normalized


# def build_per_phoneme_report(
#     full_confusion_matrix,
#     label_ids,
#     id_to_phoneme,
# ):
#     rows = []

#     for matrix_index, phoneme_id in enumerate(
#         label_ids
#     ):
#         support = int(
#             full_confusion_matrix[
#                 matrix_index
#             ].sum()
#         )

#         correct = int(
#             full_confusion_matrix[
#                 matrix_index,
#                 matrix_index,
#             ]
#         )

#         accuracy = (
#             correct / support
#             if support > 0
#             else 0.0
#         )

#         predicted_count = int(
#             full_confusion_matrix[
#                 :,
#                 matrix_index,
#             ].sum()
#         )

#         rows.append(
#             {
#                 "phoneme_id": int(
#                     phoneme_id
#                 ),
#                 "phoneme": id_to_phoneme[
#                     int(phoneme_id)
#                 ],
#                 "support": support,
#                 "correct": correct,
#                 "incorrect": (
#                     support - correct
#                 ),
#                 "predicted_count": (
#                     predicted_count
#                 ),
#                 "accuracy": accuracy,
#                 "accuracy_percent": (
#                     accuracy * 100.0
#                 ),
#             }
#         )

#     report = pd.DataFrame(
#         rows
#     )

#     report = report.sort_values(
#         by=[
#             "accuracy",
#             "support",
#         ],
#         ascending=[
#             True,
#             False,
#         ],
#     ).reset_index(
#         drop=True
#     )

#     return report


# def build_top_confusion_report(
#     full_confusion_matrix,
#     label_ids,
#     id_to_phoneme,
# ):
#     confusion_records = []

#     for true_index, true_id in enumerate(
#         label_ids
#     ):
#         true_support = int(
#             full_confusion_matrix[
#                 true_index
#             ].sum()
#         )

#         for predicted_index, predicted_id in enumerate(
#             label_ids
#         ):
#             if true_index == predicted_index:
#                 continue

#             error_count = int(
#                 full_confusion_matrix[
#                     true_index,
#                     predicted_index,
#                 ]
#             )

#             if error_count == 0:
#                 continue

#             error_rate = (
#                 error_count / true_support
#                 if true_support > 0
#                 else 0.0
#             )

#             confusion_records.append(
#                 {
#                     "actual_phoneme": (
#                         id_to_phoneme[
#                             int(true_id)
#                         ]
#                     ),
#                     "predicted_phoneme": (
#                         id_to_phoneme[
#                             int(predicted_id)
#                         ]
#                     ),
#                     "error_count": (
#                         error_count
#                     ),
#                     "actual_support": (
#                         true_support
#                     ),
#                     "error_rate": (
#                         error_rate
#                     ),
#                     "error_rate_percent": (
#                         error_rate * 100.0
#                     ),
#                 }
#             )

#     report = pd.DataFrame(
#         confusion_records
#     )

#     if len(report) == 0:
#         return report

#     report = report.sort_values(
#         by=[
#             "error_count",
#             "error_rate",
#         ],
#         ascending=[
#             False,
#             False,
#         ],
#     ).reset_index(
#         drop=True
#     )

#     return report


# def plot_confusion_matrix(
#     matrix,
#     phoneme_names,
#     title,
#     output_path,
#     normalized=True,
# ):
#     class_count = len(
#         phoneme_names
#     )

#     figure_size = max(
#         12,
#         min(
#             32,
#             class_count * 0.55,
#         ),
#     )

#     figure, axis = plt.subplots(
#         figsize=(
#             figure_size,
#             figure_size,
#         )
#     )

#     image = axis.imshow(
#         matrix,
#         interpolation="nearest",
#         aspect="auto",
#     )

#     figure.colorbar(
#         image,
#         ax=axis,
#         fraction=0.046,
#         pad=0.04,
#     )

#     axis.set_title(
#         title
#     )

#     axis.set_xlabel(
#         "Predicted phoneme"
#     )

#     axis.set_ylabel(
#         "Actual phoneme"
#     )

#     tick_positions = np.arange(
#         class_count
#     )

#     axis.set_xticks(
#         tick_positions
#     )

#     axis.set_yticks(
#         tick_positions
#     )

#     axis.set_xticklabels(
#         phoneme_names,
#         rotation=90,
#         fontsize=8,
#     )

#     axis.set_yticklabels(
#         phoneme_names,
#         fontsize=8,
#     )

#     if class_count <= 25:
#         maximum_value = (
#             matrix.max()
#             if matrix.size > 0
#             else 0
#         )

#         threshold = (
#             maximum_value / 2.0
#         )

#         for row_index in range(
#             class_count
#         ):
#             for column_index in range(
#                 class_count
#             ):
#                 value = matrix[
#                     row_index,
#                     column_index,
#                 ]

#                 if normalized:
#                     display_value = (
#                         f"{value:.2f}"
#                     )
#                 else:
#                     display_value = (
#                         str(int(value))
#                     )

#                 axis.text(
#                     column_index,
#                     row_index,
#                     display_value,
#                     horizontalalignment="center",
#                     verticalalignment="center",
#                     fontsize=7,
#                 )

#     figure.tight_layout()

#     figure.savefig(
#         output_path,
#         dpi=250,
#         bbox_inches="tight",
#     )

#     plt.show()
#     plt.close(
#         figure
#     )


# def create_phoneme_confusion_matrix(
#     model,
#     val_loader,
#     accelerator,
#     id_to_phoneme,
#     top_k=40,
#     output_directory="confusion_matrix_results",
# ):
#     os.makedirs(
#         output_directory,
#         exist_ok=True,
#     )

#     (
#         target_ids,
#         prediction_ids,
#     ) = collect_validation_predictions(
#         model=model,
#         val_loader=val_loader,
#         accelerator=accelerator,
#     )

#     total_phoneme_count = len(
#         id_to_phoneme
#     )

#     true_counts = np.bincount(
#         target_ids,
#         minlength=total_phoneme_count,
#     )

#     predicted_counts = np.bincount(
#         prediction_ids,
#         minlength=total_phoneme_count,
#     )

#     active_label_ids = np.where(
#         (
#             true_counts
#             + predicted_counts
#         ) > 0
#     )[0]

#     full_matrix = confusion_matrix(
#         target_ids,
#         prediction_ids,
#         labels=active_label_ids,
#     )

#     full_normalized_matrix = (
#         safe_row_normalize(
#             full_matrix
#         )
#     )

#     active_phoneme_names = [
#         id_to_phoneme[
#             int(phoneme_id)
#         ]
#         for phoneme_id in active_label_ids
#     ]

#     full_count_dataframe = pd.DataFrame(
#         full_matrix,
#         index=active_phoneme_names,
#         columns=active_phoneme_names,
#     )

#     full_normalized_dataframe = pd.DataFrame(
#         full_normalized_matrix,
#         index=active_phoneme_names,
#         columns=active_phoneme_names,
#     )

#     full_count_path = os.path.join(
#         output_directory,
#         "full_confusion_matrix_counts.csv",
#     )

#     full_normalized_path = os.path.join(
#         output_directory,
#         "full_confusion_matrix_normalized.csv",
#     )

#     full_count_dataframe.to_csv(
#         full_count_path
#     )

#     full_normalized_dataframe.to_csv(
#         full_normalized_path
#     )

#     per_phoneme_report = (
#         build_per_phoneme_report(
#             full_confusion_matrix=full_matrix,
#             label_ids=active_label_ids,
#             id_to_phoneme=id_to_phoneme,
#         )
#     )

#     per_phoneme_path = os.path.join(
#         output_directory,
#         "per_phoneme_accuracy.csv",
#     )

#     per_phoneme_report.to_csv(
#         per_phoneme_path,
#         index=False,
#     )

#     top_confusions = (
#         build_top_confusion_report(
#             full_confusion_matrix=full_matrix,
#             label_ids=active_label_ids,
#             id_to_phoneme=id_to_phoneme,
#         )
#     )

#     top_confusions_path = os.path.join(
#         output_directory,
#         "top_phoneme_confusions.csv",
#     )

#     top_confusions.to_csv(
#         top_confusions_path,
#         index=False,
#     )

#     overall_accuracy = float(
#         (
#             target_ids
#             == prediction_ids
#         ).mean()
#     )

#     if top_k is None:
#         selected_label_ids = (
#             active_label_ids
#         )
#     else:
#         top_k = min(
#             int(top_k),
#             len(active_label_ids),
#         )

#         active_true_counts = true_counts[
#             active_label_ids
#         ]

#         frequency_order = np.argsort(
#             active_true_counts
#         )[::-1]

#         selected_label_ids = (
#             active_label_ids[
#                 frequency_order[:top_k]
#             ]
#         )

#     other_id = -1

#     selected_set = set(
#         selected_label_ids.tolist()
#     )

#     mapped_targets = np.array(
#         [
#             target_id
#             if target_id in selected_set
#             else other_id
#             for target_id in target_ids
#         ],
#         dtype=np.int64,
#     )

#     mapped_predictions = np.array(
#         [
#             prediction_id
#             if prediction_id in selected_set
#             else other_id
#             for prediction_id in prediction_ids
#         ],
#         dtype=np.int64,
#     )

#     display_label_ids = list(
#         selected_label_ids
#     )

#     if (
#         np.any(
#             mapped_targets == other_id
#         )
#         or np.any(
#             mapped_predictions == other_id
#         )
#     ):
#         display_label_ids.append(
#             other_id
#         )

#     display_names = [
#         (
#             "<other>"
#             if phoneme_id == other_id
#             else id_to_phoneme[
#                 int(phoneme_id)
#             ]
#         )
#         for phoneme_id in display_label_ids
#     ]

#     display_matrix = confusion_matrix(
#         mapped_targets,
#         mapped_predictions,
#         labels=display_label_ids,
#     )

#     display_normalized_matrix = (
#         safe_row_normalize(
#             display_matrix
#         )
#     )

#     count_plot_path = os.path.join(
#         output_directory,
#         "confusion_matrix_counts.png",
#     )

#     normalized_plot_path = os.path.join(
#         output_directory,
#         "confusion_matrix_normalized.png",
#     )

#     plot_confusion_matrix(
#         matrix=display_matrix,
#         phoneme_names=display_names,
#         title=(
#             "Phoneme confusion matrix — counts"
#         ),
#         output_path=count_plot_path,
#         normalized=False,
#     )

#     plot_confusion_matrix(
#         matrix=display_normalized_matrix,
#         phoneme_names=display_names,
#         title=(
#             "Phoneme confusion matrix — normalized"
#         ),
#         output_path=normalized_plot_path,
#         normalized=True,
#     )

#     print()
#     print(
#         f"Total evaluated phonemes: "
#         f"{len(target_ids)}"
#     )

#     print(
#         f"Overall segment accuracy: "
#         f"{overall_accuracy:.2%}"
#     )

#     print(
#         f"Active phoneme classes: "
#         f"{len(active_label_ids)}"
#     )

#     print()
#     print(
#         "Saved files:"
#     )

#     print(
#         full_count_path
#     )

#     print(
#         full_normalized_path
#     )

#     print(
#         per_phoneme_path
#     )

#     print(
#         top_confusions_path
#     )

#     print(
#         count_plot_path
#     )

#     print(
#         normalized_plot_path
#     )

#     return {
#         "target_ids": target_ids,
#         "prediction_ids": (
#             prediction_ids
#         ),
#         "overall_accuracy": (
#             overall_accuracy
#         ),
#         "full_confusion_matrix": (
#             full_matrix
#         ),
#         "full_normalized_matrix": (
#             full_normalized_matrix
#         ),
#         "per_phoneme_report": (
#             per_phoneme_report
#         ),
#         "top_confusions": (
#             top_confusions
#         ),
#         "full_count_dataframe": (
#             full_count_dataframe
#         ),
#         "full_normalized_dataframe": (
#             full_normalized_dataframe
#         ),
#     }

# %%
# confusion_results = (
#     create_phoneme_confusion_matrix(
#         model=model,
#         val_loader=val_loader,
#         accelerator=accelerator,
#         id_to_phoneme=id_to_phoneme,
#         top_k=40,
#         output_directory=(
#             "confusion_matrix_results"
#         ),
#     )
# )

# %%
# display(
#     confusion_results[
#         "top_confusions"
#     ].head(30)
# )

# per_phoneme = confusion_results[
#     "per_phoneme_report"
# ]

# weak_phonemes = per_phoneme[
#     per_phoneme["support"] >= 20
# ].sort_values(
#     by="accuracy"
# )

# display(
#     weak_phonemes.head(30)
# )


