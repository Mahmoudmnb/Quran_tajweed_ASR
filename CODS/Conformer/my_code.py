# %%
import ast
import contextlib
import warnings
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import Wav2Vec2Model

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# %%
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# %%
# # ? this is for local training

MODEL_PATH = "../../models/best_checkpoint_v5.pth"
WORKING_MODEL_PATH = "../../models/checkpoint.pth"
WORKING_BEST_MODEL_PATH = "../../models/best_checkpoint.pth"

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 1
sr = 16000
num_epochs = 8

device

# %%
BLANK_TOKEN = "<blank>"
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
SPECIAL_PHONEMES_FOR_TAJWEED += ["nn", "mm", "yy", "ww", "rM"]


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


PHONEMES_CTC: List[str] = [BLANK_TOKEN] + PHONEMES


phoneme_to_id: Dict[str, int] = {p: i for i, p in enumerate(PHONEMES_CTC)}
blank_id: int = phoneme_to_id[BLANK_TOKEN]

# %% [markdown]
# ### Total duration (TRAIN): 106.5 hours
#
# ### Total duration  (TEST): 09.09 hours

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
                length = int(info.num_frames * sr / info.sample_rate)
            except Exception:
                length = sr * 10  # fallback: assume 10 seconds
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
        target_ids = [
            self.phoneme_to_id[p] for p in phoneme_seq if p in self.phoneme_to_id
        ]

        return (
            waveform,
            torch.tensor(target_ids, dtype=torch.long),
            waveform.shape[0],
            len(target_ids),
        )


# %%
def ctc_collate(batch):
    waveforms, targets, input_lengths, target_lengths = zip(*batch)
    padded_waveforms = pad_sequence(waveforms, batch_first=True)
    return (
        padded_waveforms,
        torch.cat(targets),
        torch.tensor(input_lengths, dtype=torch.long),
        torch.tensor(target_lengths, dtype=torch.long),
    )


# %%
# Load wav2vec2 — frozen feature extractor, runs on GPU
wav2vec2_model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", ignore_mismatched_sizes=True
)

# Freeze everything first
for param in wav2vec2_model.parameters():
    param.requires_grad = False

# Unfreeze last 4 transformer layers
for param in wav2vec2_model.encoder.layers[-6:].parameters():
    param.requires_grad = True

WAV2VEC2_STRIDE = 320  # 16kHz / 320 = 50 frames/sec


def get_feature_lengths(input_lengths):
    """Convert raw sample counts → wav2vec2 output frame counts."""
    return torch.clamp((input_lengths - 400) // WAV2VEC2_STRIDE + 1, min=1)


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
# ── Conformer building blocks ──────────────────────────────────────────────


class FeedForwardModule(nn.Module):
    """The two feed-forward half-steps that sandwich each Conformer block."""

    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion),
            nn.SiLU(),  # Swish activation
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + 0.5 * self.net(x)  # 0.5 scaling from original paper


class ConvolutionModule(nn.Module):
    """Local feature extraction — captures phoneme-level patterns."""

    def __init__(self, dim, kernel_size=31, dropout=0.1):
        super().__init__()
        # kernel_size must be odd for same-length output
        assert kernel_size % 2 == 1
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            # Pointwise expand
            nn.Linear(dim, dim * 2),
            nn.GLU(dim=-1),  # gates half the channels → back to dim
            # Depthwise conv — operates per-channel, captures local timing
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim),
            nn.GroupNorm(1, dim),
            nn.SiLU(),
            # Pointwise project back
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, T, dim)
        residual = x
        x = self.net[0](x)  # LayerNorm
        x = self.net[1](x)  # Linear expand
        x = self.net[2](x)  # GLU
        x = x.transpose(1, 2)  # → (B, dim, T) for Conv1d
        x = self.net[3](x)  # Depthwise Conv
        x = self.net[4](x)  # BatchNorm
        x = self.net[5](x)  # SiLU
        x = x.transpose(1, 2)  # → (B, T, dim)
        x = self.net[6](x)  # Linear project
        x = self.net[7](x)  # Dropout
        return residual + x


class ConformerBlock(nn.Module):
    """
    Full Conformer block:
    FF → Self-Attention → Convolution → FF → LayerNorm
    """

    def __init__(self, dim, num_heads=4, kernel_size=31, ff_expansion=4, dropout=0.1):
        super().__init__()

        self.ff1 = FeedForwardModule(dim, ff_expansion, dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_drop = nn.Dropout(dropout)
        self.conv = ConvolutionModule(dim, kernel_size, dropout)
        self.ff2 = FeedForwardModule(dim, ff_expansion, dropout)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask=None):
        # FF half-step
        x = self.ff1(x)

        # Self-attention
        residual = x
        x_norm = self.attn_norm(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        x = residual + self.attn_drop(x_attn)

        # Convolution
        x = self.conv(x)

        # FF half-step
        x = self.ff2(x)

        return self.norm_out(x)


# ── Full model ─────────────────────────────────────────────────────────────


class Wav2Vec2Conformer_CTC(nn.Module):
    """
    wav2vec2-large-xlsr-53 (frozen) → Linear projection →
    4x Conformer blocks → CTC classifier
    """

    def __init__(
        self,
        num_classes,
        conformer_dim=256,  # internal dim of Conformer
        num_heads=4,
        num_blocks=4,  # 4 blocks is enough — keeps memory manageable
        kernel_size=31,  # 31 frames = ~620ms local context
        dropout=0.1,
    ):
        super().__init__()

        # Project wav2vec2 output (1024) down to conformer_dim
        self.input_projection = nn.Sequential(
            nn.Linear(1024, conformer_dim),
            nn.Dropout(dropout),
        )

        self.conformer_blocks = nn.ModuleList(
            [
                ConformerBlock(
                    dim=conformer_dim,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        self.classifier = nn.Linear(conformer_dim, num_classes)

    def forward(self, features, feat_lengths):
        # features: (B, T', 1024) from wav2vec2
        # feat_lengths: (B,) actual frame counts before padding

        # Project to conformer dim
        x = self.input_projection(features)  # (B, T', conformer_dim)

        # Build padding mask for attention
        # True = position is padding (should be ignored)
        B, T, _ = x.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) >= feat_lengths.unsqueeze(
            1
        )  # (B, T)

        # Pass through Conformer blocks
        for block in self.conformer_blocks:
            x = block(x, key_padding_mask=mask)

        # CTC classifier
        logits = self.classifier(x)  # (B, T', num_classes)
        logits = logits.permute(1, 0, 2)  # (T', B, num_classes) for CTC

        return logits


# %%
class ASRModel(torch.nn.Module):

    def __init__(self, wav2vec2, conformer, spec_augment):
        super().__init__()
        self.wav2vec2 = wav2vec2
        self.conformer = conformer
        self.spec_augment = spec_augment

    def forward(self, waveforms, input_lengths, feat_lengths):
        self.wav2vec2.feature_extractor.eval()

        B, T = waveforms.shape
        attention_mask = torch.zeros(B, T, dtype=torch.long, device=waveforms.device)
        for i, length in enumerate(input_lengths):
            attention_mask[i, :length] = 1

        outputs = self.wav2vec2(waveforms, attention_mask=attention_mask)
        features = outputs.last_hidden_state  # (B, T', 1024)

        # Zero out contaminated padding frames from CNN
        B, T_feat, D = features.shape
        feat_mask = torch.arange(T_feat, device=features.device).unsqueeze(
            0
        ) < feat_lengths.unsqueeze(
            1
        )  # (B, T')
        features = features * feat_mask.unsqueeze(-1).float()  # zero padding frames

        if self.training:
            features = self.spec_augment(features)

        logits = self.conformer(features, feat_lengths)
        return logits


# %%
train_df = pd.read_csv(TRAIN_DS_PATH)
val_df = pd.read_csv(TEST_DS_PATH)

# %%
train_df = train_df.head(1)

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


# MAX_TOKENS = batch_size * 16000 * 20

# train_sampler = DynamicBatchSampler(
#     train_dataset,
#     max_samples_per_batch=MAX_TOKENS,
#     shuffle=True,
# )

# val_sampler = DynamicBatchSampler(
#     val_dataset,
#     max_samples_per_batch=MAX_TOKENS,
#     shuffle=False,
# )

train_loader = DataLoader(
    train_dataset,
    # batch_sampler=train_sampler,
    batch_size=batch_size,
    collate_fn=ctc_collate,
    num_workers=2,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    # batch_sampler=val_sampler,
    batch_size=batch_size,
    collate_fn=ctc_collate,
    num_workers=2,
    pin_memory=True,
)

# %%
next(enumerate(train_loader))


# %%
def save_checkpoint(
    model,
    ctc_loss,
    optimizer,
    epoch,
    # loss,
    best_val_loss,
    epochs_no_improve,
    warmup_scheduler,
    plateau_scheduler,
    path,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "ctc_loss_state": ctc_loss.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "warmup_scheduler_state": warmup_scheduler.state_dict(),
            "plateau_scheduler_state": plateau_scheduler.state_dict(),
            # "loss": loss,
            "best_val_loss": best_val_loss,
            "epochs_no_improve": epochs_no_improve,
        },
        path,
    )


def load_checkpoint(path):
    checkpoint = torch.load(path)
    return checkpoint


# %%
class BlankPenaltyCTCLoss(nn.Module):
    def __init__(self, blank_id, vocab_size, blank_penalty=2.0):
        super().__init__()
        self.blank_id = blank_id
        self.blank_penalty = blank_penalty
        self.ctc = nn.CTCLoss(blank=blank_id, zero_infinity=True, reduction="mean")

        # Fixed — NOT nn.Parameter, optimizer cannot touch it
        bias = torch.zeros(vocab_size)
        bias[blank_id] = -blank_penalty
        self.register_buffer("bias", bias)

    def forward(self, logits, targets, input_lengths, target_lengths):
        penalized = logits + self.bias
        log_probs = torch.log_softmax(penalized, dim=-1)
        return self.ctc(log_probs.float(), targets, input_lengths, target_lengths)


# %%
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(mixed_precision="fp16", kwargs_handlers=[ddp_kwargs])

print(f"Using device: {device}")
print(f"Num processes: {accelerator.num_processes}")

conformer_model = Wav2Vec2Conformer_CTC(
    num_classes=len(phoneme_to_id),
    conformer_dim=256,
    num_heads=4,
    num_blocks=4,
    kernel_size=31,
    dropout=0.15,
)

model = ASRModel(
    wav2vec2_model,
    conformer_model,
    spec_augment=SpecAugment(
        time_mask_max=6,
        freq_mask_max=6,
        num_time_masks=2,
        num_freq_masks=2,
    ),
)

# ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True, reduction="mean")

# ctc_loss = BlankPenaltyCTCLoss(
#     blank_id=blank_id, vocab_size=len(phoneme_to_id), blank_penalty=6
# )


WARMUP_EPOCHS = 5
TARGET_CONFORMER_LR = 1e-4  # ✅ raised from 1e-5
TARGET_WAV2VEC2_LR = 1e-6  # unchanged

optimizer = torch.optim.AdamW(
    [
        {"params": model.wav2vec2.encoder.parameters(), "lr": TARGET_WAV2VEC2_LR},
        {"params": model.conformer.parameters(), "lr": TARGET_CONFORMER_LR},
    ],
    weight_decay=0.01,
)


# Linear warmup for first WARMUP_EPOCHS, then hand off to ReduceLROnPlateau
def warmup_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS  # 0.2, 0.4, 0.6, 0.8, 1.0
    return 1.0  # after warmup, LR stays at target — plateau scheduler takes over


warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    min_lr=1e-9,
)

# ── Class-balance weight tensor ────────────────────────────────────────────
# Since you commented out the frequency-based weights, use uniform weights.
# This is a safe no-op — you can replace it later with real frequency weights.
weight_tensor = torch.ones(len(phoneme_to_id), device=device)

# Let accelerate handle everything
model, optimizer, train_loader, val_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader
)


best_val_loss = float("inf")
epochs_no_improve = 0
start_epoch = 0

if os.path.exists(MODEL_PATH):
    print("Loading checkpoint...")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    accelerator.unwrap_model(model).load_state_dict(ckpt["model_state"])

    optimizer.load_state_dict(ckpt["optimizer_state"])

    # Safe load — handles both old and new checkpoint formats
    if "warmup_scheduler_state" in ckpt:
        warmup_scheduler.load_state_dict(ckpt["warmup_scheduler_state"])
        print("Warmup scheduler state restored ✅")
    else:
        print("⚠️ No warmup scheduler state found — starting fresh (old checkpoint)")

    if "plateau_scheduler_state" in ckpt:
        plateau_scheduler.load_state_dict(ckpt["plateau_scheduler_state"])
        print("Plateau scheduler state restored ✅")
    elif "scheduler_state" in ckpt:
        # Old checkpoint had a single scheduler — load into plateau scheduler
        plateau_scheduler.load_state_dict(ckpt["scheduler_state"])
        print("Plateau scheduler restored from old checkpoint format ✅")
    else:
        print("⚠️ No plateau scheduler state found — starting fresh")

    best_val_loss = ckpt["best_val_loss"]
    epochs_no_improve = ckpt["epochs_no_improve"]
    start_epoch = ckpt["epoch"] + 1
    print(f"Resumed from epoch {start_epoch}")
    print(f"Best val loss so far: {best_val_loss:.4f}")
    print("done loading")
else:
    print("Starting fresh training")
    best_val_loss = float("inf")
    epochs_no_improve = 0
    start_epoch = 0

# %%
model


# %%
def ctc_decode(frame_preds, blank_id):

    decoded = []
    prev = None

    for p in frame_preds:

        p = int(p)

        if p != blank_id and p != prev:
            decoded.append(p)

        prev = p

    return decoded


# %%
def compute_per(logits, targets, feat_lengths, target_lengths, blank_id):
    """Phoneme Error Rate — lower is better."""
    pred = torch.argmax(logits, dim=2).permute(1, 0).cpu()  # (B, T)
    feat_lengths = feat_lengths.cpu()
    target_lengths = target_lengths.cpu()
    targets = targets.cpu()

    total_errors = 0
    total_phonemes = 0
    offset = 0

    for i in range(pred.shape[0]):
        decoded = ctc_decode(pred[i, : feat_lengths[i]].tolist(), blank_id)
        length = target_lengths[i].item()
        ref = targets[offset : offset + length].tolist()
        offset += length

        # Simple edit distance
        n, m = len(ref), len(decoded)
        dp = list(range(m + 1))
        for r in ref:
            new_dp = [dp[0] + 1]
            for j, h in enumerate(decoded):
                new_dp.append(
                    min(dp[j + 1] + 1, new_dp[-1] + 1, dp[j] + (0 if r == h else 1))
                )
            dp = new_dp
        total_errors += dp[m]
        total_phonemes += n

    return total_errors / max(total_phonemes, 1)


# %%
def compute_blank_rate(logits, feat_lengths, blank_id):
    """Fraction of frames predicted as blank — should drop as penalty takes effect."""
    preds = torch.argmax(logits, dim=2).permute(1, 0)  # (B, T)
    total_frames = 0
    blank_frames = 0
    for i in range(preds.shape[0]):
        frames = preds[i, : feat_lengths[i]]
        blank_frames += (frames == blank_id).sum().item()
        total_frames += feat_lengths[i].item()
    return blank_frames / max(total_frames, 1)


# %%
def prepare_targets(targets, target_lengths, blank=0):
    batch_size = targets.size(0)

    prepared = []
    prepared_lengths = []

    for b in range(batch_size):
        length = target_lengths[b].item()
        target = targets[b, :length]

        prev = target[0].item()
        new_target = [prev]

        for symbol_tensor in target[1:]:
            symbol = symbol_tensor.item()

            if symbol == prev:
                new_target.append(blank)

            new_target.append(symbol)
            prev = symbol

        prepared.append(new_target)
        prepared_lengths.append(len(new_target))

    max_len = max(prepared_lengths)

    prepared_tensor = torch.full(
        (batch_size, max_len),
        blank,
        dtype=targets.dtype,
    )

    for b, seq in enumerate(prepared):
        prepared_tensor[b, : len(seq)] = torch.tensor(seq, dtype=targets.dtype)

    prepared_lengths = torch.tensor(prepared_lengths, dtype=target_lengths.dtype)

    return prepared_tensor.to(targets.device), prepared_lengths.to(targets.device)


def ctc_loss_custom(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
    reduction: str = "none",
    finfo_min_fp32: float = torch.finfo(torch.float32).min,
    finfo_min_fp16: float = torch.finfo(torch.float16).min,
) -> torch.Tensor:

    device = log_probs.device

    targets = targets.to(device)
    input_lengths = input_lengths.to(device)
    target_lengths = target_lengths.to(device)

    if targets.dim() == 1:
        targets = targets.unsqueeze(0)

    input_time_size, batch_size = log_probs.shape[:2]
    B = torch.arange(batch_size, device=input_lengths.device)

    targets, target_lengths = prepare_targets(
        targets,
        target_lengths,
        blank,
    )

    zero_padding, zero = 2, torch.tensor(
        finfo_min_fp16 if log_probs.dtype == torch.float16 else finfo_min_fp32,
        device=log_probs.device,
        dtype=log_probs.dtype,
    )

    log_probs_ = log_probs.gather(-1, targets.expand(input_time_size, -1, -1))

    log_alpha = torch.full(
        (input_time_size, batch_size, zero_padding + targets.shape[-1]),
        zero,
        device=log_probs.device,
        dtype=log_probs.dtype,
    )

    log_alpha[0, :, zero_padding] = log_probs[0, B, targets[:, 0]]

    for t in range(1, input_time_size):
        log_alpha[t, :, 2:] = log_probs_[t] + torch.logsumexp(
            torch.stack(
                [
                    log_alpha[t - 1, :, 2:],  # stay
                    log_alpha[t - 1, :, 1:-1],  # move
                ]
            ),
            dim=0,
        )

    last_state = zero_padding + target_lengths - 1

    loss = -log_alpha[
        input_lengths - 1,
        B,
        last_state,
    ]

    if reduction == "mean":
        loss = loss / target_lengths.to(loss.dtype)
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


# %%
print("training is starting .......")

# %%
id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}


def custom_ctc_decode(frame_preds, blank_id):
    decoded = []
    prev = None
    count = 0

    for p in frame_preds:
        p = int(p)

        if p == blank_id:
            # ✅ Reset prev so identical phonemes across a blank are treated separately
            if prev is not None and prev != blank_id:
                if decoded:
                    decoded[-1]["count"] = count
                count = 0
            prev = blank_id
            continue

        if p != prev:
            # Save previous phoneme count
            if decoded and prev != blank_id:
                decoded[-1]["count"] = count
            decoded.append({"char": p, "count": 0})
            count = 0

        prev = p
        count += 1

    # Save last phoneme
    if decoded:
        decoded[-1]["count"] = count

    return decoded


# %%
def train_model(
    model,
    train_loader,
    val_loader,
    # ctc_loss,
    optimizer,
    warmup_scheduler,
    plateau_scheduler,
    accelerator,
    epochs=30,
    patience=6,
    best_val_loss=float("inf"),
    epochs_no_improve=0,
    warmup_epochs=5,
    start_epoch=0,
):
    for epoch in range(start_epoch, start_epoch + epochs):

        # ── Training loop ──────────────────────────────────────────────
        model.train()
        train_loss = 0
        for waveforms, targets, input_lengths, target_lengths in tqdm(train_loader):
            feat_lengths = get_feature_lengths(input_lengths)
            optimizer.zero_grad(set_to_none=True)

            logits = model(waveforms, input_lengths, feat_lengths)  # (T, B, V)
            # logits = logits + torch.log(weight_tensor).to(
            #     logits.device
            # )  # class balancing

            # ✅ Pass raw logits — BlankPenaltyCTCLoss handles log_softmax internally
            # loss = ctc_loss(logits, targets, feat_lengths, target_lengths)
            log_probs = torch.log_softmax(logits, dim=-1)

            loss = ctc_loss_custom(
                log_probs,
                targets,
                feat_lengths,
                target_lengths,
                reduction="mean",
                blank=blank_id,
            )

            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ── Validation loop ────────────────────────────────────────────
        model.eval()
        val_loss = 0
        per = 0
        total_blank_rate = 0

        # with torch.no_grad():
        #     for waveforms, targets, input_lengths, target_lengths in tqdm(val_loader):
        #         feat_lengths = get_feature_lengths(input_lengths)

        #         logits = model(waveforms, input_lengths, feat_lengths)
        #         # logits = logits + torch.log(weight_tensor).to(logits.device)

        #         # ✅ Same here — raw logits into ctc_loss
        #         # loss = ctc_loss(logits, targets, feat_lengths, target_lengths)

        #         log_probs = torch.log_softmax(logits, dim=-1)

        #         loss = ctc_loss_custom(
        #             log_probs,
        #             targets,
        #             feat_lengths,
        #             target_lengths,
        #             reduction="mean",
        #             blank=blank_id,
        #         )
        #         val_loss += loss.item()

        #         per += compute_per(
        #             logits, targets, feat_lengths, target_lengths, blank_id
        #         )

        #         # ✅ Track blank rate to verify penalty is working
        #         total_blank_rate += compute_blank_rate(logits, feat_lengths, blank_id)

        # avg_val_loss = val_loss / len(val_loader)
        # per = per / len(val_loader)
        # avg_blank_rate = total_blank_rate / len(val_loader)

        # ── Scheduler step ─────────────────────────────────────────────
        warmup_scheduler.step()
        plateau_scheduler.step(avg_train_loss)

        # ── Logging ────────────────────────────────────────────────────
        print(f"\nEpoch {epoch+1}/{start_epoch + epochs}")
        print(f"  Train Loss   : {avg_train_loss:.4f}")
        # print(f"  Val Loss     : {avg_val_loss:.4f}")
        print(f"  Val PER      : {per:.4f}")
        # print(f"  Blank Rate   : {avg_blank_rate:.2%}  ← target: ~40-50%")
        for group in optimizer.param_groups:
            print(f"Learning rate for group: {group['lr']}")

        # ✅ If using learnable penalty, log its current value
        # unwrapped_loss = accelerator.unwrap_model(ctc_loss_custom)
        # if hasattr(unwrapped_loss, "bias"):
        #     current_penalty = -unwrapped_loss.bias[blank_id].item()
        #     print(
        #         f"  Blank Penalty: {current_penalty:.4f}  ← fixed, should stay at 2.0"
        #     )

        print("-" * 40)

        preds = torch.argmax(logits, dim=2).permute(1, 0).cpu()

        decoded_preds = custom_ctc_decode(preds[0], blank_id)
        pred_phonemes_with_frames_count = [
            {"phoneme": id_to_phoneme[int(p["char"])], "count": p["count"]}
            for p in decoded_preds
        ]

        print(pred_phonemes_with_frames_count)

        # ── Checkpointing ──────────────────────────────────────────────
        # unwrapped = accelerator.unwrap_model(model)
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     epochs_no_improve = 0
        #     save_checkpoint(
        #         unwrapped,
        #       # unwrapped_loss,
        #         optimizer,
        #         epoch,
        #         avg_val_loss,
        #         best_val_loss,
        #         epochs_no_improve,
        #         warmup_scheduler,
        #         plateau_scheduler,
        #         WORKING_BEST_MODEL_PATH,
        #     )

        #     print("✅ Best model saved")
        # else:
        #     epochs_no_improve += 1
        #     if epochs_no_improve >= patience:
        #         print(f"⏹ Early stopping at epoch {epoch+1}")
        #         break

        # save_checkpoint(
        #     unwrapped,
        #     # unwrapped_loss,
        #     optimizer,
        #     epoch,
        #     avg_val_loss,
        #     best_val_loss,
        #     epochs_no_improve,
        #     warmup_scheduler,
        #     plateau_scheduler,
        #     WORKING_MODEL_PATH,
        # )


# %%
train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    # ctc_loss=ctc_loss,
    optimizer=optimizer,
    warmup_scheduler=warmup_scheduler,
    plateau_scheduler=plateau_scheduler,
    accelerator=accelerator,
    epochs=300,  # num_epochs,
    patience=6,
    best_val_loss=best_val_loss,
    epochs_no_improve=epochs_no_improve,
    warmup_epochs=WARMUP_EPOCHS,
    start_epoch=start_epoch,
)

# %%
# from collections import Counter
# import ast

# counter = Counter()

# for phonemes in train_df["phonemes"]:
#     seq = ast.literal_eval(phonemes)
#     counter.update(seq)

# print(counter["<sil>"])
# print(counter.most_common())

# %%
id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}


# %%
def custom_ctc_decode(frame_preds, blank_id):
    decoded = []
    prev = None
    count = 0

    for p in frame_preds:
        p = int(p)

        if p == blank_id:
            # ✅ Reset prev so identical phonemes across a blank are treated separately
            if prev is not None and prev != blank_id:
                if decoded:
                    decoded[-1]["count"] = count
                count = 0
            prev = blank_id
            continue

        if p != prev:
            # Save previous phoneme count
            if decoded and prev != blank_id:
                decoded[-1]["count"] = count
            decoded.append({"char": p, "count": 0})
            count = 0

        prev = p
        count += 1

    # Save last phoneme
    if decoded:
        decoded[-1]["count"] = count

    return decoded


# %%
def predict_phonemes(path: str, device):
    waveforms = load_waveform(path, training=False)
    waveforms = waveforms.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        input_lengths = torch.tensor(
            [waveforms.shape[1]], dtype=torch.long, device=device
        )
        feat_lengths = get_feature_lengths(input_lengths)
        logits = model(waveforms, input_lengths, feat_lengths)
        preds = torch.argmax(logits, dim=2).permute(1, 0).cpu()

        decoded_preds = custom_ctc_decode(preds[0], blank_id)
        pred_phonemes = [id_to_phoneme[int(p["char"])] for p in decoded_preds]
        pred_phonemes_with_frames_count = [
            {"phoneme": id_to_phoneme[int(p["char"])], "count": p["count"]}
            for p in decoded_preds
        ]
    return pred_phonemes, pred_phonemes_with_frames_count


# %%
item = train_df.iloc[-1]

audio_path = audio_path = os.path.join(DATASET_PATH, item["path_of_audio"])
phonemes = item["phonemes"]
phonemes

# %%
audio_path

# %%
model.eval()
with torch.no_grad():
    predicted_phonemes, pred_phonemes_with_frames_count = predict_phonemes(
        audio_path,
        device,
    )

    print(predicted_phonemes)
    print()
    print(pred_phonemes_with_frames_count)

# %%
# count = 0

# model.eval()
# with torch.no_grad():

#     for row in train_df.itertuples():
#         audio_path = row.path_of_audio
#         if row.ds_index == 1:
#             audio_path = os.path.join(DATASET_PATH, audio_path)
#         else:
#             audio_path = os.path.join(DATASET_PATH_1, audio_path)

#         predicted_phonemes = predict_phonemes(
#             audio_path,
#             device,
#         )

#         if "<sil>" in predicted_phonemes[1:-1]:
#             count += 1

# print(count)

# %%
# print(count)

# %%
# import torch
# import numpy as np
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# start = 0

# def ctc_greedy_decode(logits, blank_id):
#     """
#     logits: [T, V]
#     returns: list of token ids
#     """
#     probs = torch.softmax(logits, dim=-1)
#     pred_ids = torch.argmax(probs, dim=-1).tolist()

#     result = []
#     prev = None

#     for p in pred_ids:
#         if p != blank_id and p != prev:
#             result.append(p)
#         prev = p

#     return result


# model.eval()

# blank_id = phoneme_to_id["<blank>"]  # adjust if different

# y_true = []
# y_pred = []

# with torch.no_grad():
#     for waveforms, targets, input_lengths, target_lengths in tqdm(val_loader):

#         feat_lengths = get_feature_lengths(input_lengths)
#         logits = model(waveforms, input_lengths, feat_lengths)

#         for b in range(logits.size(1)):

#             logit_seq = logits[:, b, :]  # [T, V]

#             pred_ids = ctc_greedy_decode(logit_seq, blank_id)

#             # -------------------------
#             # FIX TARGET HANDLING
#             # -------------------------
#             L = target_lengths[b].item()

#             true_ids = targets[start:start + L].tolist()
#             start += L

#             y_pred.extend(pred_ids)
#             y_true.extend(true_ids)

# num_classes = len(phoneme_to_id)

# cm = confusion_matrix(
#     y_true,
#     y_pred,
#     labels=list(range(num_classes))
# )


# labels = [id_to_phoneme[i] for i in range(num_classes)]

# plt.figure(figsize=(20, 16))
# sns.heatmap(cm, xticklabels=False, yticklabels=False)
# plt.title("Phoneme Confusion Matrix")
# plt.show()
