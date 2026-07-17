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

        tau = 1.0

        for b in range(B):
            N = int(target_lengths[b].item())
            T_valid = int(hidden_lengths[b].item())

            if N <= 0:
                raise ValueError("Target length must be greater than zero")

            if T_valid <= 0:
                raise ValueError("Hidden length must be greater than zero")
            
            if N > T_valid:
                raise ValueError(
                    f"Target has {N} phonemes but only "
                    f"{T_valid} valid encoder frames"
                )

            embedding = segment_embedding[b, :T_valid]
            valid_progress_weights = progress_weights[b, :T_valid]

            progress = torch.cumsum(
                valid_progress_weights,
                dim=0,
            )

            progress = progress / progress[-1].clamp_min(1e-8)

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

            segment_mass = weights.sum(dim=0)

            pooled = (
                weights.transpose(0, 1)
                @ embedding
            )

            pooled = pooled / segment_mass.unsqueeze(1).clamp_min(1e-8)

            pooled_batch.append(pooled)
            lengths_batch.append(segment_mass)

        return pooled_batch, lengths_batch
    
    def forward(self, waveforms, input_lengths, target_lengths=None):

        features = self.wav2vec2(waveforms)
        features = features.transpose(1, 2)
        feat_lengths = self.get_feature_lengths(input_lengths)

        if self.training:
            features = self.spec_augment(features)

        hidden, hidden_lengths = self.conformer(
            features,
            feat_lengths,
        )

        segment_embedding, progress_weights = self.segmentation(hidden)

        if target_lengths is not None:
            pooled_batch, segment_lengths = self.build_segments(
                                                segment_embedding,
                                                progress_weights,
                                                target_lengths,
                                                hidden_lengths,
                                            )
            logits_batch = [self.segment_classifier(pooled) for pooled in pooled_batch]
            return logits_batch, segment_lengths, hidden_lengths

        return segment_embedding, progress_weights, hidden_lengths

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
next(enumerate(train_loader))

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
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(mixed_precision="fp16", kwargs_handlers=[ddp_kwargs])

print(f"Using device: {DEVICE}")
print(f"Num processes: {accelerator.num_processes}")


model = ASRModel(wav2vec2_model, SpecAugment())

del wav2vec2_model


WARMUP_EPOCHS = 5
WAV2VEC2_LR = 1e-6
CONFORMER_LR = 1e-4
SEGMENTATION_LR = 2e-4
CLASSIFIER_LR = 3e-4

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

model, optimizer, train_loader, val_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader
)


best_val_loss = float("inf")
epochs_no_improve = 0
start_epoch = 0

if os.path.exists(MODEL_PATH):
    print("Loading checkpoint...")
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
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
def segmentation_loss(
    logits_batch,
    targets,
    target_lengths,
):
    all_logits = []
    all_targets = []

    for b, logits in enumerate(logits_batch):
        N = int(target_lengths[b].item())

        target = targets[b, :N].long()
        target = target.to(logits.device)

        all_logits.append(logits)
        all_targets.append(target)

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return F.cross_entropy(
        all_logits,
        all_targets,
    )

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
def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    warmup_scheduler,
    plateau_scheduler,
    accelerator,
    epochs=30,
    patience=6,
    best_val_loss=float("inf"),
    epochs_no_improve=0,
    start_epoch=0,
):
    for epoch in range(start_epoch, start_epoch + epochs):

        # ── Training loop ──────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        train_segment_count = 0

        for waveforms, targets, input_lengths, target_lengths in tqdm(train_loader):

            optimizer.zero_grad(set_to_none=True)

            logits_batch, segment_lengths, hidden_lengths = model(
                waveforms.to(DEVICE), input_lengths, target_lengths
            )

            loss = segmentation_loss(
                logits_batch=logits_batch,
                targets=targets,
                target_lengths=target_lengths,
            )

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    model.parameters(),
                    5.0,
                )

            optimizer.step()

            num_segments = int(target_lengths.sum().item())
            train_loss_sum += loss.item() * num_segments
            train_segment_count += num_segments

            

        avg_train_loss = (
            train_loss_sum / train_segment_count
        )

        # ── Validation loop ────────────────────────────────────────────

        model.eval()

        val_loss_sum = 0.0
        val_segment_count = 0
        val_correct_segments = 0
        
        with torch.no_grad():
            for waveforms, targets, input_lengths, target_lengths in tqdm(val_loader):
            
                logits_batch, segment_lengths, hidden_lengths = model(
                    waveforms,
                    input_lengths,
                    target_lengths,
                )
        
                targets_batch = targets
        
                loss = segmentation_loss(
                    logits_batch=logits_batch,
                    targets=targets_batch,
                    target_lengths=target_lengths,
                )
        
                batch_segment_count = int(
                    target_lengths.sum().item()
                )
        
                val_loss_sum += (
                    loss.item() * batch_segment_count
                )
        
                val_segment_count += batch_segment_count
        
                for b, logits in enumerate(logits_batch):
                    N = int(target_lengths[b].item())
        
                    predictions = torch.argmax(
                        logits,
                        dim=-1,
                    )
        
                    target = targets_batch[b, :N].to(
                        predictions.device
                    )
        
                    val_correct_segments += (
                        predictions == target
                    ).sum().item()
        
        validation_stats = torch.tensor(
            [
                val_loss_sum,
                val_correct_segments,
                val_segment_count,
            ],
            dtype=torch.float64,
            device=accelerator.device,
        )
        
        validation_stats = accelerator.reduce(
            validation_stats,
            reduction="sum",
        )
        
        val_loss_sum = validation_stats[0].item()
        val_correct_segments = int(
            validation_stats[1].item()
        )
        val_segment_count = int(
            validation_stats[2].item()
        )
        
        avg_val_loss = val_loss_sum / max(
            val_segment_count,
            1,
        )
        
        val_accuracy = val_correct_segments / max(
            val_segment_count,
            1,
        )

        # ── Scheduler step ─────────────────────────────────────────────
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(avg_val_loss)

        # ── Logging ────────────────────────────────────────────────────
        print(f"\nEpoch {epoch+1}/{start_epoch + epochs}")
        print(f"  Train Loss   : {avg_train_loss:.4f}")
        print(f"  Val Loss     : {avg_val_loss:.4f}")
        print(f"  Val Accuracy : {val_accuracy:.4f}")
        print(f"  Val Accuracy : {val_accuracy:.2%}")
        for group in optimizer.param_groups:
            print(f"Learning rate for group: {group['lr']}")

        print("-" * 40)

        model_output = [
            (pred, segment_lengths[0][i].item())
            for i, pred in enumerate(
                [
                    id_to_phoneme[p.item()]
                    for p in torch.stack(
                        [
                            torch.argmax(logits, dim=-1).cpu()
                            for logits in logits_batch[0]
                        ]
                    )
                ]
            )
        ]

        print("preds", model_output)

        print(
            "TARGET:",
            [id_to_phoneme[t.item()] for t in targets[0][: target_lengths[0]]],
        )



        # ── Checkpointing ──────────────────────────────────────────────
        unwrapped = accelerator.unwrap_model(model)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            if accelerator.is_main_process:
                save_checkpoint(
                    unwrapped,
                    optimizer,
                    epoch,
                    avg_val_loss,
                    best_val_loss,
                    epochs_no_improve,
                    warmup_scheduler,
                    plateau_scheduler,
                    WORKING_BEST_MODEL_PATH,
                )
        
            print("✅ Best model saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹ Early stopping at epoch {epoch+1}")
                break
        if accelerator.is_main_process:
            
            save_checkpoint(
                unwrapped,
                optimizer,
                epoch,
                avg_val_loss,
                best_val_loss,
                epochs_no_improve,
                warmup_scheduler,
                plateau_scheduler,
                WORKING_MODEL_PATH,
            )

# %%
train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    warmup_scheduler=warmup_scheduler,
    plateau_scheduler=plateau_scheduler,
    accelerator=accelerator,
    epochs= NUM_EPOCHS,
    patience=6,
    best_val_loss=best_val_loss,
    epochs_no_improve=epochs_no_improve,
    start_epoch=start_epoch,
)

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


