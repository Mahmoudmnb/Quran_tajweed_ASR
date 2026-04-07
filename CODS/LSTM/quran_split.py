import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_quran_dataset(
    csv_path: str,
    test_reciter_ratio: float = 0.25,
    test_ayah_ratio: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the Quran recitation dataset into train and test sets with strict guarantees:
      - No reciter appears in BOTH train and test.
      - No ayah (verse) appears in BOTH train and test.

    Strategy:
      1. Split reciters: a subset goes exclusively to test.
      2. Split ayahs (surah+ayah pairs): a subset goes exclusively to test.
      3. Train = rows whose reciter AND ayah are both in the train partition.
      4. Test  = rows whose reciter AND ayah are both in the test partition.
         (Rows that mix train-reciter with test-ayah or vice versa are dropped
          to enforce the hard constraints — these are ~cross-partition rows.)

    Args:
        csv_path:           Path to the dataset CSV file.
        test_reciter_ratio: Fraction of reciters to reserve for test (default 0.25 → 2/8).
        test_ayah_ratio:    Fraction of ayahs to reserve for test (default 0.20).
        random_state:       Random seed for reproducibility.

    Returns:
        (train_df, test_df) — two DataFrames with no shared reciters or ayahs.
    """
    df = pd.read_csv(csv_path)

    # ── 1. Reciter split ────────────────────────────────────────────────────────
    reciters = df["reciter_name"].unique()
    train_reciters, test_reciters = train_test_split(
        reciters,
        test_size=test_reciter_ratio,
        random_state=random_state,
        shuffle=True,
    )
    train_reciters = set(train_reciters)
    test_reciters = set(test_reciters)

    # ── 2. Ayah split (unique surah+ayah pairs) ─────────────────────────────────
    ayah_ids = df[["surah", "ayah"]].drop_duplicates()
    train_ayahs, test_ayahs = train_test_split(
        ayah_ids,
        test_size=test_ayah_ratio,
        random_state=random_state,
        shuffle=True,
    )
    # Build fast lookup sets of (surah, ayah) tuples
    train_ayah_set = set(map(tuple, train_ayahs.values))
    test_ayah_set = set(map(tuple, test_ayahs.values))

    # ── 3. Apply masks ──────────────────────────────────────────────────────────
    ayah_tuples = list(zip(df["surah"], df["ayah"]))

    reciter_in_train = df["reciter_name"].isin(train_reciters)
    reciter_in_test = df["reciter_name"].isin(test_reciters)
    ayah_in_train = pd.Series(
        [t in train_ayah_set for t in ayah_tuples], index=df.index
    )
    ayah_in_test = pd.Series([t in test_ayah_set for t in ayah_tuples], index=df.index)

    train_mask = reciter_in_train & ayah_in_train
    test_mask = reciter_in_test & ayah_in_test

    train_df = df[train_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    # ── 4. Sanity checks ────────────────────────────────────────────────────────
    _verify_split(train_df, test_df)
    _print_stats(df, train_df, test_df, train_reciters, test_reciters)

    return train_df, test_df


def _verify_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Assert hard constraints — raises AssertionError if violated."""

    shared_reciters = set(train_df["reciter_name"]) & set(test_df["reciter_name"])
    assert not shared_reciters, f"Shared reciters found: {shared_reciters}"

    train_ayahs = set(zip(train_df["surah"], train_df["ayah"]))
    test_ayahs = set(zip(test_df["surah"], test_df["ayah"]))
    shared_ayahs = train_ayahs & test_ayahs
    assert (
        not shared_ayahs
    ), f"Shared ayahs found: {len(shared_ayahs)} overlapping verses"

    print("✓ Verification passed: no shared reciters, no shared ayahs.")


def _print_stats(
    original_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_reciters: set,
    test_reciters: set,
) -> None:
    total = len(original_df)
    used = len(train_df) + len(test_df)
    dropped = total - used

    print(f"\n{'─'*55}")
    print(f"  Dataset split summary")
    print(f"{'─'*55}")
    print(f"  Total rows           : {total:>7,}")
    print(
        f"  Train rows           : {len(train_df):>7,}  ({len(train_df)/total*100:.1f}%)"
    )
    print(
        f"  Test  rows           : {len(test_df):>7,}  ({len(test_df)/total*100:.1f}%)"
    )
    print(f"  Dropped (cross-split): {dropped:>7,}  ({dropped/total*100:.1f}%)")
    print(f"{'─'*55}")
    print(f"  Train reciters ({len(train_reciters)}): {sorted(train_reciters)}")
    print(f"  Test  reciters ({len(test_reciters)}): {sorted(test_reciters)}")
    print(f"{'─'*55}")
    print(f"  Unique ayahs in train: {train_df.groupby(['surah','ayah']).ngroups:>6,}")
    print(f"  Unique ayahs in test : {test_df.groupby(['surah','ayah']).ngroups:>6,}")
    print(f"{'─'*55}\n")


# ── Example usage ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quran_ds.csv")

    train_df, test_df = split_quran_dataset(
        csv_path=csv_path,
        test_reciter_ratio=0.25,  # 2 out of 8 reciters go to test
        test_ayah_ratio=0.20,  # 20% of ayahs go to test
        random_state=42,
    )

    # # Save
    train_df.to_csv("quran_train.csv", index=False)
    test_df.to_csv("quran_test.csv", index=False)
    print("Saved: quran_train.csv and quran_test.csv")
