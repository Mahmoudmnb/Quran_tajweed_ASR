"""
split_dataset.py
================
Speaker-aware train/val/test split for Tajweed ASR dataset.

Strategy
--------
- Hold out 1 reciter for val, 1 for test (speaker-independent eval)
- The 2 largest reciters always go to train (preserve volume)
- Val  ← medium reciter (stable loss curve)
- Test ← medium reciter (meaningful metrics)
- Remaining reciters → train

Usage
-----
    python split_dataset.py \
        --input  dataset.csv \
        --output splits/ \
        --reciter_col  reciter \
        --audio_col    audio_path \
        --text_col     arabic_text \
        --phoneme_col  phoneme_label

    # Pin specific reciters manually:
    python split_dataset.py --input dataset.csv --output splits/ \
        --val_reciter  "reciter_4" \
        --test_reciter "reciter_5"
"""

import argparse
import os
import pandas as pd


# ─────────────────────────────────────────────────────────────
# Core split logic
# ─────────────────────────────────────────────────────────────

def compute_reciter_stats(df: pd.DataFrame, reciter_col: str) -> pd.DataFrame:
    """Return a DataFrame with each reciter's sample count and row share."""
    stats = (
        df.groupby(reciter_col)
          .size()
          .reset_index(name='n_samples')
          .sort_values('n_samples', ascending=False)
          .reset_index(drop=True)
    )
    stats['pct'] = (stats['n_samples'] / stats['n_samples'].sum() * 100).round(1)
    return stats


def pick_val_test_reciters(
    stats: pd.DataFrame,
    reciter_col: str,
    val_reciter: str = None,
    test_reciter: str = None,
) -> tuple[str, str]:
    """
    Auto-select val and test reciters if not specified manually.

    Rules:
    - The top-2 largest reciters are always kept for train.
    - From the remaining reciters (sorted descending by size),
      val gets the largest, test gets the second largest.
    - This keeps the two biggest voices in training while giving
      val/test enough samples for stable metrics.
    """
    reciters = stats[reciter_col].tolist()   # already sorted largest → smallest

    if val_reciter and test_reciter:
        assert val_reciter != test_reciter, "val and test reciter must be different"
        assert val_reciter  in reciters, f"val_reciter '{val_reciter}' not found"
        assert test_reciter in reciters, f"test_reciter '{test_reciter}' not found"
        return val_reciter, test_reciter

    # Keep top-2 in train, pick val/test from the rest
    candidates = reciters[2:]          # everything after the two biggest
    if len(candidates) < 2:
        # Edge case: only 2-3 reciters total — relax and pick from all
        candidates = reciters[1:]

    val_reciter  = val_reciter  or candidates[0]
    test_reciter = test_reciter or candidates[1]
    return val_reciter, test_reciter


def split_dataset(
    df: pd.DataFrame,
    reciter_col: str,
    val_reciter: str,
    test_reciter: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_df  = df[df[reciter_col] == test_reciter].copy()
    val_df   = df[df[reciter_col] == val_reciter].copy()
    train_df = df[~df[reciter_col].isin([val_reciter, test_reciter])].copy()
    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────

def print_report(
    stats: pd.DataFrame,
    reciter_col: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_reciter: str,
    test_reciter: str,
) -> None:
    total = len(train_df) + len(val_df) + len(test_df)

    print("=" * 60)
    print("DATASET SPLIT REPORT")
    print("=" * 60)
    print(f"\nTotal samples : {total}")
    print(f"Total reciters: {stats[reciter_col].nunique()}\n")

    print("Reciter distribution (full dataset):")
    print(f"  {'Reciter':<30} {'Samples':>8}  {'%':>6}  {'Role'}")
    print("  " + "-" * 54)
    train_reciters = set(train_df[reciter_col].unique())
    for _, row in stats.iterrows():
        rec = row[reciter_col]
        if rec == val_reciter:
            role = "VAL"
        elif rec == test_reciter:
            role = "TEST"
        else:
            role = "train"
        print(f"  {rec:<30} {row['n_samples']:>8}  {row['pct']:>5.1f}%  {role}")

    print(f"\n{'Split':<8} {'Samples':>8}  {'%':>6}  {'Reciters'}")
    print("-" * 50)
    print(f"{'Train':<8} {len(train_df):>8}  {len(train_df)/total*100:>5.1f}%  "
          f"{sorted(train_reciters)}")
    print(f"{'Val':<8} {len(val_df):>8}  {len(val_df)/total*100:>5.1f}%  [{val_reciter}]")
    print(f"{'Test':<8} {len(test_df):>8}  {len(test_df)/total*100:>5.1f}%  [{test_reciter}]")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
# Surah coverage check
# ─────────────────────────────────────────────────────────────

def check_surah_coverage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
) -> None:
    """
    Warn if val/test contain ayat whose surah is not seen in training.
    Uses the audio_path or text col heuristically — works best if your
    audio paths contain surah numbers (e.g. 002_001.wav).
    This is a best-effort check; adapt the surah extraction if needed.
    """
    def extract_surah(df):
        # Try to extract leading digits from the last path component
        try:
            return df[text_col].str.extract(r'(\d{3})')[0].dropna().unique()
        except Exception:
            return None

    train_surahs = set(extract_surah(train_df) or [])
    val_surahs   = set(extract_surah(val_df)   or [])
    test_surahs  = set(extract_surah(test_df)  or [])

    if not train_surahs:
        return   # couldn't extract — skip silently

    val_unseen  = val_surahs  - train_surahs
    test_unseen = test_surahs - train_surahs
    if val_unseen:
        print(f"\n⚠  Val  contains surahs not in train: {sorted(val_unseen)}")
    if test_unseen:
        print(f"⚠  Test contains surahs not in train: {sorted(test_unseen)}")
    if not val_unseen and not test_unseen:
        print("\n✓  All val/test surahs are covered by training set.")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Speaker-aware dataset splitter")
    p.add_argument("--input",        required=True,  help="Path to input CSV/TSV")
    p.add_argument("--output",       default="splits", help="Output directory")
    p.add_argument("--sep",          default=",",    help="CSV separator (default ',')")
    p.add_argument("--reciter_col",  default="reciter",      help="Reciter column name")
    p.add_argument("--audio_col",    default="audio_path",   help="Audio path column name")
    p.add_argument("--text_col",     default="arabic_text",  help="Arabic text column name")
    p.add_argument("--phoneme_col",  default="phoneme_label",help="Phoneme label column name")
    p.add_argument("--val_reciter",  default=None,   help="Pin a specific reciter for val")
    p.add_argument("--test_reciter", default=None,   help="Pin a specific reciter for test")
    p.add_argument("--train_out",    default="train.csv")
    p.add_argument("--val_out",      default="val.csv")
    p.add_argument("--test_out",     default="test.csv")
    return p.parse_args()


def main():
    args = parse_args()

    # Load
    df = pd.read_csv(args.input, sep=args.sep)
    required = [args.reciter_col, args.audio_col, args.text_col, args.phoneme_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input file: {missing}\n"
                         f"Available columns: {df.columns.tolist()}")

    # Stats
    stats = compute_reciter_stats(df, args.reciter_col)

    # Pick val / test reciters
    val_reciter, test_reciter = pick_val_test_reciters(
        stats, args.reciter_col, args.val_reciter, args.test_reciter
    )

    # Split
    train_df, val_df, test_df = split_dataset(
        df, args.reciter_col, val_reciter, test_reciter
    )

    # Report
    print_report(stats, args.reciter_col, train_df, val_df, test_df,
                 val_reciter, test_reciter)
    check_surah_coverage(train_df, val_df, test_df, args.audio_col)

    # Save
    os.makedirs(args.output, exist_ok=True)
    train_path = os.path.join(args.output, args.train_out)
    val_path   = os.path.join(args.output, args.val_out)
    test_path  = os.path.join(args.output, args.test_out)

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,     index=False)
    test_df.to_csv(test_path,   index=False)

    print(f"\nSaved → {train_path} ({len(train_df)} rows)")
    print(f"Saved → {val_path}   ({len(val_df)} rows)")
    print(f"Saved → {test_path}  ({len(test_df)} rows)")


if __name__ == "__main__":
    main()
