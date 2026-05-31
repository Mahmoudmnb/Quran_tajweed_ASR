"""
phoneme_label_fixer.py
======================
Fix ASR training labels by detecting repeated phoneme segments in the
model-generated output and inserting the corresponding LABEL slice
(not the noisy generated phonemes) into the original label.

Algorithm
---------
1. Needleman-Wunsch alignment of `generated` vs `label`.
   Each alignment position is tagged M/X/I/D.

2. Contiguous 'I' (insert-in-generated) runs are collected with their
   label-anchor = the label index just before the inserted block.

3. Short / noisy runs are discarded.  Adjacent runs (anchors within
   `merge_gap` of each other) are merged into one candidate block.

4. Each candidate is verified: slide a window over the label region
   before the anchor and find the sub-sequence most similar to the
   candidate.  If similarity >= threshold → accept.

5. Build corrected label by inserting label[best_s:best_e]
   (the verified label slice) at the anchor position.
   *** The generated phonemes are NEVER copied into the corrected label. ***
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# Phoneme similarity / normalisation
# ─────────────────────────────────────────────────────────────

_NORM = {
    "aa": "a",
    "ii": "i",
    "uu": "u",
    "mm": "m",
    "nn": "n",
    "bb": "b",
    "nz": "n",
    "bK": "b",
    "jK": "j",
    "dK": "d",
}


def _norm(p: str) -> str:
    return _NORM.get(p, p)


def _similar(a: str, b: str) -> bool:
    return a == b or _norm(a) == _norm(b)


# ─────────────────────────────────────────────────────────────
# Step 1 — Needleman-Wunsch alignment
# ─────────────────────────────────────────────────────────────


def _align(
    generated: List[str],
    label: List[str],
    match: float = 1.0,
    mismatch: float = -0.4,
    gap: float = -0.3,
) -> Tuple[List[str], List[str], List[str]]:
    G, L = len(generated), len(label)
    dp = [[0.0] * (L + 1) for _ in range(G + 1)]
    for i in range(1, G + 1):
        dp[i][0] = dp[i - 1][0] + gap
    for j in range(1, L + 1):
        dp[0][j] = dp[0][j - 1] + gap

    for i in range(1, G + 1):
        for j in range(1, L + 1):
            sc = match if _similar(generated[i - 1], label[j - 1]) else mismatch
            dp[i][j] = max(
                dp[i - 1][j - 1] + sc,
                dp[i - 1][j] + gap,
                dp[i][j - 1] + gap,
            )

    gen_a, lbl_a, ops = [], [], []
    i, j = G, L
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sc = match if _similar(generated[i - 1], label[j - 1]) else mismatch
            if dp[i][j] == dp[i - 1][j - 1] + sc:
                gen_a.append(generated[i - 1])
                lbl_a.append(label[j - 1])
                ops.append("M" if _similar(generated[i - 1], label[j - 1]) else "X")
                i -= 1
                j -= 1
                continue
        if i > 0 and (j == 0 or dp[i][j] == dp[i - 1][j] + gap):
            gen_a.append(generated[i - 1])
            lbl_a.append("-")
            ops.append("I")
            i -= 1
        else:
            gen_a.append("-")
            lbl_a.append(label[j - 1])
            ops.append("D")
            j -= 1

    for lst in (gen_a, lbl_a, ops):
        lst.reverse()
    return gen_a, lbl_a, ops


# ─────────────────────────────────────────────────────────────
# Step 2 — Collect INSERT runs with label-anchor
# ─────────────────────────────────────────────────────────────


def _collect_runs(gen_a: List[str], ops: List[str]) -> List[dict]:
    runs = []
    label_consumed = -1
    i = 0
    while i < len(ops):
        if ops[i] in ("M", "X", "D"):
            label_consumed += 1
            i += 1
        elif ops[i] == "I":
            run_phonemes = []
            while i < len(ops) and ops[i] == "I":
                run_phonemes.append(gen_a[i])
                i += 1
            runs.append({"phonemes": run_phonemes, "anchor": label_consumed})
        else:
            i += 1
    return runs


# ─────────────────────────────────────────────────────────────
# Step 3 — Merge adjacent runs
# ─────────────────────────────────────────────────────────────


def _merge_runs(runs: List[dict], gap_tolerance: int = 4) -> List[dict]:
    if not runs:
        return []
    merged = [dict(phonemes=list(runs[0]["phonemes"]), anchor=runs[0]["anchor"])]
    for r in runs[1:]:
        last = merged[-1]
        if r["anchor"] - last["anchor"] <= gap_tolerance:
            last["phonemes"] = last["phonemes"] + r["phonemes"]
        else:
            merged.append(dict(phonemes=list(r["phonemes"]), anchor=r["anchor"]))
    return merged


# ─────────────────────────────────────────────────────────────
# Step 4 — Verify & find the matching LABEL slice
# ─────────────────────────────────────────────────────────────


def _similarity(a: List[str], b: List[str]) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _find_label_slice(
    candidate_gen: List[str],  # noisy phonemes from generated (used only for matching)
    label: List[str],
    anchor: int,
    look_back: int = 100,
) -> Tuple[float, int, int]:
    """
    Find the label sub-sequence (label[s:e]) that best matches `candidate_gen`.
    Only searches in label[max(0, anchor-look_back) : anchor+1].
    Returns (similarity, s, e).
    """
    clen = len(candidate_gen)
    if clen == 0:
        return 0.0, 0, 0

    region_s = max(0, anchor - look_back + 1)
    region_e = anchor + 1

    best_sim, best_s, best_e = 0.0, 0, 0
    lo = max(2, int(clen * 0.70))
    hi = min(len(label), int(clen * 1.30) + 1)

    for wlen in range(lo, hi + 1):
        for s in range(region_s, region_e - wlen + 2):
            e = s + wlen
            if e > len(label):
                break
            sim = _similarity(candidate_gen, label[s:e])
            if sim > best_sim:
                best_sim, best_s, best_e = sim, s, e

    return best_sim, best_s, best_e


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────


@dataclass
class RepetitionInfo:
    anchor: int  # inserted after label[anchor]
    label_slice: List[str]  # exact label phonemes that are repeated (clean)
    gen_phonemes: List[str]  # what the model generated (noisy reference)
    similarity: float
    label_start: int  # label[label_start:label_end] = label_slice
    label_end: int


@dataclass
class FixResult:
    corrected_label: List[str]
    repetitions: List[RepetitionInfo]
    report: str


def fix_label(
    generated: List[str],
    label: List[str],
    similarity_threshold: float = 0.60,
    min_run_length: int = 5,
    look_back: int = 100,
    merge_gap: int = 4,
) -> FixResult:
    """
    Detect repetitions in `generated` and insert the corresponding clean
    LABEL slices into `label` at the correct positions.

    Parameters
    ----------
    generated            : phoneme list from demo ASR model (may contain errors)
    label                : original ground-truth phoneme list
    similarity_threshold : min SequenceMatcher ratio to accept a repetition
    min_run_length       : discard insert runs shorter than this
    look_back            : search window size (label phonemes before anchor)
    merge_gap            : merge adjacent insert runs within this many positions
    """
    gen_a, lbl_a, ops = _align(generated, label)

    raw_runs = _collect_runs(gen_a, ops)
    filtered = [r for r in raw_runs if len(r["phonemes"]) >= min_run_length]
    merged = _merge_runs(filtered, gap_tolerance=merge_gap)

    accepted: List[RepetitionInfo] = []
    for run in merged:
        anchor = run["anchor"]
        gen_ph = run["phonemes"]
        sim, s, e = _find_label_slice(gen_ph, label, anchor, look_back)
        if sim >= similarity_threshold:
            accepted.append(
                RepetitionInfo(
                    anchor=anchor,
                    label_slice=label[s:e],  # ← clean label phonemes, not generated
                    gen_phonemes=gen_ph,
                    similarity=sim,
                    label_start=s,
                    label_end=e,
                )
            )

    accepted.sort(key=lambda r: r.anchor)

    # Build corrected label
    corrected: List[str] = []
    insertion_map = {r.anchor: r for r in accepted}

    if -1 in insertion_map:
        corrected.append("<sil>")
        corrected.extend(insertion_map[-1].label_slice)

    for i, ph in enumerate(label):
        corrected.append(ph)
        if i in insertion_map:
            corrected.append("<sil>")  # silence before the repetition
            corrected.extend(insertion_map[i].label_slice)

    # Report
    lines = [
        "=" * 60,
        "PHONEME LABEL FIX REPORT",
        "=" * 60,
        f"Original label length : {len(label)}",
        f"Generated seq length  : {len(generated)}",
        f"Raw insert runs       : {len(raw_runs)}",
        f"After length filter   : {len(filtered)}",
        f"After merging         : {len(merged)}",
        f"Accepted (sim ≥ {similarity_threshold:.0%}) : {len(accepted)}",
        "",
    ]
    if accepted:
        lines.append("Accepted repetitions (from LABEL, not generated):")
        for k, r in enumerate(accepted, 1):
            preview = r.label_slice[:12]
            ellipsis = "…" if len(r.label_slice) > 12 else ""
            lines.append(
                f"  [{k}] after label[{r.anchor:3d}] | "
                f"sim={r.similarity:.2f} | "
                f"label[{r.label_start}:{r.label_end}] (len={r.label_end-r.label_start}) | "
                f"{preview}{ellipsis}"
            )
    lines += [
        "",
        f"Corrected label length: {len(corrected)}",
        "=" * 60,
    ]
    return FixResult(
        corrected_label=corrected, repetitions=accepted, report="\n".join(lines)
    )


# ─────────────────────────────────────────────────────────────
# Utility: side-by-side diff
# ─────────────────────────────────────────────────────────────


def print_diff(original: List[str], corrected: List[str], chunk: int = 20) -> None:
    max_len = max(len(original), len(corrected))
    orig_p = original + ["·"] * (max_len - len(original))
    corr_p = corrected + ["·"] * (max_len - len(corrected))
    W = 6
    for s in range(0, max_len, chunk):
        e = s + chunk
        n = min(chunk, max_len - s)
        print(f"[{s:3d}-{s+n-1:3d}]")
        print("  ORIG : " + " ".join(f"{p:<{W}}" for p in orig_p[s:e]))
        print("  CORR : " + " ".join(f"{p:<{W}}" for p in corr_p[s:e]))
        marks = ["^" if orig_p[s + i] != corr_p[s + i] else " " for i in range(n)]
        print("         " + " ".join(f"{m:<{W}}" for m in marks))
        print()


# ─────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    label = [
        "ʔ",
        "a",
        "l",
        "ħ",
        "a",
        "m",
        "d",
        "u",
        "l",
        "i",
        "l",
        "l",
        "aa",
        "h",
        "i",
        "l",
        "l",
        "a",
        "ð",
        "ii",
        "ʔ",
        "a",
        "nz",
        "z",
        "a",
        "l",
        "a",
        "ʕ",
        "a",
        "l",
        "aa",
        "ʕ",
        "a",
        "bK",
        "d",
        "i",
        "h",
        "i",
        "l",
        "k",
        "i",
        "t",
        "aa",
        "b",
        "a",
        "w",
        "a",
        "l",
        "a",
        "m",
        "y",
        "a",
        "jK",
        "ʕ",
        "a",
        "l",
        "l",
        "a",
        "h",
        "uu",
        "ʕ",
        "i",
        "w",
        "a",
        "j",
        "aa",
    ]

    generated = [
        "ʔ",
        "l",
        "ħ",
        "a",
        "m",
        "m",
        "d",
        "u",
        "l",
        "i",
        "l",
        "l",
        "aa",
        "l",
        "i",
        "l",
        "l",
        "a",
        "ð",
        "ii",
        "ʔ",
        "a",
        "nz",
        "z",
        "a",
        "l",
        "a",
        "ʕ",
        "a",
        "l",
        "aa",
        "ʕ",
        "a",
        "bK",
        "d",
        "i",
        "h",
        "i",
        "l",
        "k",
        "i",
        "t",
        "aa",
        "mm",
        "b",  # model error: mm instead of a
        "ʔ",
        "l",
        "l",
        "a",
        "ð",
        "ii",
        "ʔ",
        "a",
        "nz",
        "z",
        "a",
        "l",
        "ʕ",
        "a",
        "l",
        "aa",
        "ʕ",
        "a",
        "bK",
        "d",
        "i",
        "h",
        "i",
        "l",
        "k",
        "i",
        "t",
        "aa",
        "b",
        "ɣ",
        "a",
        "l",
        "m",  # model error: ɣ not in label
        "y",
        "jK",
        "ʕ",
        "a",
        "l",
        "l",
        "a",
        "h",
        "uu",
        "ʕ",
        "i",
        "w",
        "a",
        "j",
        "aa",
        "<sil>",
    ]

    result = fix_label(
        generated=generated,
        label=label,
        similarity_threshold=0.60,
        min_run_length=5,
        merge_gap=4,
    )

    print(result.report)

    print("CORRECTED LABEL:")
    print(result.corrected_label)

    # Verify: no phoneme in corrected that isn't in label
    label_set = set(label) | {"<sil>"}
    foreign = [p for p in result.corrected_label if p not in label_set]
    print(f"\nForeign phonemes (not in label): {foreign}")
    print(
        "✓ All phonemes are from the label"
        if not foreign
        else "✗ Found foreign phonemes!"
    )

    print("\nSIDE-BY-SIDE DIFF:")
    print_diff(label, result.corrected_label)
