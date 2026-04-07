import re
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB

"""
Phoneme encoder/decoder utilities for Arabic (Quranic) text to phonemes.
"""


# Load morphology database
_db = MorphologyDB.builtin_db()

# Create analyzer
_analyzer = Analyzer(_db)


@lru_cache(maxsize=50000)
def _cached_analysis(word: str):
    return _get_best_analysis(word)


# ================================
# Arabic Diacritics
# ================================

FATHA = "َ"
DAMMA = "ُ"
KASRA = "ِ"
SUKOON = "ْ"
SHADDA = "ّ"
FATHATAN = "ً"
DAMMATAN = "ٌ"
KASRATAN = "ٍ"
LONG_ALIF = "ا"
LONG_WAW = "و"
LONG_YA = "ي"
DAGGER_ALIF = "ٰ"
WASL = "<wasl>"
HAMZAT_WASL = "ٱ"

DIACRITICS: List[str] = [
    FATHA,
    DAMMA,
    KASRA,
    SUKOON,
    SHADDA,
    FATHATAN,
    DAMMATAN,
    KASRATAN,
    DAGGER_ALIF,
]

# ================================
# Special tokens
# ================================

SPACE_TOKEN = "<space>"
BLANK_TOKEN = "<blank>"

WASL_NOUNS = {
    "اسم",
    "ابن",
    "ابنة",
    "اثنان",
    "اثنين",
    "اثنتان",
    "اثنتين",
    "امرؤ",
    "امرأة",
}

NORMALIZED_WASL_NOUNS = {re.sub(r"[أإؤئء]", "ا", w) for w in WASL_NOUNS}


RELATIVE_PRONOUNS = {
    "الذي",
    "التي",
    "الذين",
    "اللذان",
    "اللذين",
    "اللتان",
    "اللتين",
    "اللاتي",
    "اللائي",
    "اللواتي",
}

PREFIX_LETTERS = {
    "و",
    "ف",
    "ب",
    "ك",
    "ل",
}


# ================================
# Quranic marks (to remove later)
# ================================

QURANIC_MARKS: List[str] = ["ۖ", "ۗ", "ۘ", "ۙ", "ۚ", "ۛ", "ۜ", "۝", "۞", "۩"]

# ================================
# Hamza variants
# ================================

HAMZA_SET = {"ء", "أ", "إ", "ؤ", "ئ"}

# ================================
# Arabic → Phoneme (IPA)
# ================================

ARABIC_TO_PHONEME: Dict[str, str] = {
    "ء": "ʔ",
    "أ": "ʔ",
    "إ": "ʔ",
    "ؤ": "ʔ",
    "ئ": "ʔ",
    "ى": "aa",
    "ب": "b",
    "ت": "t",
    "ث": "θ",
    "ج": "j",
    "ح": "ħ",
    "خ": "x",
    "د": "d",
    "ذ": "ð",
    "ر": "r",
    "ز": "z",
    "س": "s",
    "ش": "ʃ",
    "ص": "sˤ",
    "ض": "dˤ",
    "ط": "tˤ",
    "ظ": "ðˤ",
    "ع": "ʕ",
    "غ": "ɣ",
    "ف": "f",
    "ق": "q",
    "ك": "k",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ه": "h",
    "ة": "T",
    "و": "w",
    "ي": "y",
}

# ================================
# Phoneme → Arabic (for debugging)
# ================================

PHONEME_TO_ARABIC: Dict[str, str] = {
    "ʔ": "ء",
    "ʔw": "ٱ",
    "b": "ب",
    "t": "ت",
    "θ": "ث",
    "j": "ج",
    "ħ": "ح",
    "x": "خ",
    "d": "د",
    "ð": "ذ",
    "r": "ر",
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
    "T": "ة",
    "w": "و",
    "y": "ي",
    "aa": "ا",
    "a": "َ",
    "i": "ِ",
    "u": "ُ",
}

# ================================
# Vowels
# ================================

SHORT_VOWELS: List[str] = ["a", "i", "u"]
LONG_VOWELS: List[str] = ["aa", "ii", "uu"]
TANWEEN: List[str] = ["an", "in", "un"]

NO_MADD_U_WAW_WORDS = {
    "اولات",
    "اولي",
    "اولوا",
}

MUQATTAAT = {
    "ٱلم": ["ʔa", "li", "f", "laa", "m", "mii", "m"],
    "المص": ["?a", "li", "f", "laa", "m", "mii", "m", "sˤaa", "d"],
    "الر": ["?a", "li", "f", "laa", "m", "raa"],
    "المر": ["?a", "li", "f", "laa", "m", "mii", "m", "raa"],
    "كهيعص": [
        "kaa",
        "f",
        "haa",
        "yaa",
        "ʕa",
        "y" "n",
        "sˤaa",
        "d",
    ],
    "طه": ["tˤaa", "haa"],
    "طسم": ["tˤaa", "sii", "m", "mii", "m"],
    "طس": ["tˤaa", "sii", "n"],
    "يس": ["yaa", "sii", "n"],
    "ص": ["sˤaa", "d"],
    "حم": ["ħaa", "mii", "m"],
    "ق": ["qaa", "f"],
    "ن": ["nuu", "n"],
    "عسق": ["؟a", "yii", "n", "sii", "n", "qaa", "f"],
}


# ================================
# Consonant inventory
# ================================

CONSONANTS: List[str] = [
    "ʔ",
    "ʔw",
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
]

SUN_LETTERS = {"ت", "ث", "د", "ذ", "ر", "ز", "س", "ش", "ص", "ض", "ط", "ظ", "ل", "ن"}

# Used in phonemes_to_text shadda detection
CONSONANT_BASES = {
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
}

GLIDES = {"w", "y"}

# ================================
# Build syllable phonemes
# ================================

CV_SHORT: List[str] = [c + v for c in CONSONANTS for v in SHORT_VOWELS]
CV_LONG: List[str] = [c + v for c in CONSONANTS for v in LONG_VOWELS]
CV_TANWEEN: List[str] = [c + v for c in CONSONANTS for v in TANWEEN]

# ================================
# Phoneme vocabulary
# ================================

PHONEMES: List[str] = (
    ["sil", SPACE_TOKEN, WASL] + CONSONANTS + CV_SHORT + CV_LONG + CV_TANWEEN
)

# ================================
# Index mappings
# ================================

IDX2PHONEME: Dict[int, str] = {i: p for i, p in enumerate(PHONEMES)}
PHONEME2IDX: Dict[str, int] = {p: i for i, p in enumerate(PHONEMES)}

# ================================
# CTC vocabulary
# ================================

PHONEMES_CTC: List[str] = [BLANK_TOKEN] + PHONEMES

phoneme_to_id: Dict[str, int] = {p: i for i, p in enumerate(PHONEMES_CTC)}
id_to_phoneme: Dict[int, str] = {i: p for p, i in phoneme_to_id.items()}

blank_id: int = phoneme_to_id[BLANK_TOKEN]

# # ================================
# # Madd helpers
# # ================================

SHORT_IDS: List[int] = [
    phoneme_to_id[p] for p in PHONEMES_CTC if any(p.endswith(v) for v in SHORT_VOWELS)
]

LONG_IDS: List[int] = [
    phoneme_to_id[p] for p in PHONEMES_CTC if any(p.endswith(v) for v in LONG_VOWELS)
]


def normalize_lafz_aljalala(text: str) -> str:
    """
    Normalize all forms of 'Allah' to include dagger alif:
    اللَّه → اللّٰه
    لِلَّهِ → لِلّٰهِ
    """
    result = []
    i = 0
    n = len(text)

    while i < n:
        # try to detect pattern: ل + (diacs) + ل + (diacs) + ه
        if text[i] == "ل":
            j = i + 1

            # collect diacritics after first lam
            while j < n and text[j] in DIACRITICS:
                j += 1

            # second lam
            if j < n and text[j] == "ل":
                k = j + 1

                lam2_diacs = []
                while k < n and text[k] in DIACRITICS:
                    lam2_diacs.append(text[k])
                    k += 1

                # followed by ه → THIS IS ALLAH
                if k < n and text[k] == "ه":
                    # 🔥 normalize to: ل + ل + شدة + dagger alif + ه
                    # preserve diacritics before first lam
                    result.append("ل")

                    # keep any diacritics that were already there (IMPORTANT)
                    for d in text[i + 1 : j]:
                        if d in DIACRITICS:
                            result.append(d)

                    # normalized second lam
                    result.append("ل")
                    result.append("ّ")
                    result.append("ٰ")
                    result.append("ه")

                    i = k + 1
                    continue

        # default
        result.append(text[i])
        i += 1

    return "".join(result)


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text for ASR training while preserving
    phonetic information (especially hamza forms).
    """

    # Remove Qur’anic pause marks
    for mark in QURANIC_MARKS:
        text = text.replace(mark, "")

    # Remove tatweel
    text = text.replace("ـ", "")

    # Remove non-Arabic characters but keep spaces
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    text = normalize_lafz_aljalala(text)

    return text


def _append_meta(
    metadata: List[Dict[str, object]],
    char: str,
    char_index: int,
    is_long_vowel: bool,
) -> None:
    """Track original character + position, and whether the produced phoneme is long."""
    metadata.append(
        {"char": char, "char_index": char_index, "is_long_vowel": is_long_vowel}
    )


def _collect_diacritics(chars: Sequence[str], start_idx: int) -> Tuple[List[str], int]:
    """Consume a run of Arabic diacritics starting at start_idx."""
    diacs = []
    i = start_idx
    while i < len(chars) and chars[i] in DIACRITICS:
        diacs.append(chars[i])
        i += 1
    return diacs, i


def _short_vowel_from_diacritics(diacs: Sequence[str]) -> Optional[str]:
    """Map diacritics to short vowel if present."""
    if FATHA in diacs:
        return "a"
    if KASRA in diacs:
        return "i"
    if DAMMA in diacs:
        return "u"
    return None


def _tanween_from_diacritics(diacs: Sequence[str]) -> Optional[str]:
    """Map tanween diacritics to tanween phoneme suffix."""
    if FATHATAN in diacs:
        return "an"
    if KASRATAN in diacs:
        return "in"
    if DAMMATAN in diacs:
        return "un"
    return None


def _remove_diacritics(text: str) -> str:
    return "".join(ch for ch in text if ch not in DIACRITICS)


def _is_prefix_plus_al(chars: Sequence[str], idx: int) -> bool:

    if idx - 2 < 0:
        return False

    if chars[idx - 1] not in [FATHA, KASRA, DAMMA]:  #!= FATHA:
        return False

    prefix_letter = chars[idx - 2]
    if prefix_letter not in PREFIX_LETTERS:
        return False

    prefix_pos = idx - 2
    if prefix_pos - 1 >= 0 and chars[prefix_pos - 1] != " ":
        return False

    if idx + 1 >= len(chars) or chars[idx + 1] != "ل":
        return False

    # inspect lam
    j = idx + 2
    lam_diacs = []

    while j < len(chars) and chars[j] in DIACRITICS:
        lam_diacs.append(chars[j])
        j += 1

    # 🔥 key logic
    has_vowel = any(d in [FATHA, DAMMA, KASRA] for d in lam_diacs)

    return (
        SUKOON in lam_diacs
        or SHADDA in lam_diacs
        or (not lam_diacs and not has_vowel)  # implicit sukoon
    )


def _append_wasl_safe(
    phonemes: List[str], metadata: List[Dict[str, object]], char_index: int
) -> None:
    if phonemes and phonemes[-1] == WASL:
        return
    phonemes.append(WASL)
    _append_meta(metadata, "ا", char_index, False)


def _handle_pre_mapping(
    chars: Sequence[str],
    i: int,
    phonemes: List[str],
    metadata: List[Dict[str, object]],
) -> Tuple[bool, int]:

    ch = chars[i]
    char_index = i

    # Only act on very specific trigger characters
    if ch not in [HAMZAT_WASL, " ", "آ", "ى", "ل", "س", "ا"]:
        return False, i

    # -------------------
    # COLLAPSED "ال" (NO ALIF) — morphology-driven
    # -------------------
    if ch == "ل" and i >= 1:

        # detect previous letter (prefix)
        prev_char = chars[i - 1]

        if prev_char in DIACRITICS:
            # get previous real letter
            k = i - 2
            while k >= 0 and chars[k] in DIACRITICS:
                k -= 1

            # -------------------
            # 🔥 CRITICAL FILTER: this lam must be "ال" lam
            # → it must NOT have vowel or shadda
            # -------------------
            j = i + 1
            lam_diacs = []

            while j < len(chars) and chars[j] in DIACRITICS:
                lam_diacs.append(chars[j])
                j += 1

            # ❌ if lam has vowel OR shadda → NOT "ال"
            if any(d in [FATHA, DAMMA, KASRA, SHADDA] for d in lam_diacs):
                return False, i

            # -------------------
            # ❌ EXCLUDE "اللّٰه"
            # -------------------
            if j < len(chars) and chars[j] == "ه" and DAGGER_ALIF in lam_diacs:
                return False, i

            if k >= 0 and chars[k] in PREFIX_LETTERS:

                # 🔥 get the WORD (not just chars)
                word_start = i
                while word_start > 0 and chars[word_start - 1] != " ":
                    word_start -= 1

                word_end = i
                while word_end < len(chars) and chars[word_end] != " ":
                    word_end += 1

                word = "".join(chars[word_start:word_end])
                clean = _remove_diacritics(word)

                analysis = _cached_analysis(clean)

                # 🔥 HARD CONDITION (no guessing)
                has_al_morph = analysis.get("_force_has_al", False)

                # -------------------
                # STRICT surface fallback (ONLY for collapsed lam-lam case)
                # -------------------
                has_al_surface = False

                # next real letter after current lam
                j = i + 1
                while j < len(chars) and chars[j] in DIACRITICS:
                    j += 1

                if j < len(chars):
                    next_letter = chars[j]

                    # must be valid Arabic letter AND not lam (avoid لَلَل patterns)
                    if next_letter in ARABIC_TO_PHONEME and next_letter != "ل":
                        has_al_surface = True

                # 🔥 final decision
                if has_al_morph or has_al_surface:

                    # insert wasl
                    _append_wasl_safe(phonemes, metadata, char_index)

                    j = i + 1

                    # skip lam if sun letter
                    if j < len(chars):
                        next_letter = chars[j]

                        if next_letter in SUN_LETTERS:
                            return True, j

                    # moon letter → keep lam
                    phonemes.append("l")
                    _append_meta(metadata, "ل", char_index, False)

                    return True, j

    # -------------------
    # IMPLICIT "ال" after prefixes (كَال، فَال، وَال)
    # -------------------
    if (
        ch == "ل"
        and i >= 3
        and chars[i - 1] == "ا"
        and chars[i - 2] in DIACRITICS
        and chars[i - 3] in PREFIX_LETTERS
        and (i - 4 < 0 or chars[i - 4] == " ")
        and _is_prefix_plus_al(chars, i - 1)
    ):
        j = i + 1
        lam_diacs = []

        # collect lam diacritics
        while j < len(chars) and chars[j] in DIACRITICS:
            lam_diacs.append(chars[j])
            j += 1
        next_letter = chars[j] if j < len(chars) else None

        if not (
            SUKOON in lam_diacs or SHADDA in lam_diacs or next_letter in SUN_LETTERS
        ):

            return False, i  # ❗ NOT definite article → skip this rule

        if j < len(chars):
            next_letter = chars[j]

            # insert wasl
            _append_wasl_safe(phonemes, metadata, char_index)

            # 🔥 CASE 1: lam has shadda (rare but valid)
            if SHADDA in lam_diacs:
                phonemes.append("l")
                _append_meta(metadata, "ل", char_index, False)

                phonemes.append("la")
                _append_meta(metadata, "ل", char_index, False)

                return True, j

            # 🔥 CASE 2: SUN LETTER → SKIP LAM COMPLETELY
            if next_letter in SUN_LETTERS:
                return True, j  # ✅ NO "l" appended

            # 🔥 CASE 3: MOON LETTER → KEEP LAM
            phonemes.append("l")
            _append_meta(metadata, "ل", char_index, False)

            return True, j

    # HIDDEN WASL in "بالله", "كالله", "فلله", etc.
    if (
        ch == "ا"
        and i > 0
        and chars[i - 1] in DIACRITICS
        and i + 1 < len(chars)
        and chars[i + 1] == "ل"
    ):

        # treat it exactly like definite article
        _append_wasl_safe(phonemes, metadata, char_index)

        j = i + 2  # move after lam
        lam_diacs = []

        while j < len(chars) and chars[j] in DIACRITICS:
            lam_diacs.append(chars[j])
            j += 1

        # 🔥 CASE 1: LAM HAS SHADDA (الَّذِينَ ، اللَّه)
        if SHADDA in lam_diacs:
            # DO NOT emit phoneme here
            # let main loop handle lam normally (it will double it)
            return True, i + 1  # go to lam

        # 🔥 CASE 2: NORMAL SUN LETTER
        if j < len(chars):
            next_letter = chars[j]

            if next_letter in SUN_LETTERS:
                return True, j  # skip lam

        # 🌙 CASE 3: MOON LETTER
        phonemes.append("l")
        _append_meta(metadata, "ل", char_index + 1, False)

        return True, j

    # -------------------
    # DEFINITE ARTICLE (ٱل)
    # -------------------
    if (
        ch == HAMZAT_WASL
        and i + 1 < len(chars)
        and chars[i + 1] == "ل"
        # ❗ NEW: skip if preceded by prefix pattern
        and not (
            i >= 2 and chars[i - 1] in DIACRITICS and chars[i - 2] in PREFIX_LETTERS
        )
    ):

        # check if lam itself has shadda
        j = i + 2
        lam_diacs = []

        while j < len(chars) and chars[j] in DIACRITICS:
            lam_diacs.append(chars[j])
            j += 1
        # -------------------
        # FIX: lam with SUKOON → NOT definite article
        # (e.g., الْتَقَطَ)
        # -------------------
        if SUKOON in lam_diacs:
            _append_wasl_safe(phonemes, metadata, char_index)

            phonemes.append("l")
            _append_meta(metadata, "ل", char_index + 1, False)

            return True, j

        # case 1: lam has shadda (الَّذِينَ , الَّتِي ...)
        if SHADDA in lam_diacs:

            # add wasl
            _append_wasl_safe(phonemes, metadata, char_index)

            # DO NOT generate phoneme here
            # just move to lam so normal pipeline handles it
            return True, i + 1

        # case 2: normal definite article
        if j < len(chars):
            next_letter = chars[j]

            _append_wasl_safe(phonemes, metadata, char_index)

            if next_letter in SUN_LETTERS:
                return True, j

            else:
                phonemes.append("l")
                _append_meta(metadata, "ل", char_index + 1, False)

                return True, j

    # Detect collapsed Hamzat-Wasl nouns after prefixes (e.g., بِابْنِ)
    if i >= 2 and chars[i - 1] in DIACRITICS and chars[i - 2] in PREFIX_LETTERS:
        # collect letters starting from current position
        j = i
        letters = []

        while j < len(chars) and len(letters) < 4:
            if chars[j] not in DIACRITICS:
                letters.append(chars[j])
            j += 1

        candidate = "".join(letters)

        # try matching BOTH:
        # 1) full noun (for cases where alif exists)
        # 2) noun without alif (for collapsed cases)
        for noun in WASL_NOUNS:
            if candidate.startswith(noun) or candidate.startswith(noun[1:]):
                _append_wasl_safe(phonemes, metadata, i)
                break

    # -------------------
    # SPACE
    # -------------------
    if ch == " ":
        phonemes.append(SPACE_TOKEN)
        _append_meta(metadata, ch, char_index, False)
        return True, i + 1

    # -------------------
    # EXPLICIT HAMZAT WASL (ٱ)
    # -------------------
    if ch == HAMZAT_WASL:

        # ✅ prevent duplicate <wasl>
        if phonemes and phonemes[-1] == WASL:
            return True, i + 1

        _append_wasl_safe(phonemes, metadata, char_index)
        return True, i + 1

    # -------------------
    # ALIF MADDAH (آ)
    # -------------------
    # phonetic: ʔaa
    if ch == "آ":
        phonemes.append("ʔaa")
        _append_meta(metadata, ch, char_index, True)
        return True, i + 1
    # -------------------
    # ALIF MAQSURA (ى)
    # -------------------
    # behaves like long aa

    if ch == "ى":

        if phonemes:
            last = phonemes[-1]

            if last.endswith("a") and last not in ["aa", "an"]:
                phonemes[-1] = last[:-1] + "aa"
            else:
                phonemes.append("aa")
        else:
            phonemes.append("aa")

        _append_meta(metadata, ch, char_index, True)

        return True, i + 1

    return False, i


def _is_pure_madd_letter(chars: Sequence[str], idx: int) -> bool:
    """
    A madd letter is valid if:
    - it has NO vowel on it
    - it has NO shadda on it
    (we do NOT care about later letters)
    """

    j = idx + 1

    while j < len(chars) and chars[j] in DIACRITICS:
        if chars[j] in [FATHA, DAMMA, KASRA, FATHATAN, DAMMATAN, KASRATAN, SHADDA]:
            return False
        j += 1

    return True


def _is_real_madd_context(chars: Sequence[str], idx: int) -> bool:
    """
    Avoid false madd in patterns like جِبَال
    """
    j = idx + 1

    # skip diacritics
    while j < len(chars) and chars[j] in DIACRITICS:
        j += 1

    # if next letter has sukoon → NOT madd
    if j + 1 < len(chars) and chars[j + 1] == SUKOON:
        return False

    return True


def _prev_letters(chars: Sequence[str], idx: int, n: int = 2) -> List[str]:
    letters = []
    j = idx - 1

    while j >= 0 and len(letters) < n:
        if chars[j] not in DIACRITICS:
            letters.append(chars[j])
        j -= 1

    return letters


from functools import lru_cache


@lru_cache(maxsize=50000)
def _get_best_analysis(word: str) -> dict:
    """
    Analyzer-only with fallback signals from ALL analyses.
    """

    analyses = _analyzer.analyze(word)

    if not analyses:
        return {}

    # -------------------
    # GLOBAL SIGNALS (from all analyses)
    # -------------------
    has_al_any = any(a.get("prc2") == "ال" for a in analyses)

    # -------------------
    # SCORING (pick best analysis)
    # -------------------
    def score(a):
        s = 0

        if a.get("prc2") == "ال":
            s += 5

        if a.get("prc3") not in (None, "0"):
            s += 2

        if a.get("pos") in ("noun", "verb"):
            s += 2

        if a.get("pos") == "unk":
            s -= 5

        return s

    best = max(analyses, key=score)

    # -------------------
    # 🔥 Inject robust signal
    # -------------------
    best["_force_has_al"] = has_al_any

    return best


def _has_definite_article(analysis: dict, word: str) -> bool:
    """
    Detect definite article using BOTH morphology and surface form.
    """

    # morphology
    if analysis.get("prc2") == "ال":
        return True

    # 🔥 fallback: surface form
    clean = _remove_diacritics(word)

    if clean.startswith("ال"):
        return True

    return False


def _has_prefix(analysis: dict) -> str:
    """
    Convert CAMeL prefix tags into actual Arabic letters.
    """

    mapping = {
        "wa_sub": "و",
        "fa_conj": "ف",
        "bi_prep": "ب",
        "li_prep": "ل",
        "ka_prep": "ك",
    }

    prefixes = []

    for key in ["prc3", "prc2", "prc1", "prc0"]:
        val = analysis.get(key)

        if not val or val == "0" or val == "ال":
            continue

        if val in mapping:
            prefixes.append(mapping[val])

    return "".join(prefixes)


def _is_verb_with_wasl(analysis: dict, word: str) -> bool:
    """
    Detect verbs that take hamzat wasl using surface form,
    including prefixes (و، ف، ب، ك، ل).
    """

    stem = analysis.get("stem", "")

    if stem.startswith("ٱ"):
        return True

    if analysis.get("pos") != "verb":
        return False

    clean = _remove_diacritics(word)

    # -------------------
    # strip prefix if exists
    # -------------------
    prefix = _has_prefix(analysis)

    if prefix:
        stem = clean[len(prefix) :]
    else:
        stem = clean

    # -------------------
    # check patterns on stem
    # -------------------
    if stem.startswith("است"):  # Form X
        return True

    if stem.startswith("ان"):  # Form VII
        return True

    if stem.startswith("ات"):  # Form VIII
        return True

    if stem.startswith("ا"):  # Imperative
        return True

    return False


def _insert_wasl(word: str, insert_pos: int) -> str:
    """
    Replace the FIRST real alif AFTER prefix (skipping diacritics properly).
    """

    chars = list(word)

    i = insert_pos

    # 🔥 skip diacritics AFTER prefix
    while i < len(chars) and chars[i] in DIACRITICS:
        i += 1

    # 🔥 now we MUST hit alif
    if i < len(chars) and chars[i] == "ا":
        chars[i] = HAMZAT_WASL
        return "".join(chars)

    # ❌ if not found → DO NOTHING (do NOT insert blindly)
    return word


def _is_wasl_noun(analysis: dict) -> bool:
    lex = analysis.get("lex", "")

    # remove suffix like _1, _2
    lex = lex.split("_")[0]

    return lex in WASL_NOUNS


def _is_imperative_wasl_form(word: str) -> bool:
    """
    Detect imperative verbs like:
    اِفْعِلْ ، اِفْعَلْ ، اِفْعُلْ
    """

    chars = list(word)

    if not chars or chars[0] != "ا":
        return False

    # must have kasra on first letter (strong signal)
    if len(chars) > 1 and chars[1] != KASRA:
        return False

    # find next real letter
    j = 2
    while j < len(chars) and chars[j] in DIACRITICS:
        j += 1

    # must exist and have sukoon → pattern اِفْ
    if j < len(chars):
        k = j + 1
        while k < len(chars) and chars[k] in DIACRITICS:
            if chars[k] == SUKOON:
                return True
            k += 1

    return False


def _mark_hamzat_wasl(text: str) -> str:
    """
    Morphology-driven Hamzat Wasl insertion using CAMeL Tools.
    """

    words = text.split()
    new_words = []

    for word in words:

        # remove diacritics for analysis ONLY
        clean = _remove_diacritics(word)

        if not clean:
            new_words.append(word)
            continue

        # analysis = _get_best_analysis(clean)
        analysis = _cached_analysis(clean)

        if not analysis:
            new_words.append(word)
            continue

        has_al = _has_definite_article(analysis, word)
        prefix = _has_prefix(analysis)

        # -------------------
        # STRONG prefix detection
        # -------------------
        if prefix:
            stem = clean[len(prefix) :]

        # fallback (ONLY if morphology failed)
        elif (
            len(clean) > 2
            and clean[0] in PREFIX_LETTERS
            and clean[1] == "ا"  # 🔥 CRITICAL constraint
        ):
            # only strip if followed by alif (strong signal of wasl word)
            stem = clean[1:]

        else:
            stem = clean

        needs_wasl = False

        # -------------------
        # RULE 1: definite article (ال)
        # -------------------
        if has_al:
            needs_wasl = True

        # -------------------
        # RULE 2: wasl nouns
        # -------------------
        elif _is_wasl_noun(analysis) or any(
            _normalize_hamza(stem).startswith(noun) for noun in NORMALIZED_WASL_NOUNS
        ):
            needs_wasl = True

        # -------------------
        # RULE 3: verbs
        # -------------------
        elif _is_verb_with_wasl(analysis, word) or _is_imperative_wasl_form(word):
            needs_wasl = True

        if not needs_wasl:
            new_words.append(word)
            continue

        # -------------------
        # Determine insertion position
        # -------------------
        # determine insertion position directly from text (NOT morphology)
        chars = list(word)

        insert_pos = 0

        # if word starts with prefix letter + diacritic → skip both
        if len(chars) >= 2 and chars[0] in PREFIX_LETTERS and chars[1] in DIACRITICS:
            insert_pos = 2
        else:
            insert_pos = 0

        # -------------------
        # Apply insertion
        # -------------------
        new_word = _insert_wasl(word, insert_pos)

        new_words.append(new_word)

    return " ".join(new_words)


def _normalize_hamza(text: str) -> str:
    return re.sub(r"[أإؤئء]", "ا", text)


def _strip_prefixes_morph(word: str) -> str:
    """
    Remove prefixes using morphological analysis (SAFE).
    """
    analysis = _cached_analysis(word)

    if not analysis:
        return word

    prefix = _has_prefix(analysis)

    if prefix and word.startswith(prefix):
        return word[len(prefix) :]

    return word


def encode_Moqatta_letters(text, i):

    new_i = i
    word = ""

    if new_i == 0 or text[new_i] == " ":

        new_i += 1

        while new_i < len(text) and text[new_i] != " ":
            word += text[new_i]
            new_i += 1

        if word in MUQATTAAT:
            return MUQATTAAT[word], new_i

    return [], i


def text_to_phonemes_with_mapping(
    text: str,
    for_debug=True,
) -> Tuple[List[str], List[Dict[str, object]]]:

    # Normalize text first (remove marks, unify spacing)
    text = normalize_arabic(text)
    text = _mark_hamzat_wasl(text)

    phonemes = []
    metadata = []
    block_next_madd = False
    last_vowel = None

    # Iterate character-by-character
    i = 0
    chars = list(text)
    current_word_start = 0

    while i < len(chars):

        ph, i = encode_Moqatta_letters(chars, i)
        if len(ph) > 0:
            phonemes.append(SPACE_TOKEN)
            metadata.append(None)
            phonemes.extend(ph)
            metadata.extend([("muqattaat", None)] * len(ph))
            continue

        current_block_madd = block_next_madd
        block_next_madd = False

        # Pre-mapping handles special cases; if handled, move on
        handled, i = _handle_pre_mapping(chars, i, phonemes, metadata)
        if handled:
            continue

        ch = chars[i]
        char_index = i

        # track word boundaries
        if ch == " ":
            current_word_start = i + 1

        # extract current word
        word_end = current_word_start
        while word_end < len(chars) and chars[word_end] != " ":
            word_end += 1

        word = "".join(chars[current_word_start:word_end])
        core_word = _strip_prefixes_morph(word)  # ← use original word
        core_word = _normalize_hamza(_remove_diacritics(core_word))

        # Skip any character that isn't in our mapping (safety)
        if ch not in ARABIC_TO_PHONEME:
            i += 1
            continue

        # Base consonant phoneme
        base = ARABIC_TO_PHONEME[ch]
        i += 1

        # Collect trailing diacritics attached to this letter
        diacs, i = _collect_diacritics(chars, i)

        # Quick flags for common diacritics
        has_shadda = SHADDA in diacs
        has_sukun = SUKOON in diacs

        # Resolve short vowel or tanween, if any
        short_vowel = _short_vowel_from_diacritics(diacs)
        tanween = _tanween_from_diacritics(diacs)

        # Shadda duplicates the consonant
        if has_shadda:
            phonemes.append(base)
            _append_meta(metadata, ch, char_index, False)

        # Tanween handling (an/in/un)
        if tanween:

            phonemes.append(base + tanween)

            _append_meta(metadata, ch, char_index, False)

            # skip tanween support letters (كتابًا , هدىً)
            if tanween == "an" and i < len(chars) and chars[i] in ["ا", "ى"]:
                i += 1

            continue

        # -------------------
        # LONG VOWELS (MADD)
        # -------------------

        # Fatha + Alif → aa
        if (
            short_vowel == "a"
            and i < len(chars)
            and chars[i] == LONG_ALIF
            # ❗ NEW: do NOT treat as madd if this alif is part of "ال"
            # and not (i + 1 < len(chars) and chars[i + 1] == "ل")
            and not _is_prefix_plus_al(chars, i)
            and _is_real_madd_context(chars, i)
            and _is_pure_madd_letter(chars, i)
        ):
            phonemes.append(base + "aa")
            _append_meta(metadata, ch, char_index, True)
            i += 1
            continue

        # Kasra + Ya → ii (STRICT madd only)
        if (
            short_vowel == "i"
            and i < len(chars)
            and chars[i] == LONG_YA
            and not current_block_madd
        ):

            # 🔍 check if there is ANOTHER ya after this one
            lookahead = i + 1

            # skip diacritics
            while lookahead < len(chars) and chars[lookahead] in DIACRITICS:
                lookahead += 1

            # ❌ CASE: يِ + ي + ي  → NOT madd
            if lookahead < len(chars) and chars[lookahead] == LONG_YA:
                # ✅ emit normal short vowel
                phonemes.append(base + "i")
                _append_meta(metadata, ch, char_index, False)
                continue

            # ✅ REAL madd
            if _is_pure_madd_letter(chars, i):
                phonemes.append(base + "ii")
                last_vowel = "ii"
                _append_meta(metadata, ch, char_index, True)
                i += 1
                continue

        # Damma + Waw handling (madd + exceptions)
        if (
            short_vowel == "u"
            and i < len(chars)
            and chars[i] == LONG_WAW
            and _is_pure_madd_letter(chars, i)
        ):
            if core_word in NO_MADD_U_WAW_WORDS:
                # ❌ No madd, and waw is NOT pronounced
                phonemes.append(base + "u")
                _append_meta(metadata, ch, char_index, False)
                i += 1  # skip waw
                continue
            else:
                # ✅ Normal madd
                phonemes.append(base + "uu")
                _append_meta(metadata, ch, char_index, True)
                i += 1  # skip waw
                continue

        # -------------------
        # DAGGER ALIF
        # -------------------
        if DAGGER_ALIF in diacs:

            # extend previous consonant with aa
            phonemes.append(base + "aa")

            _append_meta(metadata, ch, char_index, True)

            continue
        # -------------------
        # SUKUN → pure consonant
        # -------------------
        if has_sukun:
            phonemes.append(base)
            _append_meta(metadata, ch, char_index, False)

            block_next_madd = True  # 🔥 TRACK IT
            continue

        # -------------------
        # MADD SILAH (SUGHRA + KUBRA)
        # -------------------

        prev_letters = _prev_letters(chars, char_index, 2)

        if ch == "ه" and short_vowel in ["u", "i"] and prev_letters != ["ل", "ل"]:

            applied = False

            # --- check next letter ---
            j = i
            while j < len(chars) and (chars[j] in DIACRITICS):
                j += 1

            same_word = j < len(chars) and chars[j] != " "

            next_has_vowel = False
            next_is_hamza = False

            if not same_word and j < len(chars):
                next_is_hamza = chars[j] in HAMZA_SET

                k = j + 2
                while k < len(chars) and chars[k] in DIACRITICS:
                    if chars[k] in [FATHA, DAMMA, KASRA]:
                        next_has_vowel = True
                        break
                    k += 1

            # --- previous vowel ---
            prev_has_vowel = False
            k = char_index - 1
            while k >= 0:
                if chars[k] in DIACRITICS:
                    if chars[k] in [FATHA, DAMMA, KASRA]:
                        prev_has_vowel = True
                        break
                else:
                    break
                k -= 1

            # --- apply silah ---
            if prev_has_vowel and not same_word:
                if next_is_hamza or next_has_vowel:
                    phonemes.append(base + short_vowel * 2)
                    _append_meta(metadata, ch, char_index, True)
                    applied = True

            # ✅ FALLBACK (CRITICAL FIX)
            if not applied:
                phonemes.append(base + short_vowel)
                _append_meta(metadata, ch, char_index, False)

            continue

        # Short vowel on the consonant
        if short_vowel:
            phonemes.append(base + short_vowel)
            _append_meta(metadata, ch, char_index, False)
            last_vowel = short_vowel
            continue

        # Consonant without vowel
        # 🔥 Standalone madd ya (after kasra sound)
        if ch == LONG_YA and last_vowel == "i":
            phonemes.append("yii")
            _append_meta(metadata, ch, char_index, True)
            last_vowel = "ii"
            continue

        phonemes.append(base)
        last_vowel = None

        _append_meta(metadata, ch, char_index, False)

        # -------------------

    # POST-PROCESSING (DEBUG CONTROL)
    # -------------------
    if not for_debug:
        filtered_phonemes = []
        filtered_metadata = []

        for ph, meta in zip(phonemes, metadata):
            if ph in {WASL, SPACE_TOKEN}:
                continue
            filtered_phonemes.append(ph)
            filtered_metadata.append(meta)

        phonemes = filtered_phonemes
        metadata = filtered_metadata
    return phonemes, metadata


def test_encoder():

    test_words = {
        "ثُمَّ اتَّبَعُوا": ["θu", "m", "ma", "<space>", "<wasl>", "t", "ta", "ba", "ʕuu"],
        "وَاسْتَيْقَنَتْهَا": ["wa", "<wasl>", "s", "ta", "y", "qa", "na", "t", "haa"],
        "الَّذِينَ": ["<wasl>", "l", "la", "ðii", "na"],
        "اِضْرِبْ": ["<wasl>", "dˤ", "ri", "b"],
        "اِذْهَبْ": ["<wasl>", "ð", "ha", "b"],
        "اِصْبِرْ": ["<wasl>", "sˤ", "bi", "r"],
        "اهْبِطُوا": ["<wasl>", "h", "bi", "tˤuu"],
        "اُكْتُبْ": ["<wasl>", "k", "tu", "b"],
        "اسْتُهْزِئَ": ["<wasl>", "s", "tu", "h", "zi", "ʔa"],
        "اقْتَتَلُوا": ["<wasl>", "q", "ta", "ta", "luu"],
        "اسْتُضْعِفُوا": ["<wasl>", "s", "tu", "dˤ", "ʕi", "fuu"],
        "فَاتَّقُوا": ["fa", "<wasl>", "t", "ta", "quu"],
        "مَا اسْتَطَعْتُمْ": ["maa", "<space>", "<wasl>", "s", "ta", "tˤa", "ʕ", "tu", "m"],
        "وَأُولَاتُ": ["wa", "ʔu", "laa", "tu"],
        "أُولَاتِ": ["ʔu", "laa", "ti"],
        # "لِأُولِي": ["li", "ʔu", "lii"],
        "فَاسْتَكْبَرُوا": ["fa", "<wasl>", "s", "ta", "k", "ba", "ruu"],
        "وَاسْتَغْفِرْ": ["wa", "<wasl>", "s", "ta", "ɣ", "fi", "r"],
        "بِاسْمِ": ["bi", "<wasl>", "s", "mi"],
        "بِسْمِ": ["bi", "<wasl>", "s", "mi"],
        "لِاسْتَقَامُوا": ["li", "<wasl>", "s", "ta", "qaa", "muu"],
        "فَالْحَقُّ": ["fa", "<wasl>", "l", "ħa", "q", "qu"],
        "كَالْجِبَالِ": ["ka", "<wasl>", "l", "ji", "baa", "li"],
        "هُزُوًا": ["hu", "zu", "wan"],
        "سوى": ["s", "w", "aa"],
        "فَاجْعَلْ": ["fa", "<wasl>", "j", "ʕa", "l"],
        "وَٱسْتَغْفِرْ": ["wa", "<wasl>", "s", "ta", "ɣ", "fi", "r"],
        "فَالْتَقَطَهُ": ["fa", "<wasl>", "l", "ta", "qa", "tˤa", "hu"],
        "جَاءَ": ["jaa", "ʔa"],
        "السَّمَاءِ": ["<wasl>", "s", "sa", "maa", "ʔi"],
        "مِن وَلِيٍّ": ["mi", "n", "<space>", "wa", "li", "y", "yin"],
        "وَالْتَفَّتِ": ["wa", "<wasl>", "l", "ta", "f", "fa", "ti"],
        "الْتَقَى": ["<wasl>", "l", "ta", "qaa"],
        "الْتَقَيْتُمْ": ["<wasl>", "l", "ta", "qa", "y", "tu", "m"],
        "كَالظُّلَلِ": ["ka", "<wasl>", "ðˤ", "ðˤu", "la", "li"],
        "كَالْمُجْرِمِينَ": ["ka", "<wasl>", "l", "mu", "j", "ri", "mii", "na"],
        "كَالْأَنْعَامِ": ["ka", "<wasl>", "l", "ʔa", "n", "ʕaa", "mi"],
        "اطَّلَعْتَ": ["<wasl>", "tˤ", "tˤa", "la", "ʕ", "ta"],
        "الضَّلَالَةَ": ["<wasl>", "dˤ", "dˤa", "laa", "la", "Ta"],
        "وَالَّذِينَ": ["wa", "<wasl>", "l", "la", "ðii", "na"],
        "لِّلَّذِينَ": ["l", "li", "l", "la", "ðii", "na"],
        "وَالنَّاسِ": ["wa", "<wasl>", "n", "naa", "si"],
        "وَالْقَمَرِ": ["wa", "<wasl>", "l", "qa", "ma", "ri"],
        "وَالِدَيَّ": ["waa", "li", "da", "y", "ya"],
        "إِنَّهُ كَانَ": ["ʔi", "n", "na", "huu", "<space>", "kaa", "na"],
        "ٱلصَّافَّات": ["<wasl>", "sˤ", "sˤaa", "f", "faa", "t"],
        "بِابْنِ": ["bi", "<wasl>", "b", "ni"],
        "فِي ٱلْكِتَابِ": ["fii", "<space>", "<wasl>", "l", "ki", "taa", "bi"],
        "بِٱسْمِ ": ["bi", "<wasl>", "s", "mi"],
        "بِسِحْر": ["bi", "si", "ħ", "r"],
        "لِلَّهِ أَندَادًا": ["li", "l", "laa", "hi", "<space>", "ʔa", "n", "daa", "dan"],
        "اللَّٰهِ إِن": ["<wasl>", "l", "laa", "hi", "<space>", "ʔi", "n"],
        "فَأَخْرَجَهُمَا": ["fa", "ʔa", "x", "ra", "ja", "hu", "maa"],
        "فَهِيَ": ["fa", "hi", "ya"],
        "حَيِيَ": ["ħa", "yi", "ya"],
        "قِيْلَ": ["qii", "la"],
        "مَحْيَا": ["ma", "ħ", "yaa"],
        "يَسْتَحْيِي": ["ya", "s", "ta", "ħ", "yi", "yii"],
        "يُحْيِيكُمْ": ["yu", "ħ", "yi", "yii", "ku", "m"],
        "يُحْيِي": ["yu", "ħ", "yi", "yii"],
        "فَأَقَامَهُ ۖ قَالَ": ["fa", "ʔa", "qaa", "ma", "huu", "<space>", "qaa", "la"],
        "يُبْدِلَهُمَا رَبُّهُمَا": [
            "yu",
            "b",
            "di",
            "la",
            "hu",
            "maa",
            "<space>",
            "ra",
            "b",
            "bu",
            "hu",
            "maa",
        ],
        "سُبْحَانَهُ ۚ إِذَا": ["su", "b", "ħaa", "na", "huu", "<space>", "ʔi", "ðaa"],
        "رَحْمَتِهِ ۗ أَإِلَـٰهٌ": [
            "ra",
            "ħ",
            "ma",
            "ti",
            "hii",
            "<space>",
            "ʔa",
            "ʔi",
            "laa",
            "hun",
        ],
        "فَارِضٌ": ["faa", "ri", "dˤun"],
        "لِاسْمِ": ["li", "<wasl>", "s", "mi"],
        "فَاسْتَغْفَرَ": ["fa", "<wasl>", "s", "ta", "ɣ", "fa", "ra"],
        "وَامْرَأَة": ["wa", "<wasl>", "m", "ra", "ʔa", "T"],
        "فَامْرَأَةٌ": ["fa", "<wasl>", "m", "ra", "ʔa", "Tun"],
        "الْقَمَر": ["<wasl>", "l", "qa", "ma", "r"],
        "فَاسِقِينَ": ["faa", "si", "qii", "na"],
        "فَاتَّقُوا": ["fa", "<wasl>", "t", "ta", "quu"],
        "اطَّلَعْتَ": ["<wasl>", "tˤ", "tˤa", "la", "ʕ", "ta"],
        "وَالَّذِينَ": ["wa", "<wasl>", "l", "la", "ðii", "na"],
        "وَامْرَأَتُهُ": ["wa", "<wasl>", "m", "ra", "ʔa", "tu", "hu"],
        "امْرَأَتَانِ": ["<wasl>", "m", "ra", "ʔa", "taa", "ni"],
        "فِيهِ أَجْرٌ": ["fii", "hi", "<space>", "ʔa", "j", "run"],
        "فَارِضٌ": ["faa", "ri", "dˤun"],
        "وَاسِعٌ": ["waa", "si", "ʕun"],
        "فَاسِقِينَ": ["faa", "si", "qii", "na"],
        "إِلًّا": ["ʔi", "l", "lan"],
        "جَاءَهُمُ": ["jaa", "ʔa", "hu", "mu"],
        "وَاقِعٌ": ["waa", "qi", "ʕun"],
        "أَصَابَهُمُ": ["ʔa", "sˤaa", "ba", "hu", "mu"],
        "بِالصَّلَاةِ": ["bi", "<wasl>", "sˤ", "sˤa", "laa", "Ti"],
        "بِالسَّيِّئَةِ": ["bi", "<wasl>", "s", "sa", "y", "yi", "ʔa", "Ti"],
        "بِالصَّبْرِ": ["bi", "<wasl>", "sˤ", "sˤa", "b", "ri"],
        "بِالظَّالِمِينَ": ["bi", "<wasl>", "ðˤ", "ðˤaa", "li", "mii", "na"],
        "بِالرُّسُلِ": ["bi", "<wasl>", "r", "ru", "su", "li"],
        "اللَّيل": ["<wasl>", "l", "la", "y", "l"],
        "بِاللَّٰهِ": ["bi", "<wasl>", "l", "laa", "hi"],
        "اللُّؤْلُؤ": ["<wasl>", "l", "lu", "ʔ", "lu", "ʔ"],
        "لِّلشَّارِبِينَ": ["l", "li", "<wasl>", "ʃ", "ʃaa", "ri", "bii", "na"],
        "لِّلسَّائِلِينَ": ["l", "li", "<wasl>", "s", "saa", "ʔi", "lii", "na"],
        "لِلشَّمْسِ": ["li", "<wasl>", "ʃ", "ʃa", "m", "si"],
        "لِلنَّاسِ": ["li", "<wasl>", "n", "naa", "si"],
        "لِلرُّسُلِ": ["li", "<wasl>", "r", "ru", "su", "li"],
        "لِلرَّحْمَـٰنِ": ["li", "<wasl>", "r", "ra", "ħ", "maa", "ni"],
        "لِّلشَّارِبِينَ": ["l", "li", "<wasl>", "ʃ", "ʃaa", "ri", "bii", "na"],
        "الْأُولَىٰ": ["<wasl>", "l", "ʔuu", "laa"],
    }

    for word, phoneme in test_words.items():

        ph, _ = text_to_phonemes_with_mapping(word)
        if phoneme != ph:
            print("=" * 100)
            print("correct encoding : ")
            print(f"word : {word} => {phoneme}")

            print("current encoding : ")
            print(f"word : {word} => {ph}")
