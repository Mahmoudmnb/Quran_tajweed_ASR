from phoneme_encoder import *


def compute_avg_short_duration(segments):

    short_durations = [frames for pid, frames in segments if pid in SHORT_IDS]

    if len(short_durations) == 0:
        return None

    return sum(short_durations) / len(short_durations)


def compute_observed_harakat(segments):

    avg_short = compute_avg_short_duration(segments)

    if avg_short is None:
        return []

    observed = []

    for index, (pid, frames) in enumerate(segments):

        if pid in LONG_IDS:

            harakat = frames / avg_short

            observed.append(
                {
                    "segment_index": index,
                    "phoneme_id": pid,
                    "frames": frames,
                    "observed_harakat": harakat,
                }
            )

    return observed


def find_madd_positions(text):

    positions = []
    i = 0

    while i < len(text) - 1:

        # Fatha + Alif
        if text[i] == FATHA and text[i + 1] == "ا":
            positions.append(i + 1)

        # Kasra + Ya
        if text[i] == KASRA and text[i + 1] == "ي":
            positions.append(i + 1)

        # Damma + Waw
        if text[i] == DAMMA and text[i + 1] == "و":
            positions.append(i + 1)

        i += 1

    return positions


def get_next_significant_char(text, index):

    i = index + 1

    while i < len(text):

        if text[i] == " ":
            return "SPACE"

        if text[i] not in DIACRITICS:
            return text[i]

        i += 1

    return None


def extract_madd_rules(text):

    text = normalize_arabic(text)

    phoneme_sequence, phoneme_metadata = text_to_phonemes_with_mapping(text)

    rules = []

    for idx, meta in enumerate(phoneme_metadata):

        if not meta["is_long_vowel"]:
            continue

        char_pos = meta["char_index"]

        next_char = get_next_significant_char(text, char_pos)

        # ----------------------------
        # Madd Muttasil
        # ----------------------------
        if next_char in HAMZA_SET:

            madd_type = "muttasil"
            expected = 5

        # ----------------------------
        # Word boundary (possible Munfasil)
        # ----------------------------
        elif next_char == "SPACE":

            # Look at first char of next word
            after_space = get_next_significant_char(text, char_pos + 1)

            if after_space in HAMZA_SET:
                madd_type = "munfasil"
                expected = 5
            else:
                madd_type = "tabeei"
                expected = 2

        # ----------------------------
        # Madd Lazim
        # ----------------------------
        elif next_char == SUKOON or next_char == SHADDA:

            madd_type = "lazim"
            expected = 6

        # ----------------------------
        # Default Madd Tabeei
        # ----------------------------
        else:
            madd_type = "tabeei"
            expected = 2

        rules.append(
            {"phoneme_index": idx, "type": madd_type, "expected_harakat": expected}
        )

    return phoneme_sequence, phoneme_metadata, rules
