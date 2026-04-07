import importlib
import sys

import phoneme_encoder

importlib.reload(phoneme_encoder)

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
INPUT_FILE = "quran-simple.txt"  # your Quran text file
OUTPUT_FILE = "quran_phonemes.txt"


# -------------------------------------------------
# MAIN PROCESS
# -------------------------------------------------
def generate_phoneme_dataset():

    total_lines = 0
    processed_lines = 0
    error_lines = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, open(
        OUTPUT_FILE, "w", encoding="utf-8"
    ) as outfile:

        for line in infile:

            total_lines += 1
            line = line.strip()

            if not line:
                continue

            try:
                # ---------------------------------
                # Extract Quran text
                # Format: surah|ayah|text
                # ---------------------------------
                parts = line.split("|", 2)

                if len(parts) != 3:
                    continue

                text = parts[2].strip()

                if not text:
                    continue

                # ---------------------------------
                # Run encoder
                # ---------------------------------
                phonemes, metadata = phoneme_encoder.text_to_phonemes_with_mapping(text)

                # ---------------------------------
                # Convert phoneme list to string
                # ---------------------------------
                phoneme_string = " ".join(phonemes)

                # ---------------------------------
                # Save result
                # ---------------------------------
                outfile.write(f"{text} || {phoneme_string}\n")

                processed_lines += 1

            except Exception as e:
                error_lines += 1
                print(f"Error on line {total_lines}: {e}", file=sys.stderr)
                continue

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------
    print("Processing complete")
    print("Total lines:", total_lines)
    print("Processed:", processed_lines)
    print("Errors:", error_lines)
    print("Output saved to:", OUTPUT_FILE)


def split_ds():
    # Read the uploaded file
    input_path = "quran_phonemes.txt"

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Split into 6 roughly equal parts
    n = len(lines)
    parts = 6
    chunk_size = (n + parts - 1) // parts  # ceiling division

    names = [
        "Mahmoud Bannan",
        "Mohammad Deep Jalab",
        "Amjad Haj Hammada",
        "Mohannad Abdullah",
        "Haidar Kasem",
        "Mahmoud Abd Alraheem",
    ]

    file_paths = []

    for i, name in enumerate(names):
        chunk = lines[i * chunk_size : (i + 1) * chunk_size]
        out_path = f"{name}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(chunk)
        file_paths.append(out_path)

    file_paths
