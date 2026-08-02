import random

def verify_split(
    train_df,
    val_df,
):
    print()
    print("Verifying dataset split...")

    train_reciters = set(
        train_df["reciter_name"].unique()
    )

    val_reciters = set(
        val_df["reciter_name"].unique()
    )

    common_reciters = (
        train_reciters &
        val_reciters
    )

    if len(common_reciters) > 0:
        raise RuntimeError(
            "Reciter leakage detected:\n"
            f"{sorted(common_reciters)}"
        )

    train_ayahs = set(
        zip(
            train_df["surah"],
            train_df["ayah"],
        )
    )

    val_ayahs = set(
        zip(
            val_df["surah"],
            val_df["ayah"],
        )
    )

    common_ayahs = (
        train_ayahs &
        val_ayahs
    )

    if len(common_ayahs) > 0:
        raise RuntimeError(
            f"Ayah leakage detected.\n"
            f"Number of overlapping ayahs: "
            f"{len(common_ayahs)}"
        )

    duplicated_train = train_df.duplicated().sum()

    duplicated_val = val_df.duplicated().sum()

    if duplicated_train > 0:
        raise RuntimeError(
            f"Training set contains "
            f"{duplicated_train} duplicated rows."
        )

    if duplicated_val > 0:
        raise RuntimeError(
            f"Validation set contains "
            f"{duplicated_val} duplicated rows."
        )

    if len(train_df) == 0:
        raise RuntimeError(
            "Training dataframe is empty."
        )

    if len(val_df) == 0:
        raise RuntimeError(
            "Validation dataframe is empty."
        )

    print("✓ No reciter leakage")

    print("✓ No ayah leakage")

    print("✓ No duplicated rows")

    print("✓ Split verification passed")

def split_dataset(
    df,
    train_reciters=6,
    val_reciters=2,
    train_ayah_ratio=0.75,
    seed=42,
):
    rng = random.Random(seed)

    df = df.copy()

    required_columns = [
        "surah",
        "ayah",
        "reciter_name",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    reciters = sorted(
        df["reciter_name"].unique().tolist()
    )

    if len(reciters) != train_reciters + val_reciters:
        raise ValueError(
            f"Expected {train_reciters + val_reciters} reciters "
            f"but found {len(reciters)}"
        )

    rng.shuffle(reciters)

    val_reciter_names = sorted(
        reciters[:val_reciters]
    )

    train_reciter_names = sorted(
        reciters[val_reciters:]
    )

    df["ayah_id"] = list(
        zip(
            df["surah"],
            df["ayah"],
        )
    )

    unique_ayahs = sorted(
        df["ayah_id"].unique().tolist()
    )

    rng.shuffle(unique_ayahs)

    train_count = int(
        len(unique_ayahs) * train_ayah_ratio
    )

    train_ayahs = set(
        unique_ayahs[:train_count]
    )

    val_ayahs = set(
        unique_ayahs[train_count:]
    )

    train_df = df[
        df["reciter_name"].isin(
            train_reciter_names
        )
        &
        df["ayah_id"].isin(
            train_ayahs
        )
    ].copy()

    val_df = df[
        df["reciter_name"].isin(
            val_reciter_names
        )
        &
        df["ayah_id"].isin(
            val_ayahs
        )
    ].copy()

    train_df.drop(
        columns=["ayah_id"],
        inplace=True,
    )

    val_df.drop(
        columns=["ayah_id"],
        inplace=True,
    )

    print("=" * 60)

    print("Training reciters:")
    for name in train_reciter_names:
        print(" ", name)

    print()

    print("Validation reciters:")
    for name in val_reciter_names:
        print(" ", name)

    print()

    print(
        "Training samples:",
        len(train_df),
    )

    print(
        "Validation samples:",
        len(val_df),
    )

    print()

    print(
        "Training unique ayahs:",
        train_df[
            ["surah", "ayah"]
        ].drop_duplicates().shape[0],
    )

    print(
        "Validation unique ayahs:",
        val_df[
            ["surah", "ayah"]
        ].drop_duplicates().shape[0],
    )

    print("=" * 60)

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        train_reciter_names,
        val_reciter_names,
    )