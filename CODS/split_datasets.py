import pandas as pd
import numpy as np
from itertools import combinations


def split_quran_dataset(
    df,
    test_hours=20,
    val_hours=40,
    max_test_reciters=2,
    excluded_test_reciters=None,
    seed=42,
    verbose=True
):
    """
    Automatically find the best Quran dataset split.

    TEST:
        - approximately test_hours
        - test reciters must NOT exist in Train
        - test ayahs must NOT exist in Train

    VALIDATION:
        - approximately val_hours
        - uses the SAME held-out reciters as Test
        - validation ayahs MUST exist in Train

    TRAIN:
        - contains all remaining usable data

    Optimization priority:
        1. Minimize number of unique ayahs removed from Train
        2. Minimize unused audio
        3. Minimize Train audio removed because of Test ayahs
        4. Keep Test duration close to target

    Parameters
    ----------
    df : pandas.DataFrame

    test_hours : float
        Target Test duration.

    val_hours : float
        Target Validation duration.

    max_test_reciters : int
        Maximum number of held-out reciters to try.
        Usually 2 is a good choice.

    excluded_test_reciters : list
        Reciter names that should NEVER be selected
        as Test/Validation reciters.

    seed : int
        Random seed.

    verbose : bool
        Print detailed results.

    Returns
    -------
    train_df
    val_df
    test_df
    summary
    """

    # =========================================================
    # Prepare
    # =========================================================

    df = df.copy()

    if excluded_test_reciters is None:
        excluded_test_reciters = []

    excluded_test_reciters = set(
        excluded_test_reciters
    )

    # Unique Quran ayah identifier
    # Important because ayah numbers repeat between Surahs.
    df["ayah_id"] = list(
        zip(
            df["surah"],
            df["ayah"]
        )
    )

    target_test_seconds = test_hours * 3600
    target_val_seconds = val_hours * 3600

    total_dataset_seconds = df["duration"].sum()

    # =========================================================
    # Available reciters for Test selection
    # =========================================================

    all_reciters = [
        reciter
        for reciter in df["reciter_name"].unique()
        if reciter not in excluded_test_reciters
    ]

    if verbose:

        print("=" * 70)
        print("QURAN DATASET SPLIT OPTIMIZER")
        print("=" * 70)

        print(
            f"\nTotal dataset: "
            f"{total_dataset_seconds / 3600:.2f} hours"
        )

        print(
            "Total reciters:",
            df["reciter_name"].nunique()
        )

        print(
            "Available Test reciters:",
            len(all_reciters)
        )

        if excluded_test_reciters:

            print("\nExcluded Test reciters:")

            for r in sorted(excluded_test_reciters):
                print("  -", r)

    # =========================================================
    # Store all valid candidate splits
    # =========================================================

    candidates = []

    # =========================================================
    # Try different numbers of Test reciters
    # =========================================================

    for number_of_reciters in range(
        1,
        max_test_reciters + 1
    ):

        if verbose:
            print(
                f"\nTrying combinations of "
                f"{number_of_reciters} reciter(s)..."
            )

        # -----------------------------------------------------
        # Try every possible reciter combination
        # -----------------------------------------------------

        for reciter_tuple in combinations(
            all_reciters,
            number_of_reciters
        ):

            reciter_group = set(
                reciter_tuple
            )

            # =================================================
            # Held-out data
            # =================================================

            heldout = df[
                df["reciter_name"].isin(
                    reciter_group
                )
            ].copy()

            # All other reciters
            normal = df[
                ~df["reciter_name"].isin(
                    reciter_group
                )
            ].copy()

            heldout_seconds = (
                heldout["duration"].sum()
            )

            # -------------------------------------------------
            # Must have enough audio for Test + Validation
            # -------------------------------------------------

            if heldout_seconds < (
                target_test_seconds
                + target_val_seconds
            ):
                continue

            # =================================================
            # Calculate held-out duration for each ayah
            # =================================================

            heldout_stats = (
                heldout
                .groupby("ayah_id")
                .agg(
                    heldout_duration=(
                        "duration",
                        "sum"
                    ),
                    heldout_samples=(
                        "duration",
                        "size"
                    )
                )
            )

            # =================================================
            # Calculate Train duration for each ayah
            # =================================================

            normal_stats = (
                normal
                .groupby("ayah_id")
                .agg(
                    train_duration=(
                        "duration",
                        "sum"
                    ),
                    train_samples=(
                        "duration",
                        "size"
                    )
                )
            )

            stats = heldout_stats.join(
                normal_stats,
                how="left"
            )

            stats[
                [
                    "train_duration",
                    "train_samples"
                ]
            ] = (
                stats[
                    [
                        "train_duration",
                        "train_samples"
                    ]
                ]
                .fillna(0)
            )

            # =================================================
            # Validation availability
            # =================================================
            #
            # Validation ayah must exist in Train.
            #
            # Therefore only held-out ayahs where:
            #
            # train_duration > 0
            #
            # can be used for Validation.
            # =================================================

            initial_val_seconds = (
                stats.loc[
                    stats["train_duration"] > 0,
                    "heldout_duration"
                ]
                .sum()
            )

            if (
                initial_val_seconds
                < target_val_seconds
            ):
                continue

            # =================================================
            # Choose TEST ayahs
            # =================================================

            selected_test_ayahs = []

            current_test_seconds = 0

            remaining_val_seconds = (
                initial_val_seconds
            )

            # =================================================
            # STEP 1:
            # Ayahs that already DON'T exist in Train
            # =================================================
            #
            # These are perfect.
            #
            # They add Test data without deleting
            # anything from Train.
            # =================================================

            free_ayahs = stats[
                stats["train_duration"] == 0
            ].copy()

            # Prefer largest duration first:
            # fewer unique Test ayahs required.
            free_ayahs = free_ayahs.sort_values(
                "heldout_duration",
                ascending=False
            )

            for ayah_id, row in free_ayahs.iterrows():

                if (
                    current_test_seconds
                    >= target_test_seconds
                ):
                    break

                selected_test_ayahs.append(
                    ayah_id
                )

                current_test_seconds += (
                    row["heldout_duration"]
                )

            # =================================================
            # STEP 2:
            # Need additional Test ayahs from ones
            # currently existing in Train.
            # =================================================

            if (
                current_test_seconds
                < target_test_seconds
            ):

                costly_ayahs = stats[
                    stats["train_duration"] > 0
                ].copy()

                # ------------------------------------------------
                # How much Train audio is sacrificed
                # per Test second obtained?
                # ------------------------------------------------

                costly_ayahs["loss_ratio"] = (
                    costly_ayahs["train_duration"]
                    /
                    costly_ayahs["heldout_duration"]
                )

                # ------------------------------------------------
                # Our PRIMARY target:
                #
                # minimize number of unique ayahs removed.
                #
                # Therefore large Test duration per ayah
                # is especially useful.
                #
                # Secondary:
                # minimize Train loss.
                # ------------------------------------------------

                costly_ayahs = (
                    costly_ayahs.sort_values(
                        [
                            "heldout_duration",
                            "loss_ratio",
                            "train_duration"
                        ],
                        ascending=[
                            False,
                            True,
                            True
                        ]
                    )
                )

                for ayah_id, row in costly_ayahs.iterrows():

                    if (
                        current_test_seconds
                        >= target_test_seconds
                    ):
                        break

                    # --------------------------------------------
                    # If this ayah becomes Test,
                    # it cannot also be Validation.
                    # --------------------------------------------

                    possible_val_seconds = (
                        remaining_val_seconds
                        - row["heldout_duration"]
                    )

                    # Protect Validation target
                    if (
                        possible_val_seconds
                        < target_val_seconds
                    ):
                        continue

                    selected_test_ayahs.append(
                        ayah_id
                    )

                    current_test_seconds += (
                        row["heldout_duration"]
                    )

                    remaining_val_seconds = (
                        possible_val_seconds
                    )

            # =================================================
            # Failed to reach Test duration
            # =================================================

            if (
                current_test_seconds
                < target_test_seconds
            ):
                continue

            selected_test_ayahs = set(
                selected_test_ayahs
            )

            # =================================================
            # Create TEST
            # =================================================

            test_df = heldout[
                heldout["ayah_id"].isin(
                    selected_test_ayahs
                )
            ].copy()

            # =================================================
            # Make sure all selected reciters really occur
            # in Test.
            # =================================================

            actual_test_reciters = set(
                test_df[
                    "reciter_name"
                ].unique()
            )

            if (
                actual_test_reciters
                != reciter_group
            ):
                continue

            # =================================================
            # Create TRAIN
            # =================================================
            #
            # Train cannot contain:
            #
            # 1. Held-out reciters
            # 2. Test ayahs
            #
            # =================================================

            train_df = normal[
                ~normal["ayah_id"].isin(
                    selected_test_ayahs
                )
            ].copy()

            train_ayahs = set(
                train_df[
                    "ayah_id"
                ].unique()
            )

            # =================================================
            # Validation candidates
            # =================================================
            #
            # Same held-out reciters.
            #
            # But ayah MUST still exist in Train.
            #
            # =================================================

            val_candidates = heldout[
                (
                    ~heldout["ayah_id"].isin(
                        selected_test_ayahs
                    )
                )
                &
                (
                    heldout["ayah_id"].isin(
                        train_ayahs
                    )
                )
            ].copy()

            available_val_seconds = (
                val_candidates[
                    "duration"
                ].sum()
            )

            if (
                available_val_seconds
                < target_val_seconds
            ):
                continue

            # =================================================
            # Build Validation close to requested hours
            # =================================================

            val_candidates = (
                val_candidates.sample(
                    frac=1,
                    random_state=seed
                )
            )

            cumulative = (
                val_candidates[
                    "duration"
                ].cumsum()
            )

            mask = (
                cumulative
                <= target_val_seconds
            )

            val_df = (
                val_candidates[
                    mask
                ].copy()
            )

            # -------------------------------------------------
            # Maybe add the next sample if it gets closer
            # to exactly val_hours.
            # -------------------------------------------------

            if (
                len(val_df)
                < len(val_candidates)
            ):

                next_position = len(
                    val_df
                )

                next_row = (
                    val_candidates.iloc[
                        [next_position]
                    ]
                )

                current_seconds = (
                    val_df[
                        "duration"
                    ].sum()
                )

                next_seconds = (
                    next_row[
                        "duration"
                    ].sum()
                )

                current_error = abs(
                    target_val_seconds
                    - current_seconds
                )

                new_error = abs(
                    target_val_seconds
                    - (
                        current_seconds
                        + next_seconds
                    )
                )

                if (
                    new_error
                    < current_error
                ):
                    val_df = pd.concat(
                        [
                            val_df,
                            next_row
                        ]
                    )

            # =================================================
            # Calculate losses
            # =================================================

            train_seconds = (
                train_df[
                    "duration"
                ].sum()
            )

            val_seconds = (
                val_df[
                    "duration"
                ].sum()
            )

            test_seconds = (
                test_df[
                    "duration"
                ].sum()
            )

            used_seconds = (
                train_seconds
                + val_seconds
                + test_seconds
            )

            unused_seconds = (
                total_dataset_seconds
                - used_seconds
            )

            # =================================================
            # Unique Train ayahs lost because they
            # became Test ayahs
            # =================================================

            normal_ayahs_before = set(
                normal[
                    "ayah_id"
                ].unique()
            )

            train_ayahs_after = set(
                train_df[
                    "ayah_id"
                ].unique()
            )

            lost_train_ayahs = (
                normal_ayahs_before
                - train_ayahs_after
            )

            # =================================================
            # Train AUDIO lost because those ayahs
            # became Test ayahs
            # =================================================

            removed_train_audio = (
                normal[
                    normal["ayah_id"].isin(
                        selected_test_ayahs
                    )
                ]["duration"]
                .sum()
            )

            # =================================================
            # Data lost because held-out reciters were
            # not used by Test or Validation
            # =================================================

            used_heldout_indices = set(
                test_df.index
            ) | set(
                val_df.index
            )

            unused_heldout = heldout[
                ~heldout.index.isin(
                    used_heldout_indices
                )
            ]

            unused_heldout_seconds = (
                unused_heldout[
                    "duration"
                ].sum()
            )

            # =================================================
            # Save candidate
            # =================================================

            candidate = {

                "reciters":
                    tuple(
                        sorted(
                            reciter_group
                        )
                    ),

                "test_hours":
                    test_seconds / 3600,

                "val_hours":
                    val_seconds / 3600,

                "train_hours":
                    train_seconds / 3600,

                "total_used_hours":
                    used_seconds / 3600,

                "unused_hours":
                    unused_seconds / 3600,

                "unused_heldout_hours":
                    unused_heldout_seconds / 3600,

                "num_test_ayahs":
                    len(
                        selected_test_ayahs
                    ),

                "lost_train_ayahs":
                    len(
                        lost_train_ayahs
                    ),

                "train_audio_removed_hours":
                    removed_train_audio / 3600,

                "selected_test_ayahs":
                    selected_test_ayahs,

                "train_df":
                    train_df,

                "val_df":
                    val_df,

                "test_df":
                    test_df,
            }

            candidates.append(
                candidate
            )

    # =========================================================
    # No valid split found
    # =========================================================

    if not candidates:

        raise ValueError(
            "\nNo valid split was found.\n\n"
            "Possible solutions:\n"
            "- reduce val_hours\n"
            "- reduce test_hours\n"
            "- increase max_test_reciters\n"
            "- remove some names from excluded_test_reciters"
        )

    # =========================================================
    # Rank candidates
    # =========================================================
    #
    # IMPORTANT:
    #
    # Priority #1:
    # minimum unique Train ayahs lost.
    #
    # Priority #2:
    # minimum total unused audio.
    #
    # Priority #3:
    # minimum Train audio removed.
    #
    # Priority #4:
    # Test hours closest to requested target.
    #
    # =========================================================

    candidates.sort(
        key=lambda x: (
            x["lost_train_ayahs"],
            x["unused_hours"],
            x["train_audio_removed_hours"],
            abs(
                x["test_hours"]
                - test_hours
            )
        )
    )

    best = candidates[0]

    # =========================================================
    # Extract final DataFrames
    # =========================================================

    train_df = (
        best["train_df"]
        .drop(
            columns=["ayah_id"]
        )
        .reset_index(
            drop=True
        )
    )

    val_df = (
        best["val_df"]
        .drop(
            columns=["ayah_id"]
        )
        .reset_index(
            drop=True
        )
    )

    test_df = (
        best["test_df"]
        .drop(
            columns=["ayah_id"]
        )
        .reset_index(
            drop=True
        )
    )

    # =========================================================
    # FINAL VERIFICATION
    # =========================================================

    train_reciters = set(
        train_df[
            "reciter_name"
        ].unique()
    )

    val_reciters = set(
        val_df[
            "reciter_name"
        ].unique()
    )

    test_reciters = set(
        test_df[
            "reciter_name"
        ].unique()
    )

    train_ayahs_check = set(
        zip(
            train_df["surah"],
            train_df["ayah"]
        )
    )

    val_ayahs_check = set(
        zip(
            val_df["surah"],
            val_df["ayah"]
        )
    )

    test_ayahs_check = set(
        zip(
            test_df["surah"],
            test_df["ayah"]
        )
    )

    # ---------------------------------------------------------
    # Test reciters completely unseen during training
    # ---------------------------------------------------------

    assert train_reciters.isdisjoint(
        test_reciters
    ), (
        "ERROR: Test reciter exists in Train!"
    )

    # ---------------------------------------------------------
    # Validation uses held-out reciters
    # ---------------------------------------------------------

    assert val_reciters.issubset(
        test_reciters
    ), (
        "ERROR: Validation contains "
        "non-Test reciter!"
    )

    # ---------------------------------------------------------
    # Test ayahs completely unseen during training
    # ---------------------------------------------------------

    assert train_ayahs_check.isdisjoint(
        test_ayahs_check
    ), (
        "ERROR: Test ayah exists in Train!"
    )

    # ---------------------------------------------------------
    # Validation ayahs must be known during training
    # ---------------------------------------------------------

    assert val_ayahs_check.issubset(
        train_ayahs_check
    ), (
        "ERROR: Validation contains "
        "unseen Train ayah!"
    )

    # =========================================================
    # Print best result
    # =========================================================

    if verbose:

        print("\n")
        print("=" * 70)
        print("BEST DATASET SPLIT FOUND")
        print("=" * 70)

        print(
            "\nSelected Test reciter(s):"
        )

        for reciter in best[
            "reciters"
        ]:

            print(
                "  -",
                reciter
            )

        print(
            "\n---------------- DATASET ----------------"
        )

        print(
            f"Train       : "
            f"{len(train_df):,} samples"
            f" | "
            f"{best['train_hours']:.2f} hours"
        )

        print(
            f"Validation  : "
            f"{len(val_df):,} samples"
            f" | "
            f"{best['val_hours']:.2f} hours"
        )

        print(
            f"Test        : "
            f"{len(test_df):,} samples"
            f" | "
            f"{best['test_hours']:.2f} hours"
        )

        print(
            "\n-------------- OPTIMIZATION --------------"
        )

        print(
            "Test unique ayahs:",
            best[
                "num_test_ayahs"
            ]
        )

        print(
            "Unique ayahs removed from Train:",
            best[
                "lost_train_ayahs"
            ]
        )

        print(
            "Train audio removed because "
            "of Test ayahs:",
            f"{best['train_audio_removed_hours']:.2f} hours"
        )

        print(
            "Unused held-out reciter audio:",
            f"{best['unused_heldout_hours']:.2f} hours"
        )

        print(
            "Total unused dataset audio:",
            f"{best['unused_hours']:.2f} hours"
        )

        print(
            "\n--------------- CHECKS ----------------"
        )

        print(
            "Test reciters unseen in Train:",
            train_reciters.isdisjoint(
                test_reciters
            )
        )

        print(
            "Test ayahs unseen in Train:",
            train_ayahs_check.isdisjoint(
                test_ayahs_check
            )
        )

        print(
            "Validation ayahs exist in Train:",
            val_ayahs_check.issubset(
                train_ayahs_check
            )
        )

        print(
            "Validation reciters are "
            "Test reciters:",
            val_reciters.issubset(
                test_reciters
            )
        )

        print("=" * 70)

    # =========================================================
    # Summary without DataFrames
    # =========================================================

    summary = {

        "test_reciters":
            best["reciters"],

        "train_hours":
            best["train_hours"],

        "val_hours":
            best["val_hours"],

        "test_hours":
            best["test_hours"],

        "test_unique_ayahs":
            best["num_test_ayahs"],

        "unique_train_ayahs_lost":
            best["lost_train_ayahs"],

        "train_audio_removed_hours":
            best[
                "train_audio_removed_hours"
            ],

        "unused_heldout_hours":
            best[
                "unused_heldout_hours"
            ],

        "unused_hours":
            best["unused_hours"],

        "number_of_candidates_tested":
            len(candidates)
    }

    return (
        train_df,
        val_df,
        test_df,
        summary
    )