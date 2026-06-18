"""
Train/Test Splitter for Dwelling Annotations

Rules
-----
1. Splitting is performed at the larva level.
   - A larva is uniquely identified by (source, ID).
   - All annotations from a larva stay together.

2. The optimizer balances annotation tags.
   - Target: ~20% of each tag in the test set.
   - Rare tags (< MIN_TAG_COUNT) are ignored.

3. A registry stores permanent assignments:
       source, ID, split (train/test/blank)

4. Once assigned:
       train -> always train
       test  -> always test

5. New annotations inherit the larva's existing split.

6. Only previously unseen larvae are optimized.
   Existing train/test larvae remain locked.

7. Each run creates:
       annotations_train.csv
       annotations_test.csv
       split_log.txt
       larva_registry_after_split.csv

"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# ==========================================================
# CONFIG
# ==========================================================
INPUT_DIR = Path(r"C:\Users\Tomoko\Desktop\Dwelling_Project\annotation\annotation_csvs_all")
OUTPUT_DIR = Path(r"C:\Users\Tomoko\Desktop\Dwelling_Project\annotation\annotation_csvs")

def split_dataset(INPUT_DIR,OUTPUT_DIR,VAL_SPLIT_REG_DIR,TARGET_TEST_RATIO=0.20,MIN_TAG_COUNT=5,N_ITER=5000,RANDOM_SEED=42,mode="all"):
    if mode == "all":
        SPLIT_DIR = OUTPUT_DIR / "split_data"
    elif mode == "train":
        SPLIT_DIR = OUTPUT_DIR
    REGISTRY_FILE = OUTPUT_DIR / "larva_registry.csv"

    # --- generate folders and output files ---
    from datetime import datetime

    today = datetime.now().strftime("%Y-%m-%d")

    existing_runs = sorted(SPLIT_DIR.glob(f"{today}_*"))

    run_num = len(existing_runs) + 1
    run_name = f"{today}_{run_num:03d}"

    if mode == "all":
        RUN_DIR = SPLIT_DIR / run_name
    elif mode == "train":
        RUN_DIR = SPLIT_DIR
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    TRAIN_DIR = RUN_DIR / "train"
    if mode == "all":
        TEST_DIR= RUN_DIR / "test"
        TEST_DIR.mkdir(exist_ok=True)
        TEST_OUT = TEST_DIR / "annotations_test.csv"
    elif mode == "train":
        TEST_DIR = RUN_DIR / "validation"
        TEST_DIR.mkdir(exist_ok=True)
        TEST_OUT = TEST_DIR / "annotations_validation.csv"
    TRAIN_DIR.mkdir(exist_ok=True)
    TRAIN_OUT = TRAIN_DIR / "annotations_train.csv"
    

    LOG_FILE = RUN_DIR / "split_log.txt"

    if mode == "all":
        REGISTRY_SNAPSHOT = RUN_DIR / "larva_registry_after_split.csv"
    if mode == "train":
        REGISTRY_SNAPSHOT = VAL_SPLIT_REG_DIR / "larva_registry_after_split.csv"

    logging.basicConfig(
        level=logging.INFO,format="%(message)s", handlers=[
            logging.FileHandler(LOG_FILE,mode="w"),
            logging.StreamHandler()
        ]
    )
    log = logging.info

    # ==========================================================
    # HELPER FUNCTIONS
    # ==========================================================

    def split_tags(tag_string):
        """
        Parse semicolon-separated tags.

        Examples
        --------
        'crawl;sharp_turn'
            -> ['crawl', 'sharp_turn']
        """
        if pd.isna(tag_string):
            return []

        tags = [t.strip().lower() for t in str(tag_string).split(";")]
        tags = [t for t in tags if t and t != "consult"]

        return tags


    # ==========================================================
    # LOAD DATA
    # ==========================================================

    if REGISTRY_FILE.exists():
        registry = pd.read_csv(REGISTRY_FILE)
    else:
        registry = pd.DataFrame(
            columns=["source","ID","split"]
        )

    csv_files = list(INPUT_DIR.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {INPUT_DIR}")

    ann = pd.concat(
        [pd.read_csv(f) for f in csv_files],
        ignore_index=True
    )

    log(f"Loaded {len(csv_files)} CSV files")
    log(f"Total rows: {len(ann):,}")


    ann["source"] = ann.get("File", ann.get("source"))
    ann = ann.drop(columns=[c for c in ["File", "source_file"] if c in ann.columns])

    ann = ann[ann["behavior"].isin(["dwelling","nondwelling"])].copy()

    ann["_tag_list"] = ann["tags"].apply(split_tags)

    all_tags = sorted({
        tag
        for tags in ann["_tag_list"]
        for tag in tags
    })

    # ==========================================================
    # COUNT TOTAL OCCURRENCES OF EACH TAG
    # ==========================================================

    global_tag_counts = {}

    for tag in all_tags:
        count = sum(tag in tags for tags in ann["_tag_list"])
        global_tag_counts[tag] = count

    # Remove rare tags from optimization
    all_tags = [tag for tag in all_tags if global_tag_counts[tag] >= MIN_TAG_COUNT]
    log("\nTags used for balancing:")

    for tag in all_tags:
        log(f"  {tag:20s} {global_tag_counts[tag]}")

    # ==========================================================
    # UNIQUE LARVAE
    # ==========================================================

    larvae = (
        ann[["source", "ID"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    current_pairs = set(zip(larvae["source"], larvae["ID"]))
    known_pairs = set(zip(registry["source"], registry["ID"]))
    new_pairs = current_pairs - known_pairs

    if new_pairs:
        new_rows = pd.DataFrame(
            list(new_pairs),
            columns=["source", "ID"]
        )

        new_rows["split"] = np.nan

        registry = pd.concat(
            [registry, new_rows],
            ignore_index=True
        )

    # ==========================================================
    # RANDOM SEARCH FOR BEST SPLIT
    # ==========================================================

    # larvae already assigned in registry

    fixed_train_pairs = set(
        zip(
            registry.loc[registry["split"] == "train", "source"],
            registry.loc[registry["split"] == "train", "ID"]
        )
    )

    fixed_test_pairs = set(
        zip(
            registry.loc[registry["split"] == "test", "source"],
            registry.loc[registry["split"] == "test", "ID"]
        )
    )

    # only optimize split for larvae not already assigned

    if mode=="all":
        new_larvae = larvae[
            ~larvae.apply(
                lambda r:
                (
                    (r["source"], r["ID"]) in fixed_train_pairs
                    or
                    (r["source"], r["ID"]) in fixed_test_pairs
                ),
                axis=1
            )
        ].reset_index(drop=True)
    elif mode == "train":
        new_larvae = larvae

    n_new = len(new_larvae)

    log(f"\nLocked train larvae: {len(fixed_train_pairs)}")
    log(f"Locked test larvae : {len(fixed_test_pairs)}")
    log(f"New larvae         : {n_new}")

    # ----------------------------------------------------------
    # Optimized split - 20:80 test:train ratio among new larvae

    if n_new == 0:
        log("\nNo new larvae found. Reusing existing split.")
        best_test_pairs = fixed_test_pairs
        best_score = 0

    else:
        n_test_larvae = round(TARGET_TEST_RATIO * n_new)
        n_test_larvae = min(n_test_larvae, n_new)
        log(f"Target new test/validation larvae: {n_test_larvae}")

        rng = np.random.default_rng(RANDOM_SEED)

        best_score = np.inf
        best_test_pairs = None

        for _ in range(N_ITER):

            selected_idx = rng.choice(
                n_new,
                size=n_test_larvae,
                replace=False
            )

            candidate_larvae = new_larvae.iloc[selected_idx]

            new_test_pairs = set(
                zip(
                    candidate_larvae["source"],
                    candidate_larvae["ID"]
                )
            )

            if mode == "all":
                # old test larvae stay test forever
                candidate_pairs = (
                    fixed_test_pairs
                    | new_test_pairs
                )

                test_mask = ann.apply(
                    lambda r:
                    (r["source"], r["ID"])
                    in candidate_pairs,
                    axis=1
                )

                test_df = ann[test_mask]
            
            elif mode == "train": 
                candidate_pairs = new_test_pairs
                test_df = ann[ann.apply(
                    lambda r:
                    (r["source"], r["ID"])
                    in new_test_pairs,
                    axis=1
                )]

            score = 0.0

            for tag in all_tags:
                total = global_tag_counts[tag]
                if total == 0:
                    continue

                test_count = sum(tag in tags for tags in test_df["_tag_list"])
                achieved_ratio = test_count / total

                score += (achieved_ratio - TARGET_TEST_RATIO) ** 2

            # also balance overall event count
            event_ratio = len(test_df) / len(ann)

            score += (event_ratio - TARGET_TEST_RATIO) ** 2

            if score < best_score:
                best_score = score
                best_test_pairs = candidate_pairs

    # ==========================================================
    # FINAL SPLIT
    # ==========================================================

    test_mask = ann.apply(
        lambda r:
        (r["source"], r["ID"])
        in best_test_pairs,
        axis=1
    )

    test_df = ann[test_mask].copy()
    train_df = ann[~test_mask].copy()

    # ==========================================================
    # REGISTRY UPDATE & SAVING SPLIT DATASETS
    # ==========================================================
    train_pairs = set(
        zip(
            train_df["source"],
            train_df["ID"]
        )
    )

    train_df.drop(columns=["_tag_list"], inplace=True)
    test_df.drop(columns=["_tag_list"], inplace=True)

    train_df.to_csv(TRAIN_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)

    test_pairs = set(
        zip(
            test_df["source"],
            test_df["ID"]
        )
    )

    train_pairs = set(
        zip(
            train_df["source"],
            train_df["ID"]
        )
    )
    
    if mode == "all":
        registry.loc[
            registry.apply(
                lambda r:
                (r["source"], r["ID"]) in test_pairs,
                axis=1
            ),
            "split"
        ] = "test"

    elif mode == "train":
        registry.loc[
            registry.apply(
                lambda r:
                (r["source"], r["ID"]) in test_pairs,
                axis=1
            ),
            "split"
        ] = "validation"

    registry.loc[
        registry.apply(
            lambda r:
            (r["source"], r["ID"]) in train_pairs,
            axis=1
        ),
        "split"
    ] = "train"

    registry.to_csv(
            REGISTRY_SNAPSHOT,
            index=False
        )
    
    registry.to_csv(
            REGISTRY_FILE,
            index=False
        )    
    
    print(f"registry snapshot saved -> {REGISTRY_SNAPSHOT}")

    # ----------------------------------------------------------
    # REPORT LARVA COUNTS
    # ----------------------------------------------------------

    log("\n" + "=" * 70)
    log("SPLIT SUMMARY")
    log("=" * 70)

    n_train_larvae = (
        train_df[["source", "ID"]]
        .drop_duplicates()
        .shape[0]
    )

    n_test_larvae = (
        test_df[["source", "ID"]]
        .drop_duplicates()
        .shape[0]
    )

    log(f"\nTrain larvae: {n_train_larvae}")
    log(f"Test larvae : {n_test_larvae}")

    log(f"\nTrain rows: {len(train_df):,}")
    log(f"Test rows : {len(test_df):,}")

    # ----------------------------------------------------------
    # Larvae per source
    # ----------------------------------------------------------

    log("\nLarvae per source:")

    train_by_source = (
        train_df[["source", "ID"]]
        .drop_duplicates()
        .groupby("source")
        .size()
    )

    test_by_source = (
        test_df[["source", "ID"]]
        .drop_duplicates()
        .groupby("source")
        .size()
    )

    all_sources = sorted(
        set(train_by_source.index)
        | set(test_by_source.index)
    )

    for src in all_sources:

        tr = train_by_source.get(src, 0)
        te = test_by_source.get(src, 0)

        pct = 100 * te / (tr + te)

        log(
            f"{src:10s}"
            f" train={tr:4d}"
            f" test={te:4d}"
            f" test%={pct:6.2f}"
        )

    # ----------------------------------------------------------
    # Tag distribution
    # ----------------------------------------------------------

    log("\nTag distribution:")

    for tag in all_tags:

        train_count = sum(
            tag in split_tags(x)
            for x in train_df["tags"]
        )

        test_count = sum(
            tag in split_tags(x)
            for x in test_df["tags"]
        )

        total = train_count + test_count

        if total == 0:
            continue

        test_pct = 100 * test_count / total

        log(
            f"{tag:20s}"
            f" train={train_count:5d}"
            f" test={test_count:5d}"
            f" test%={test_pct:6.2f}"
        )

    log(f"\nBest score: {best_score:.6f}")

    log("\nSaved:")
    log(TRAIN_OUT)
    log(TEST_OUT)


# ------------------- main-------------------------
# generates test and train datasets from all
# split_dataset(INPUT_DIR,OUTPUT_DIR, mode = "all")
