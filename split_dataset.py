"""
Generate larva-level train/test annotation splits.

This script:
    1. Loads all annotation CSVs from all_annotations/
    2. Preserves every annotation row exactly
    3. Splits by larva (source + ID)
    4. Attempts to maintain an ~80/20 train/test ratio
       across behavioral tags
    5. Saves annotations_train.csv and annotations_test.csv

Notes
-----
- Larvae are never split across train and test.
- Multiple tags per row are supported.
- 'consult' is ignored when balancing tags.
- Rare tags (< MIN_TAG_COUNT occurrences) are excluded
  from the balancing objective.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# ==========================================================
# CONFIG
# ==========================================================

INPUT_DIR = Path(
    r"C:\Users\Tomoko\Desktop\Dwelling_Project\annotation\annotation_csvs\all_annotations"
)

OUTPUT_DIR = Path(
    r"C:\Users\Tomoko\Desktop\Dwelling_Project\annotation\annotation_csvs"
)

TRAIN_OUT = OUTPUT_DIR / "annotations_train.csv"
TEST_OUT = OUTPUT_DIR / "annotations_test.csv"

LOG_FILE = OUTPUT_DIR / "split_log.txt"

logging.basicConfig(
    level=logging.INFO,format="%(message)s", handlers=[
        logging.FileHandler(LOG_FILE,mode="w"),
        logging.StreamHandler()
    ]
)

log = logging.info

TARGET_TEST_RATIO = 0.20
MIN_TAG_COUNT = 5
N_ITER = 5000
RANDOM_SEED = 42

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

    'wonderful;consult'
        -> ['wonderful']
    """
    if pd.isna(tag_string):
        return []

    tags = [
        t.strip().lower()
        for t in str(tag_string).split(";")
    ]

    tags = [
        t for t in tags
        if t and t != "consult"
    ]

    return tags


# ==========================================================
# LOAD DATA
# ==========================================================

csv_files = list(INPUT_DIR.glob("*.csv"))

if not csv_files:
    raise ValueError(f"No CSV files found in {INPUT_DIR}")

ann = pd.concat(
    [pd.read_csv(f) for f in csv_files],
    ignore_index=True
)

log(f"Loaded {len(csv_files)} CSV files")
log(f"Total rows: {len(ann):,}")


ann = ann.rename(columns={
    "File": "source",
    "source_file": "source"
})

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

    count = sum(
        tag in tags
        for tags in ann["_tag_list"]
    )

    global_tag_counts[tag] = count

# Remove rare tags from optimization
all_tags = [
    tag
    for tag in all_tags
    if global_tag_counts[tag] >= MIN_TAG_COUNT
]

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

n_larvae = len(larvae)
n_test_larvae = max(1, round(TARGET_TEST_RATIO * n_larvae))

log(f"\nTotal larvae: {n_larvae}")
log(f"Target test larvae: {n_test_larvae}")

# ==========================================================
# RANDOM SEARCH FOR BEST SPLIT
# ==========================================================

rng = np.random.default_rng(RANDOM_SEED)

best_score = np.inf
best_test_pairs = None

for _ in range(N_ITER):

    selected_idx = rng.choice(
        n_larvae,
        size=n_test_larvae,
        replace=False
    )

    candidate_larvae = larvae.iloc[selected_idx]

    candidate_pairs = set(
        zip(
            candidate_larvae["source"],
            candidate_larvae["ID"]
        )
    )

    test_mask = ann.apply(
        lambda r: (r["source"], r["ID"]) in candidate_pairs,
        axis=1
    )

    test_df = ann[test_mask]

    score = 0.0

    for tag in all_tags:

        total = global_tag_counts[tag]

        if total == 0:
            continue

        test_count = sum(
            tag in tags
            for tags in test_df["_tag_list"]
        )

        achieved_ratio = test_count / total

        score += (
            achieved_ratio - TARGET_TEST_RATIO
        ) ** 2

    if score < best_score:
        best_score = score
        best_test_pairs = candidate_pairs

# ==========================================================
# FINAL SPLIT
# ==========================================================

test_mask = ann.apply(
    lambda r: (r["source"], r["ID"]) in best_test_pairs,
    axis=1
)

test_df = ann[test_mask].copy()
train_df = ann[~test_mask].copy()

# Remove helper column before saving
train_df.drop(columns=["_tag_list"], inplace=True)
test_df.drop(columns=["_tag_list"], inplace=True)

# ==========================================================
# SAVE
# ==========================================================

train_df.to_csv(TRAIN_OUT, index=False)
test_df.to_csv(TEST_OUT, index=False)

# ==========================================================
# REPORT
# ==========================================================

log("\n" + "=" * 70)
log("SPLIT SUMMARY")
log("=" * 70)

# ----------------------------------------------------------
# Larvae counts
# ----------------------------------------------------------

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