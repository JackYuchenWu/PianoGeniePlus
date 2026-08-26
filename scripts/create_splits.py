from pathlib import Path
import json
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.pop909 import discover_songs


DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "POP909"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

SPLIT_PATH = OUTPUT_DIR / "pop909_splits.json"

SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SEQUENCE_LENGTH = 128


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover songs

    songs = discover_songs(DATA_ROOT)

    song_ids = [song["song_id"] for song in songs]

    if len(song_ids) != len(set(song_ids)):
        raise ValueError("Duplicate song IDs found.")

    print(f"Found {len(song_ids)} songs.")

    # Validate split ratios

    if abs(
        TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0
    ) > 1e-8:
        raise ValueError(
            "TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1."
        )

    # Reproducible shuffle

    rng = random.Random(SEED)

    shuffled_ids = song_ids.copy()
    rng.shuffle(shuffled_ids)

    # Calculate split sizes

    n_songs = len(shuffled_ids)

    n_train = int(n_songs * TRAIN_RATIO)
    n_val = int(n_songs * VAL_RATIO)

    # Assign everything remaining to test so that every song
    # belongs to exactly one split.
    n_test = n_songs - n_train - n_val

    train_ids = shuffled_ids[:n_train]
    val_ids = shuffled_ids[n_train:n_train + n_val]
    test_ids = shuffled_ids[n_train + n_val:]

    # Verify split

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    assert len(train_set) == len(train_ids)
    assert len(val_set) == len(val_ids)
    assert len(test_set) == len(test_ids)

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    assert (
        train_set | val_set | test_set
    ) == set(song_ids)

    assert (
        len(train_ids)
        + len(val_ids)
        + len(test_ids)
        == n_songs
    )

    # Save split

    splits = {
        "dataset": "POP909",

        "seed": SEED,

        "ratios": {
            "train": TRAIN_RATIO,
            "validation": VAL_RATIO,
            "test": TEST_RATIO,
        },

        "sequence_length": SEQUENCE_LENGTH,

        "n_songs": {
            "total": n_songs,
            "train": len(train_ids),
            "validation": len(val_ids),
            "test": len(test_ids),
        },

        "train": train_ids,
        "validation": val_ids,
        "test": test_ids,
    }

    with SPLIT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            splits,
            f,
            indent=2,
        )

    # Print summary

    print("\n" + "=" * 60)
    print("POP909 SPLIT")
    print("=" * 60)

    print(f"Seed:             {SEED}")
    print(f"Sequence length:  {SEQUENCE_LENGTH}")

    print()
    print(f"Total:            {n_songs}")
    print(
        f"Train:            {len(train_ids)} "
        f"({len(train_ids) / n_songs:.1%})"
    )
    print(
        f"Validation:       {len(val_ids)} "
        f"({len(val_ids) / n_songs:.1%})"
    )
    print(
        f"Test:             {len(test_ids)} "
        f"({len(test_ids) / n_songs:.1%})"
    )

    print()
    print(f"Saved to:")
    print(SPLIT_PATH)

    # Show first few IDs for reproducibility/debuggings

    print("\nFirst 10 training IDs:")
    print(train_ids[:10])

    print("\nFirst 10 validation IDs:")
    print(val_ids[:10])

    print("\nFirst 10 test IDs:")
    print(test_ids[:10])


if __name__ == "__main__":
    main()