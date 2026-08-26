from pathlib import Path
import json
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.piano_genie_dataset import PianoGenieDataset


DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "pop909_events.pt"
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "pop909_splits.json"


def assert_equal(actual, expected, message=""):
    if actual != expected:
        raise AssertionError(
            f"{message}\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )


def assert_tensor_equal(actual, expected, message=""):
    expected = torch.tensor(expected, dtype=actual.dtype)

    if not torch.equal(actual, expected):
        raise AssertionError(
            f"{message}\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )


def test_extract_note_sequence():
    """
    Verify that:
        1. rests are removed from the note sequence;
        2. rests still contribute to elapsed time;
        3. the first note gets delta_time == 0.
    """

    song = {
        "song_id": "synthetic",
        "pitches": torch.tensor(
            [60, 62, 0, 64, 0, 65],
            dtype=torch.long,
        ),
        "durations": torch.tensor(
            [4, 2, 2, 8, 4, 1],
            dtype=torch.long,
        ),
    }

    pitches, delta_times = (
        PianoGenieDataset._extract_note_sequence(song)
    )

    # Original events:
    #
    # 60 starts at 0, duration 4
    # 62 starts at 4, duration 2
    # rest starts at 6, duration 2
    # 64 starts at 8, duration 8
    # rest starts at 16, duration 4
    # 65 starts at 20
    #
    # Therefore:
    #
    # pitches:
    #     [60, 62, 64, 65]
    #
    # delta times:
    #     [0, 4, 4, 12]

    assert_tensor_equal(
        pitches,
        [60, 62, 64, 65],
        "Rest removal is incorrect.",
    )

    assert_tensor_equal(
        delta_times,
        [0, 4, 4, 12],
        "Delta-time calculation is incorrect.",
    )

    print("PASS: note extraction and delta-time calculation")


def test_single_song_with_no_rests():
    """
    Verify delta times when there are no rests.
    """

    song = {
        "song_id": "synthetic_no_rests",
        "pitches": torch.tensor(
            [60, 62, 64, 65],
            dtype=torch.long,
        ),
        "durations": torch.tensor(
            [4, 2, 8, 4],
            dtype=torch.long,
        ),
    }

    pitches, delta_times = (
        PianoGenieDataset._extract_note_sequence(song)
    )

    assert_tensor_equal(
        pitches,
        [60, 62, 64, 65],
        "Pitch sequence is incorrect.",
    )

    assert_tensor_equal(
        delta_times,
        [0, 4, 2, 8],
        "Delta times are incorrect.",
    )

    print("PASS: no-rest delta-time calculation")


def test_dataset_real_data():
    """
    Basic sanity checks against the actual POP909 dataset.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(SPLIT_PATH)

    with SPLIT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        splits = json.load(f)

    train_dataset = PianoGenieDataset(
        DATASET_PATH,
        SPLIT_PATH,
        split="train",
    )

    val_dataset = PianoGenieDataset(
        DATASET_PATH,
        SPLIT_PATH,
        split="validation",
        validation_mode="non_overlapping",
    )

    test_dataset = PianoGenieDataset(
        DATASET_PATH,
        SPLIT_PATH,
        split="test",
        validation_mode="non_overlapping",
    )

    # Dataset must not be empty.

    assert len(train_dataset) > 0
    assert len(val_dataset) > 0
    assert len(test_dataset) > 0

    print(
        f"PASS: real dataset loaded "
        f"(train={len(train_dataset)}, "
        f"validation={len(val_dataset)}, "
        f"test={len(test_dataset)})"
    )

    # Every returned segment must have length 128.

    for dataset in [
        train_dataset,
        val_dataset,
        test_dataset,
    ]:
        sample = dataset[0]

        assert_equal(
            sample["pitches"].shape,
            torch.Size([128]),
            "Incorrect pitch segment shape.",
        )

        assert_equal(
            sample["delta_times"].shape,
            torch.Size([128]),
            "Incorrect delta-time segment shape.",
        )

        assert sample["pitches"].dtype == torch.long
        assert sample["delta_times"].dtype == torch.long

    print("PASS: segment shapes and dtypes")

    # Pitches must be valid non-rest MIDI pitches.

    sample = train_dataset[0]

    pitches = sample["pitches"]

    assert torch.all(pitches >= 42)
    assert torch.all(pitches <= 98)

    print("PASS: pitch ranges")


def test_first_validation_mode():
    """
    "first" should return exactly one segment per eligible song,
    beginning at note 0.
    """

    dataset = PianoGenieDataset(
        DATASET_PATH,
        SPLIT_PATH,
        split="validation",
        validation_mode="first",
    )

    # Every eligible song contributes exactly one segment.
    assert len(dataset) == len(dataset.songs)

    for i in range(
        min(len(dataset), 10)
    ):
        sample = dataset[i]

        assert_equal(
            sample["start"],
            0,
            "First validation segment does not start at 0.",
        )

    print(
        "PASS: validation_mode='first'"
    )


def test_non_overlapping_validation_mode():
    """
    Every evaluation segment should be a complete 128-note
    window, and starts should be 0, 128, 256, ...
    """

    dataset = PianoGenieDataset(
        DATASET_PATH,
        SPLIT_PATH,
        split="validation",
        validation_mode="non_overlapping",
    )

    segments_by_song = {}

    for segment in dataset.segments:
        song_index = segment["song_index"]
        start = segment["start"]

        segments_by_song.setdefault(
            song_index,
            [],
        ).append(start)

    for song_index, starts in segments_by_song.items():
        expected_starts = list(
            range(
                0,
                len(dataset.songs[song_index]["pitches"])
                - dataset.seq_len
                + 1,
                dataset.seq_len,
            )
        )

        assert_equal(
            starts,
            expected_starts,
            "Incorrect non-overlapping segment starts.",
        )

    print(
        "PASS: validation_mode='non_overlapping'"
    )


def test_random_validation_mode():
    """
    Random evaluation mode should be deterministic for a fixed
    seed and should produce exactly one segment per eligible song.
    """

    dataset_a = PianoGenieDataset(
        DATASET_PATH,
        SPLIT_PATH,
        split="validation",
        validation_mode="random",
        seed=123,
    )

    dataset_b = PianoGenieDataset(
        DATASET_PATH,
        SPLIT_PATH,
        split="validation",
        validation_mode="random",
        seed=123,
    )

    assert_equal(
        len(dataset_a),
        len(dataset_a.songs),
        "Random mode should have one segment per song.",
    )

    assert_equal(
        len(dataset_b),
        len(dataset_b.songs),
        "Random mode should have one segment per song.",
    )

    starts_a = [
        segment["start"]
        for segment in dataset_a.segments
    ]

    starts_b = [
        segment["start"]
        for segment in dataset_b.segments
    ]

    assert_equal(
        starts_a,
        starts_b,
        "Random validation mode is not deterministic.",
    )

    print(
        "PASS: validation_mode='random'"
    )


def test_training_randomness():
    """
    Verify that training samples are generated dynamically.

    We don't require every pair of samples to differ, but over
    several samples we should normally observe more than one crop.
    """

    dataset = PianoGenieDataset(
        DATASET_PATH,
        SPLIT_PATH,
        split="train",
    )

    # Find a song with more than one possible crop.
    candidate_index = None

    for i, song in enumerate(dataset.songs):
        if len(song["pitches"]) > dataset.seq_len:
            candidate_index = i
            break

    if candidate_index is None:
        raise AssertionError(
            "No training song has more than 128 notes."
        )

    # Dataset indexing is song-based for training.
    sample_index = candidate_index

    starts = set()

    for _ in range(50):
        sample = dataset[sample_index]
        starts.add(sample["start"])

    # With a song longer than 128 notes, we expect random sampling
    # to produce multiple starting positions.
    assert len(starts) > 1, (
        "Training Dataset does not appear to be sampling "
        "random crops."
    )

    print(
        "PASS: training uses dynamic random crops"
    )


def test_crop_boundaries():
    """
    Verify that every generated crop lies entirely inside its song.
    """

    for split, mode in [
        ("validation", "first"),
        ("validation", "non_overlapping"),
        ("validation", "random"),
        ("test", "first"),
        ("test", "non_overlapping"),
        ("test", "random"),
    ]:
        dataset = PianoGenieDataset(
            DATASET_PATH,
            SPLIT_PATH,
            split=split,
            validation_mode=mode,
            seed=42,
        )

        for segment in dataset.segments:
            song = dataset.songs[
                segment["song_index"]
            ]

            start = segment["start"]
            end = start + dataset.seq_len

            assert start >= 0
            assert end <= len(song["pitches"])

        print(
            f"PASS: crop boundaries "
            f"({split}, {mode})"
        )


def main():
    print("=" * 60)
    print("Piano Genie Dataset Tests")
    print("=" * 60)

    print("\nSynthetic transformation tests")
    print("-" * 60)

    test_extract_note_sequence()
    test_single_song_with_no_rests()

    print("\nReal POP909 tests")
    print("-" * 60)

    test_dataset_real_data()
    test_first_validation_mode()
    test_non_overlapping_validation_mode()
    test_random_validation_mode()
    test_training_randomness()
    test_crop_boundaries()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()