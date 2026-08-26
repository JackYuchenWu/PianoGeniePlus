from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

class PianoGenieDataset(Dataset):
    """
    Dataset for Piano Genie-style training on POP909.

    Each model timestep corresponds to one non-rest melody note.

    The canonical POP909 representation contains:
        pitches:   [event_0, event_1, ...]
        durations: [duration_0, duration_1, ...]

    Rests are represented by pitch == 0.

    Before removing rests, we compute the onset time of every event.
    The onset times of non-rest notes are then converted to delta-times.

    For example:

        pitch:     [60, 62,  0, 64]
        duration:  [ 4,  2,  2,  8]

    gives:

        note_pitch: [60, 62, 64]
        delta_time: [ 0,  4,  4]

    where delta_time == 0 is the start-of-sequence token.

    Parameters
    ----------
    dataset_path:
        Path to pop909_events.pt.

    split_path:
        Path to pop909_splits.json.

    split:
        One of "train", "validation", or "test".

    seq_len:
        Number of note events in each model input.

    validation_mode:
        How to construct segments for validation/test.

        "first":
            Return only the first seq_len notes from each song.

        "non_overlapping":
            Return every complete non-overlapping seq_len window.

        "random":
            Return one random seq_len window from each song.

        This option is ignored for the training split.

    seed:
        Random seed used by the "random" validation/test mode.

    """

    def __init__(
        self,
        dataset_path: str | Path,
        split_path: str | Path,
        split: str,
        seq_len: int = 128,
        validation_mode: str = "non_overlapping",
        seed: int = 42,
    ):
        super().__init__()

        if seq_len <= 0:
            raise ValueError(
                f"seq_len must be positive, got {seq_len}"
            )

        if split not in {
            "train",
            "validation",
            "test",
        }:
            raise ValueError(
                "split must be 'train', 'validation', or 'test'"
            )

        if validation_mode not in {
            "first",
            "non_overlapping",
            "random",
        }:
            raise ValueError(
                "validation_mode must be one of "
                "'first', 'non_overlapping', or 'random'"
            )

        self.dataset_path = Path(dataset_path)
        self.split_path = Path(split_path)

        self.split = split
        self.seq_len = seq_len
        self.validation_mode = validation_mode
        self.seed = seed

        # Load canonical dataset

        dataset = torch.load(
            self.dataset_path,
            weights_only=False,
        )

        if "songs" not in dataset:
            raise ValueError(
                "Dataset must contain a 'songs' key."
            )

        songs = dataset["songs"]

        songs_by_id = {
            song["song_id"]: song
            for song in songs
        }

        # Load split

        with self.split_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            splits = json.load(f)

        if split not in splits:
            raise ValueError(
                f"Split '{split}' not found in {self.split_path}"
            )

        split_song_ids = splits[split]

        # Validate sequence length against split metadata

        split_seq_len = splits.get("sequence_length")

        if split_seq_len is not None:
            if seq_len != split_seq_len:
                raise ValueError(
                    f"seq_len={seq_len} does not match the "
                    f"sequence length recorded in the split file "
                    f"({split_seq_len})."
                )

        # Prepare eligible songs

        self.songs = []

        for song_id in split_song_ids:
            if song_id not in songs_by_id:
                raise ValueError(
                    f"Song '{song_id}' from split file "
                    f"was not found in the dataset."
                )

            song = songs_by_id[song_id]

            note_pitches, delta_times = (
                self._extract_note_sequence(song)
            )

            n_notes = len(note_pitches)

            # A song cannot produce a complete segment if it
            # contains fewer than seq_len notes.
            if n_notes < self.seq_len:
                continue

            self.songs.append(
                {
                    "song_id": song_id,
                    "pitches": note_pitches,
                    "delta_times": delta_times,
                }
            )

        # Construct segment index

        self.segments = []

        if self.split == "train":
            # Training samples are generated dynamically.
            #
            # Each song contributes one Dataset item. Every call
            # to __getitem__ chooses a new random crop.
            self.segments = [
                {
                    "song_index": i,
                }
                for i in range(len(self.songs))
            ]

        else:
            self._create_evaluation_segments()

    @staticmethod
    def _extract_note_sequence(song):
        """
        Convert canonical event representation into a note-only
        sequence with onset delta-times.

        Rest events contribute to elapsed time but do not become
        model timesteps.
        """

        pitches = song["pitches"]
        durations = song["durations"]

        if len(pitches) != len(durations):
            raise ValueError(
                f"Song {song['song_id']} has mismatched "
                f"pitches/durations lengths."
            )

        # Current absolute onset position.
        current_time = 0

        note_pitches = []
        note_onsets = []

        for pitch, duration in zip(
            pitches.tolist(),
            durations.tolist(),
        ):
            if pitch != 0:
                note_pitches.append(pitch)
                note_onsets.append(current_time)

            current_time += duration

        if not note_pitches:
            return (
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )

        note_pitches = torch.tensor(
            note_pitches,
            dtype=torch.long,
        )

        note_onsets = torch.tensor(
            note_onsets,
            dtype=torch.long,
        )

        # Delta time between consecutive notes.
        #
        # The first note receives 0, which is reserved for
        # START in our representation.
        delta_times = torch.zeros(
            len(note_onsets),
            dtype=torch.long,
        )

        if len(note_onsets) > 1:
            delta_times[1:] = (
                note_onsets[1:] - note_onsets[:-1]
            )

        return note_pitches, delta_times

    def _create_evaluation_segments(self):
        """
        Construct deterministic evaluation segments.

        "first":
            one segment starting at 0.

        "non_overlapping":
            all complete windows of length seq_len.

        "random":
            one deterministic pseudo-random window per song.
        """

        rng = random.Random(self.seed)

        for song_index, song in enumerate(self.songs):
            n_notes = len(song["pitches"])

            if self.validation_mode == "first":
                starts = [0]

            elif self.validation_mode == "non_overlapping":
                starts = range(
                    0,
                    n_notes - self.seq_len + 1,
                    self.seq_len,
                )

            elif self.validation_mode == "random":
                max_start = n_notes - self.seq_len

                if max_start == 0:
                    start = 0
                else:
                    start = rng.randint(
                        0,
                        max_start,
                    )

                starts = [start]

            else:
                raise RuntimeError(
                    f"Unknown validation mode: "
                    f"{self.validation_mode}"
                )

            for start in starts:
                self.segments.append(
                    {
                        "song_index": song_index,
                        "start": start,
                    }
                )

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        segment = self.segments[index]

        song_index = segment["song_index"]
        song = self.songs[song_index]

        pitches = song["pitches"]
        delta_times = song["delta_times"]

        # Training: dynamically choose a random crop

        if self.split == "train":
            max_start = len(pitches) - self.seq_len

            start = random.randint(
                0,
                max_start,
            )

        # Validation/test: use predetermined segment

        else:
            start = segment["start"]

        end = start + self.seq_len

        return {
            "pitches": pitches[start:end],
            "delta_times": delta_times[start:end],
            "song_id": song["song_id"],
            "start": start,
        }

def dataset_test_case():
    train_dataset = PianoGenieDataset(
        "data/processed/pop909_events.pt",
        "data/processed/pop909_splits.json",
        split="train",
    )

    val_dataset = PianoGenieDataset(
        "data/processed/pop909_events.pt",
        "data/processed/pop909_splits.json",
        split="validation",
        validation_mode="non_overlapping",
    )

    test_dataset = PianoGenieDataset(
        "data/processed/pop909_events.pt",
        "data/processed/pop909_splits.json",
        split="test",
        validation_mode="non_overlapping",
    )

    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))

    sample = train_dataset[0]

    print(sample["song_id"])
    print(sample["start"])
    print(sample["pitches"].shape)
    print(sample["delta_times"].shape)
    print(sample["pitches"][:20])
    print(sample["delta_times"][:20])

if __name__=="__main__":
    dataset_test_case()