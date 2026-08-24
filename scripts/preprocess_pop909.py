from pathlib import Path
import json
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.pop909 import discover_songs, load_melody


DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "POP909"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

DATASET_PATH = OUTPUT_DIR / "pop909_events.pt"
METADATA_PATH = OUTPUT_DIR / "pop909_metadata.json"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    songs = discover_songs(DATA_ROOT)

    print(f"Found {len(songs)} songs.")

    processed_songs = []

    all_note_pitches = []
    all_durations = []

    total_events = 0
    total_note_events = 0
    total_rest_events = 0

    for i, song in enumerate(songs, start=1):
        song_id = song["song_id"]

        pitches, durations = load_melody(song["melody_path"])

        pitches_tensor = torch.tensor(
            pitches,
            dtype=torch.long,
        )

        durations_tensor = torch.tensor(
            durations,
            dtype=torch.long,
        )

        note_mask = pitches_tensor != 0

        n_events = len(pitches_tensor)
        n_note_events = int(note_mask.sum().item())
        n_rest_events = n_events - n_note_events

        processed_song = {
            "song_id": song_id,
            "pitches": pitches_tensor,
            "durations": durations_tensor,
        }

        processed_songs.append(processed_song)

        total_events += n_events
        total_note_events += n_note_events
        total_rest_events += n_rest_events

        all_note_pitches.extend(
            pitches_tensor[note_mask].tolist()
        )

        all_durations.extend(
            durations_tensor.tolist()
        )

        if i % 100 == 0 or i == len(songs):
            print(
                f"Processed {i}/{len(songs)} songs."
            )

    # Save dataset

    dataset = {
        "songs": processed_songs,
    }

    torch.save(dataset, DATASET_PATH)

    print(f"\nSaved dataset to:")
    print(DATASET_PATH)

    # Create metadata

    unique_pitches = sorted(set(all_note_pitches))
    unique_durations = sorted(set(all_durations))

    metadata = {
        "dataset": "POP909",
        "n_songs": len(processed_songs),

        "representation": {
            "type": "event_sequence",
            "description": (
                "Each song is represented as a sequence of "
                "(pitch, duration) events."
            ),
            "pitch_representation": "absolute_midi",
            "duration_unit": "sixteenth_note",
            "rest_token": 0,
        },

        "statistics": {
            "n_events": total_events,
            "n_note_events": total_note_events,
            "n_rest_events": total_rest_events,

            "pitch_min": min(all_note_pitches),
            "pitch_max": max(all_note_pitches),
            "n_unique_pitches": len(unique_pitches),
            "unique_pitches": unique_pitches,

            "duration_min": min(all_durations),
            "duration_max": max(all_durations),
            "n_unique_durations": len(unique_durations),
            "unique_durations": unique_durations,
        },

        "files": {
            "dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        },
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(f"\nSaved metadata to:")
    print(METADATA_PATH)

    # Verification

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    loaded_dataset = torch.load(
        DATASET_PATH,
        weights_only=False,
    )

    loaded_songs = loaded_dataset["songs"]

    assert len(loaded_songs) == len(processed_songs)

    for original, loaded in zip(
        processed_songs,
        loaded_songs,
    ):
        assert original["song_id"] == loaded["song_id"]

        assert torch.equal(
            original["pitches"],
            loaded["pitches"],
        )

        assert torch.equal(
            original["durations"],
            loaded["durations"],
        )

    print(
        f"Successfully verified "
        f"{len(loaded_songs)} songs."
    )

    print("\nFirst song:")
    first_song = loaded_songs[0]

    print(
        f"song_id: "
        f"{first_song['song_id']}"
    )

    print(
        f"number of events: "
        f"{len(first_song['pitches'])}"
    )

    print(
        f"pitches[:10]: "
        f"{first_song['pitches'][:10].tolist()}"
    )

    print(
        f"durations[:10]: "
        f"{first_song['durations'][:10].tolist()}"
    )


if __name__ == "__main__":
    main()