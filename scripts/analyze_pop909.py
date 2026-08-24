from collections import Counter
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.pop909 import discover_songs, load_melody


DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "POP909"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "statistics"
FIGURE_DIR = PROJECT_ROOT / "figures"


def percentile_summary(values):
    """Return useful summary statistics for a list of numbers."""
    values = np.asarray(values)

    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
    }


def print_summary(name, summary):
    print(f"\n{name}")
    print("-" * len(name))

    for key, value in summary.items():
        print(f"{key:>8}: {value:.3f}")


def compute_statistics(songs):
    """
    Compute dataset-level and song-level statistics.

    Rests (pitch == 0) are excluded from pitch and interval statistics.
    """

    all_pitches = []
    all_durations = []
    all_intervals = []

    song_event_counts = []
    song_note_counts = []
    song_rest_counts = []
    song_total_durations = []
    song_note_durations = []
    song_rest_durations = []

    song_statistics = []

    for song in songs:
        song_id = song["song_id"]
        pitches, durations = load_melody(song["melody_path"])

        pitches = np.asarray(pitches)
        durations = np.asarray(durations)

        note_mask = pitches != 0

        note_pitches = pitches[note_mask]
        note_durations = durations[note_mask]

        rest_durations = durations[~note_mask]

        # Intervals between consecutive non-rest note events.
        #
        # Example:
        # [60, 62, 0, 64] -> [60, 62, 64]
        # intervals = [+2, +2]
        if len(note_pitches) >= 2:
            intervals = np.diff(note_pitches)
        else:
            intervals = np.array([], dtype=int)

        # Dataset-level collections
        all_pitches.extend(note_pitches.tolist())
        all_durations.extend(durations.tolist())
        all_intervals.extend(intervals.tolist())

        # Song-level statistics
        n_events = len(pitches)
        n_notes = int(np.sum(note_mask))
        n_rests = int(np.sum(~note_mask))

        total_duration = int(np.sum(durations))
        note_duration = int(np.sum(note_durations))
        rest_duration = int(np.sum(rest_durations))

        song_event_counts.append(n_events)
        song_note_counts.append(n_notes)
        song_rest_counts.append(n_rests)

        song_total_durations.append(total_duration)
        song_note_durations.append(note_duration)
        song_rest_durations.append(rest_duration)

        song_statistics.append(
            {
                "song_id": song_id,
                "n_events": n_events,
                "n_notes": n_notes,
                "n_rests": n_rests,
                "rest_event_ratio": (
                    n_rests / n_events if n_events > 0 else 0.0
                ),
                "total_duration": total_duration,
                "note_duration": note_duration,
                "rest_duration": rest_duration,
                "rest_duration_ratio": (
                    rest_duration / total_duration
                    if total_duration > 0
                    else 0.0
                ),
                "min_pitch": (
                    int(np.min(note_pitches))
                    if len(note_pitches) > 0
                    else None
                ),
                "max_pitch": (
                    int(np.max(note_pitches))
                    if len(note_pitches) > 0
                    else None
                ),
                "pitch_range": (
                    int(np.max(note_pitches) - np.min(note_pitches))
                    if len(note_pitches) > 0
                    else None
                ),
            }
        )

    pitch_counter = Counter(all_pitches)
    duration_counter = Counter(all_durations)
    interval_counter = Counter(all_intervals)

    dataset_statistics = {
        "n_songs": len(songs),

        "n_events": len(all_durations),
        "n_note_events": len(all_pitches),
        "n_rest_events": int(
            len(all_durations) - len(all_pitches)
        ),

        "total_duration": int(sum(all_durations)),

        "pitch_summary": (
            percentile_summary(all_pitches)
            if all_pitches
            else None
        ),

        "duration_summary": percentile_summary(all_durations),

        "interval_summary": (
            percentile_summary(all_intervals)
            if all_intervals
            else None
        ),

        "events_per_song": percentile_summary(song_event_counts),

        "notes_per_song": percentile_summary(song_note_counts),

        "rests_per_song": percentile_summary(song_rest_counts),

        "total_duration_per_song": percentile_summary(
            song_total_durations
        ),

        "note_duration_per_song": percentile_summary(
            song_note_durations
        ),

        "rest_duration_per_song": percentile_summary(
            song_rest_durations
        ),

        "unique_pitches": sorted(
            int(pitch) for pitch in pitch_counter.keys()
        ),

        "unique_durations": sorted(
            int(duration) for duration in duration_counter.keys()
        ),

        "pitch_counts": {
            str(pitch): count
            for pitch, count in sorted(pitch_counter.items())
        },

        "duration_counts": {
            str(duration): count
            for duration, count in sorted(
                duration_counter.items()
            )
        },

        "interval_counts": {
            str(interval): count
            for interval, count in sorted(
                interval_counter.items()
            )
        },
    }

    return dataset_statistics, song_statistics


def plot_pitch_histogram(pitch_counts, output_path):
    pitches = np.array(
        [int(pitch) for pitch in pitch_counts.keys()]
    )
    counts = np.array(list(pitch_counts.values()))

    plt.figure(figsize=(12, 5))
    plt.bar(pitches, counts, width=0.8)

    plt.xlabel("MIDI pitch")
    plt.ylabel("Number of note events")
    plt.title("POP909 Melody Pitch Distribution")

    plt.xticks(range(
        int(pitches.min()),
        int(pitches.max()) + 1,
        2
    ))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_duration_histogram(duration_counts, output_path):
    durations = np.array(
        [int(duration) for duration in duration_counts.keys()]
    )
    counts = np.array(list(duration_counts.values()))

    plt.figure(figsize=(10, 5))
    plt.bar(durations, counts)

    plt.xlabel("Duration (sixteenth-note units)")
    plt.ylabel("Number of events")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("POP909 Melody Duration Distribution")

    plt.xticks(durations)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_interval_histogram(interval_counts, output_path):
    intervals = np.array(
        [int(interval) for interval in interval_counts.keys()]
    )
    counts = np.array(list(interval_counts.values()))

    plt.figure(figsize=(14, 5))
    plt.bar(intervals, counts)

    plt.xlabel("Pitch interval (semitones)")
    plt.ylabel("Number of consecutive note pairs")
    plt.title("POP909 Melodic Interval Distribution")

    # Show the musically most relevant region clearly.
    plt.xlim(
        min(-24, int(intervals.min())),
        max(24, int(intervals.max()))
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_song_length_histogram(
    song_event_counts,
    song_total_durations,
    output_dir,
):
    # Number of events
    plt.figure(figsize=(10, 5))

    plt.hist(song_event_counts, bins=40)

    plt.xlabel("Number of melody events")
    plt.ylabel("Number of songs")
    plt.title("POP909 Song Length Distribution by Events")

    plt.tight_layout()
    plt.savefig(
        output_dir / "song_event_lengths.png",
        dpi=150,
    )
    plt.close()

    # Total duration
    plt.figure(figsize=(10, 5))

    plt.hist(song_total_durations, bins=40)

    plt.xlabel("Total duration (sixteenth-note units)")
    plt.ylabel("Number of songs")
    plt.title("POP909 Song Length Distribution by Duration")

    plt.tight_layout()
    plt.savefig(
        output_dir / "song_duration_lengths.png",
        dpi=150,
    )
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Searching for songs in:\n{DATA_ROOT}")

    songs = discover_songs(DATA_ROOT)

    print(f"\nFound {len(songs)} songs.")

    dataset_statistics, song_statistics = compute_statistics(
        songs
    )

    # Print overall statistics

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"Number of songs:       {dataset_statistics['n_songs']}")
    print(f"Number of events:      {dataset_statistics['n_events']}")
    print(
        f"Number of note events: "
        f"{dataset_statistics['n_note_events']}"
    )
    print(
        f"Number of rest events: "
        f"{dataset_statistics['n_rest_events']}"
    )
    print(
        f"Total duration:        "
        f"{dataset_statistics['total_duration']}"
    )

    print_summary(
        "Pitch statistics",
        dataset_statistics["pitch_summary"],
    )

    print_summary(
        "Duration statistics",
        dataset_statistics["duration_summary"],
    )

    print_summary(
        "Melodic interval statistics",
        dataset_statistics["interval_summary"],
    )

    print_summary(
        "Events per song",
        dataset_statistics["events_per_song"],
    )

    print_summary(
        "Notes per song",
        dataset_statistics["notes_per_song"],
    )

    print_summary(
        "Total duration per song",
        dataset_statistics["total_duration_per_song"],
    )

    print(
        "\nUnique pitches:",
        dataset_statistics["unique_pitches"],
    )

    print(
        "\nUnique durations:",
        dataset_statistics["unique_durations"],
    )

    # Reconstruct arrays needed for song-length plots

    song_event_counts = [
        song["n_events"]
        for song in song_statistics
    ]

    song_total_durations = [
        song["total_duration"]
        for song in song_statistics
    ]

    # Save statistics

    statistics_path = OUTPUT_DIR / "dataset_statistics.json"

    with statistics_path.open("w") as f:
        json.dump(
            dataset_statistics,
            f,
            indent=2,
        )

    song_statistics_path = (
        OUTPUT_DIR / "song_statistics.json"
    )

    with song_statistics_path.open("w") as f:
        json.dump(
            song_statistics,
            f,
            indent=2,
        )

    # Create figures

    plot_pitch_histogram(
        dataset_statistics["pitch_counts"],
        FIGURE_DIR / "pitch_distribution.png",
    )

    plot_duration_histogram(
        dataset_statistics["duration_counts"],
        FIGURE_DIR / "duration_distribution.png",
    )

    plot_interval_histogram(
        dataset_statistics["interval_counts"],
        FIGURE_DIR / "interval_distribution.png",
    )

    plot_song_length_histogram(
        song_event_counts,
        song_total_durations,
        FIGURE_DIR,
    )

    print("\n" + "=" * 60)
    print("FILES CREATED")
    print("=" * 60)

    print(statistics_path)
    print(song_statistics_path)

    for path in sorted(FIGURE_DIR.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()