from pathlib import Path


def load_melody(path: str | Path):
    """Load one POP909 melody.txt file.

    Returns
    -------
    pitches : list[int]
        MIDI pitches, where 0 represents a rest.
    durations : list[int]
        Durations in sixteenth-note units.
    """
    path = Path(path)

    pitches = []
    durations = []

    with path.open("r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 2:
                raise ValueError(
                    f"{path}, line {line_number}: "
                    f"expected 2 values, got {len(parts)}"
                )

            pitch, duration = map(int, parts)

            if not 0 <= pitch <= 127:
                raise ValueError(
                    f"{path}, line {line_number}: "
                    f"invalid MIDI pitch {pitch}"
                )

            if duration <= 0:
                raise ValueError(
                    f"{path}, line {line_number}: "
                    f"duration must be positive, got {duration}"
                )

            pitches.append(pitch)
            durations.append(duration)

    if not pitches:
        raise ValueError(f"{path} contains no melody events")

    return pitches, durations

def discover_songs(root: str | Path):
    """Find all song folders containing melody.txt."""
    root = Path(root)

    songs = []

    for song_dir in sorted(root.iterdir()):
        if not song_dir.is_dir():
            continue

        melody_path = song_dir / "melody.txt"

        if melody_path.exists():
            songs.append(
                {
                    "song_id": song_dir.name,
                    "melody_path": melody_path,
                }
            )

    return songs

def validate_songs():
    songs = discover_songs("data/raw/POP909")

    print(f"Found {len(songs)} songs")

    for song in songs[:5]:
        pitches, durations = load_melody(song["melody_path"])

        print(
            song["song_id"],
            len(pitches),
            min(pitches),
            max(pitches),
            sum(durations),
        )

if __name__=="__main__":
    validate_songs()