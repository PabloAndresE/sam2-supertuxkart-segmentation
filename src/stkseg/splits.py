"""Split strategies for a dataset made of consecutive video frames.

The frames are not independent samples: each track contributes 250 consecutive
frames from a single race, and a frame is measurably more similar to its
neighbour than to a distant frame (mean absolute difference 0.066 vs 0.126 on
``abyss``). A uniformly random split therefore places correlated frames on both
sides of the boundary and reports an optimistic score.

Three strategies are provided, in increasing order of difficulty:

``random``
    Uniform over frames. Optimistic; kept only to reproduce the original result.
``temporal_block``
    Frames are cut into contiguous blocks per track and whole blocks are
    assigned to a split, so no validation frame sits adjacent to a training one.
``track_holdout``
    Entire tracks are held out. Measures generalisation to unseen scenery,
    which is the question a random split cannot answer at all.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

STRATEGIES = ("random", "temporal_block", "track_holdout")


@dataclass(frozen=True)
class SampleKey:
    """Identifies a frame by the track it came from and its position in the race."""

    track: str
    frame_index: int


def _assign(units: list, fractions: tuple[float, float, float], seed: int) -> dict[str, list]:
    """Shuffle opaque units and cut them into train/val/test by fraction."""
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {fractions}")

    shuffled = list(units)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    if n_train + n_val > n:
        n_val = n - n_train

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def make_splits(
    keys: list[SampleKey],
    strategy: str = "temporal_block",
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    block_size: int = 25,
    holdout_tracks: tuple[str, str] | None = None,
) -> dict[str, list[int]]:
    """Map each split name to the dataset indices it owns.

    Returns indices into ``keys``. The three lists are always disjoint and their
    union is every index, whatever the strategy.

    ``holdout_tracks`` pins which ``(val, test)`` tracks are held out under
    ``track_holdout`` instead of drawing them from the seed. Naming them is what
    makes leave-one-track-out possible: the measured drop on unseen scenery is
    otherwise a single random draw of one track.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}, expected one of {STRATEGIES}")

    if strategy == "random":
        groups = {i: [i] for i in range(len(keys))}

    elif strategy == "temporal_block":
        groups = defaultdict(list)
        for i, key in enumerate(keys):
            groups[(key.track, key.frame_index // block_size)].append(i)

    else:  # track_holdout
        groups = defaultdict(list)
        for i, key in enumerate(keys):
            groups[key.track].append(i)
        if len(groups) < 3:
            raise ValueError(
                f"track_holdout needs at least 3 tracks, found {len(groups)}"
            )

        if holdout_tracks is not None:
            val_track, test_track = holdout_tracks
            missing = {val_track, test_track} - set(groups)
            if missing:
                raise ValueError(f"unknown holdout track(s) {sorted(missing)}")
            if val_track == test_track:
                raise ValueError("val and test holdout tracks must differ")
            return {
                "train": sorted(
                    i
                    for track, idx in groups.items()
                    if track not in (val_track, test_track)
                    for i in idx
                ),
                "val": sorted(groups[val_track]),
                "test": sorted(groups[test_track]),
            }

    if holdout_tracks is not None and strategy != "track_holdout":
        raise ValueError("holdout_tracks only applies to the track_holdout strategy")

    assigned = _assign(list(groups.keys()), fractions, seed)
    return {
        split: sorted(i for group in group_keys for i in groups[group])
        for split, group_keys in assigned.items()
    }


def describe(keys: list[SampleKey], splits: dict[str, list[int]]) -> str:
    """Human-readable summary: sizes plus which tracks landed where."""
    lines = []
    for split, indices in splits.items():
        tracks = sorted({keys[i].track for i in indices})
        shown = ", ".join(tracks[:4]) + (" ..." if len(tracks) > 4 else "")
        lines.append(f"  {split:5s} {len(indices):5d} frames  {len(tracks)} tracks  [{shown}]")
    return "\n".join(lines)
