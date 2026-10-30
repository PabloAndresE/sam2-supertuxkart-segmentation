import pytest

from stkseg.splits import SampleKey, make_splits


@pytest.mark.parametrize("strategy", ["random", "temporal_block", "track_holdout"])
def test_splits_are_disjoint_and_exhaustive(keys, strategy):
    splits = make_splits(keys, strategy=strategy, seed=1)
    indices = [i for group in splits.values() for i in group]
    assert sorted(indices) == list(range(len(keys)))


@pytest.mark.parametrize("strategy", ["random", "temporal_block", "track_holdout"])
def test_splits_are_deterministic(keys, strategy):
    assert make_splits(keys, strategy=strategy, seed=7) == make_splits(
        keys, strategy=strategy, seed=7
    )


def test_seed_changes_the_partition(keys):
    assert make_splits(keys, strategy="temporal_block", seed=1) != make_splits(
        keys, strategy="temporal_block", seed=2
    )


def test_track_holdout_shares_no_track_between_splits(keys):
    splits = make_splits(keys, strategy="track_holdout", seed=3)
    tracks = {name: {keys[i].track for i in idx} for name, idx in splits.items()}
    assert tracks["train"] & tracks["val"] == set()
    assert tracks["train"] & tracks["test"] == set()
    assert tracks["val"] & tracks["test"] == set()


def test_temporal_block_never_splits_a_block(keys):
    block = 25
    splits = make_splits(keys, strategy="temporal_block", seed=5, block_size=block)
    owner = {}
    for name, indices in splits.items():
        for i in indices:
            owner[(keys[i].track, keys[i].frame_index // block)] = owner.get(
                (keys[i].track, keys[i].frame_index // block), name
            )
            assert owner[(keys[i].track, keys[i].frame_index // block)] == name


def test_temporal_block_keeps_neighbours_together(keys):
    """No validation frame should sit immediately next to a training frame."""
    splits = make_splits(keys, strategy="temporal_block", seed=5, block_size=25)
    location = {}
    for name, indices in splits.items():
        for i in indices:
            location[(keys[i].track, keys[i].frame_index)] = name

    adjacent_leaks = 0
    for (track, index), name in location.items():
        neighbour = location.get((track, index + 1))
        if neighbour and neighbour != name:
            adjacent_leaks += 1
    # Only block boundaries may touch: 6 tracks x 9 internal boundaries at most.
    assert adjacent_leaks <= 6 * (250 // 25)


def test_random_split_leaks_many_neighbours(keys):
    """Documents why the random strategy is optimistic, rather than assuming it."""
    splits = make_splits(keys, strategy="random", seed=5)
    location = {}
    for name, indices in splits.items():
        for i in indices:
            location[(keys[i].track, keys[i].frame_index)] = name

    leaks = sum(
        1
        for (track, index), name in location.items()
        if location.get((track, index + 1), name) != name
    )
    assert leaks > 300


def test_fractions_must_sum_to_one(keys):
    with pytest.raises(ValueError, match="sum to 1.0"):
        make_splits(keys, fractions=(0.8, 0.3, 0.1))


def test_unknown_strategy_is_rejected(keys):
    with pytest.raises(ValueError, match="unknown strategy"):
        make_splits(keys, strategy="chronological")


def test_track_holdout_requires_three_tracks():
    keys = [SampleKey(track="abyss", frame_index=i) for i in range(10)]
    with pytest.raises(ValueError, match="at least 3 tracks"):
        make_splits(keys, strategy="track_holdout")
