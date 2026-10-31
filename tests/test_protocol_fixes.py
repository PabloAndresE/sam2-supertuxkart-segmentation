"""Regressions for two protocol defects found after the first full suite."""

import pytest
import torch

from stkseg.config import DataConfig
from stkseg.metrics import SHARED_CLASSES, mean_iou_over
from stkseg.splits import make_splits


class TestSeedSeparation:
    """`data.seed` used to drive both the split and the initialisation."""

    def test_init_seed_defaults_to_split_seed(self):
        assert DataConfig(seed=7).effective_init_seed == 7

    def test_init_seed_overrides_independently(self):
        cfg = DataConfig(seed=7, init_seed=99)
        assert cfg.effective_init_seed == 99
        assert cfg.seed == 7

    def test_changing_init_seed_leaves_the_split_untouched(self, keys):
        """Varying initialisation must not move a single frame between splits."""
        a = DataConfig(seed=42, init_seed=1)
        b = DataConfig(seed=42, init_seed=2)
        assert make_splits(keys, seed=a.seed) == make_splits(keys, seed=b.seed)
        assert a.effective_init_seed != b.effective_init_seed


class TestNamedHoldoutTracks:
    def test_named_tracks_land_in_their_splits(self, keys):
        splits = make_splits(
            keys, strategy="track_holdout", holdout_tracks=("abyss", "hacienda")
        )
        assert {keys[i].track for i in splits["val"]} == {"abyss"}
        assert {keys[i].track for i in splits["test"]} == {"hacienda"}
        assert "abyss" not in {keys[i].track for i in splits["train"]}

    def test_every_track_can_be_the_unseen_one(self, keys):
        tracks = sorted({k.track for k in keys})
        for held in tracks:
            other = next(t for t in tracks if t != held)
            splits = make_splits(
                keys, strategy="track_holdout", holdout_tracks=(held, other)
            )
            assert {keys[i].track for i in splits["val"]} == {held}

    def test_named_tracks_are_seed_independent(self, keys):
        kwargs = {"strategy": "track_holdout", "holdout_tracks": ("abyss", "hacienda")}
        assert make_splits(keys, seed=1, **kwargs) == make_splits(keys, seed=2, **kwargs)

    def test_unknown_track_is_rejected(self, keys):
        with pytest.raises(ValueError, match="unknown holdout track"):
            make_splits(keys, strategy="track_holdout", holdout_tracks=("abyss", "mars"))

    def test_val_and_test_must_differ(self, keys):
        with pytest.raises(ValueError, match="must differ"):
            make_splits(keys, strategy="track_holdout", holdout_tracks=("abyss", "abyss"))

    def test_rejected_for_other_strategies(self, keys):
        with pytest.raises(ValueError, match="only applies to the track_holdout"):
            make_splits(keys, strategy="temporal_block", holdout_tracks=("abyss", "hacienda"))


class TestSharedClassMean:
    """Plain mIoU is not comparable across splits with different class sets."""

    def test_fixed_list_ignores_an_extra_present_class(self):
        # Projectile (6) predicted correctly in one split, absent in another:
        # the shared-six mean must not move.
        base = torch.tensor([0, 1, 2, 3, 4, 5])
        with_extra = torch.tensor([0, 1, 2, 3, 4, 5, 6])
        assert mean_iou_over(base.clone(), base, SHARED_CLASSES) == pytest.approx(1.0)
        assert mean_iou_over(with_extra.clone(), with_extra, SHARED_CLASSES) == pytest.approx(1.0)

    def test_hallucinated_class_drags_the_plain_mean_further(self):
        """A class absent from the target but predicted inflates the denominator."""
        from stkseg.metrics import mean_iou

        target = torch.tensor([0, 1, 2, 3, 4, 5])
        pred = torch.tensor([0, 1, 2, 3, 4, 6])  # Bomb missed, Projectile invented

        # Shared-six: Bomb is genuinely in the target and missed, so it counts 0.
        assert mean_iou_over(pred, target, SHARED_CLASSES) == pytest.approx(5 / 6)
        # Plain mIoU additionally averages in the invented Projectile at 0.
        assert mean_iou(pred, target) == pytest.approx(5 / 7)
        assert mean_iou(pred, target) < mean_iou_over(pred, target, SHARED_CLASSES)

    def test_returns_none_rather_than_inventing_a_number(self):
        target = torch.tensor([0, 1])
        assert mean_iou_over(target.clone(), target, SHARED_CLASSES) is None
