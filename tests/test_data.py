import torch

from stkseg.data import SuperTuxKartDataset, build_dataloaders


def test_dataset_pairs_images_with_masks(tiny_dataset):
    dataset = SuperTuxKartDataset(tiny_dataset, img_size=(32, 32))
    assert len(dataset) == 30
    assert {k.track for k in dataset.keys} == {"abyss", "gran_paradiso_island", "hacienda"}


def test_dataset_returns_normalised_image_and_integer_mask(tiny_dataset):
    image, mask = SuperTuxKartDataset(tiny_dataset, img_size=(32, 32))[0]
    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert mask.shape == (32, 32)
    assert mask.dtype == torch.int64
    assert int(mask.min()) >= 0 and int(mask.max()) <= 6


def test_build_dataloaders_yields_three_disjoint_splits(tiny_dataset):
    loaders = build_dataloaders(
        tiny_dataset,
        img_size=(32, 32),
        batch_size=2,
        strategy="track_holdout",
        num_workers=0,
        verbose=False,
    )
    assert set(loaders) == {"train", "val", "test"}

    owned = [set(loaders[s].dataset.indices) for s in ("train", "val", "test")]
    assert owned[0] & owned[1] == set()
    assert owned[0] & owned[2] == set()
    assert sum(len(o) for o in owned) == 30


def test_validation_loader_is_not_augmented(tiny_dataset):
    loaders = build_dataloaders(
        tiny_dataset, img_size=(32, 32), batch_size=2, num_workers=0, verbose=False
    )
    assert loaders["val"].dataset.dataset.augment is False
    assert loaders["train"].dataset.dataset.augment is True
