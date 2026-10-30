import numpy as np
import pytest
from PIL import Image

from stkseg.splits import SampleKey

TRACKS = ["abyss", "gran_paradiso_island", "hacienda", "lighthouse", "olivermath", "volcano_island"]
FRAMES_PER_TRACK = 250


@pytest.fixture
def keys() -> list[SampleKey]:
    """Mirrors the real dataset: 6 tracks x 250 consecutive frames."""
    return [
        SampleKey(track=t, frame_index=i)
        for t in TRACKS
        for i in range(FRAMES_PER_TRACK)
    ]


@pytest.fixture
def tiny_dataset(tmp_path):
    """A miniature on-disk dataset with the real naming convention."""
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()

    rng = np.random.default_rng(0)
    for track in TRACKS[:3]:
        for i in range(10):
            Image.fromarray(
                rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
            ).save(images / f"{track}_frame_{i:04d}.png")
            Image.fromarray(
                rng.integers(0, 7, (32, 32), dtype=np.uint8)
            ).save(masks / f"{track}_mask_combined_{i:04d}.png")
    return tmp_path
