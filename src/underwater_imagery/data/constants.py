from pathlib import Path

from torch import tensor, uint8

SHAPE = (240, 320)
# SHAPE = (480, 640)

TRAIN_PATH = Path("./data/train_val/")
TEST_PATH = Path("./data/TEST/")

CLASSES = [
    ("BW", tensor([0, 0, 0], dtype=uint8).view(3, 1, 1)),
    ("HD", tensor([0, 0, 1], dtype=uint8).view(3, 1, 1)),
    ("PF", tensor([0, 1, 0], dtype=uint8).view(3, 1, 1)),
    ("WR", tensor([0, 1, 1], dtype=uint8).view(3, 1, 1)),
    ("RO", tensor([1, 0, 0], dtype=uint8).view(3, 1, 1)),
    ("RI", tensor([1, 0, 1], dtype=uint8).view(3, 1, 1)),
    ("FV", tensor([1, 1, 0], dtype=uint8).view(3, 1, 1)),
    ("SR", tensor([1, 1, 1], dtype=uint8).view(3, 1, 1)),
]
