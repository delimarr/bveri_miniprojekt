from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io as t_io
from torchvision import transforms


class UIMDataset(Dataset):
    def __init__(self, root_path: Path) -> None:
        self.root_path = root_path
        self.train_path = root_path / "train_val"
        self.test_path = root_path / "TEST"

        self.classes = [
            p.parts[-1] for p in self.test_path.glob("./masks/*") if p.is_dir()
        ]

        # build train set
        self.train_images: List[Tensor] = []
        self.train_masks: List[Tensor] = []

        transform = transforms.ToTensor()
        for image in self.train_path.glob("./images/*.jpg"):
            self.train_images.append(t_io.read_image(str(image.absolute())))
        for mask in self.train_path.glob("./masks/*.bmp"):
            image = Image.open(str(mask.absolute()))
            self.train_masks.append(transform(image))

    def __len__(self) -> int:
        return len(self.train_images)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self.train_images[index], self.train_masks[index]
