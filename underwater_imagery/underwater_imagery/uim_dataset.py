from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch import Tensor, round, uint8, empty, all
from torch.utils.data import Dataset
from torchvision import io as t_io
from torchvision import transforms


class UIMDataset(Dataset):
    def __init__(self, root_path: Path, classes: List[Tuple[str, Tensor]]) -> None:
        self.root_path = root_path

        self.classes = classes

        self.images: List[Tensor] = []
        self.masks: List[Tensor] = []
        self.label_images: List[Tensor] = []

        transform = transforms.ToTensor()
        for image in self.root_path.glob("./images/*.jpg"):
            self.images.append(t_io.read_image(str(image.absolute())))
        for label_image in self.root_path.glob("./masks/*.bmp"):
            image = Image.open(str(label_image.absolute()))
            image_tensor = transform(image)
            self.label_images.append(round(image_tensor).to(uint8))

            _, w, h = image_tensor.shape
            mask = empty((len(self.classes), w, h), dtype=uint8)
            for i, cls in enumerate(self.classes):
                val = cls[1]
                val = val.view(3, 1, 1)
                mask[i] = all(image_tensor == val, dim=0).to(uint8)
            self.masks.append(mask)

        assert len(self.images) == len(self.label_images)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        return self.images[index], self.masks[index], self.label_images[index]
