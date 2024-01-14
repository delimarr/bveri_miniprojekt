from pathlib import Path
from typing import Callable, List, Tuple

from PIL import Image
from torch import Tensor, all, empty
from torch import float as t_float
from torch.utils.data import Dataset


class UIMDataset(Dataset):
    def __init__(
        self,
        root_path: Path,
        classes: List[Tuple[str, Tensor]],
        image_size: Tuple[int, int],
        transf: Callable,
        transf_labels: Callable,
    ) -> None:
        self.root_path = root_path
        self.classes = classes
        self.image_size = image_size
        self.transf = transf
        self.transf_labels = transf_labels

        w, h = self.image_size
        self.size = len(list(self.root_path.glob("./images/*.jpg")))
        self.images = empty((self.size, 3, w, h))
        self.masks = empty((self.size, len(self.classes), w, h))
        self.label_images = empty((self.size, 3, w, h))

        for i, image_path in enumerate(self.root_path.glob("./images/*.jpg")):
            image = Image.open(str(image_path.absolute()))
            self.images[i] = self.transf(image)

        for i, label_path in enumerate(self.root_path.glob("./masks/*.bmp")):
            image = Image.open(str(label_path.absolute()))
            label_tensor = self.transf_labels(image)
            self.label_images[i] = label_tensor

            mask = empty((len(self.classes), w, h), dtype=t_float)
            for j, cls in enumerate(self.classes):
                val = cls[1]
                mask[j] = all(label_tensor == val, dim=0).to(t_float)
            self.masks[i] = mask

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        return self.images[index], self.masks[index], self.label_images[index]
