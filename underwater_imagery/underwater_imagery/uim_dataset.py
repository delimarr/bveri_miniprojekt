from pathlib import Path
from typing import Tuple, Callable, List

from PIL import Image
from torch import Tensor, tensor, empty, all
from torch.utils.data import Dataset
from torchvision import transforms
from torch import float as t_float


class UIMDataset(Dataset):
    def __init__(
        self, 
        root_path: Path, 
        classes: List[Tuple[str, Tensor]],
        transform: Callable = None,
        transform_labels: Callable = None,
        w: int = 200,
        h: int = 300
    ) -> None:
        self.root_path = root_path
        self.classes = classes

        self.w = w
        self.h = h

        self.default_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop((self.w, self.h)),
            transforms.ConvertImageDtype(t_float)
        ])

        self.transform = self.default_transform
        self.transfrom_labels = self.default_transform
        if transform:
            self.transform = transform
        if transform_labels:
            self.transfrom_labels = transform_labels

        self.size = len(list(self.root_path.glob("./images/*.jpg")))
        self.images = empty((self.size, 3, self.w, self.h), device='cuda')
        self.masks = empty((self.size, len(self.classes), self.w, self.h), device='cuda')
        self.label_images = empty((self.size, 3, self.w, self.h), device='cuda')

        for i, image_file in enumerate(self.root_path.glob("./images/*.jpg")):
            image = Image.open(str(image_file.absolute()))
            self.images[i] = self.transform(image)

        for i, label_image in enumerate(self.root_path.glob("./masks/*.bmp")):
            image = Image.open(str(label_image.absolute()))
            label_tensor = self.transfrom_labels(image).to('cuda')
            self.label_images[i] = label_tensor

            mask = empty((len(self.classes), self.w, self.h), dtype=t_float, device='cuda')
            for j, cls in enumerate(self.classes):
                val = cls[1]
                val = val.view(3, 1, 1).to('cuda')
                mask[j] = all(label_tensor == val, dim=0).to(t_float)
            self.masks[i] = mask

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        return self.images[index], self.masks[index], self.label_images[index]
