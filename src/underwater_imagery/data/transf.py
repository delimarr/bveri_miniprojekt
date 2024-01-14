from torch import Tensor, empty, flatten
from torch import float as t_float
from torch import stack
from torchvision import transforms

from underwater_imagery.data.constants import CLASSES, SHAPE

resize_transf = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(size=SHAPE, antialias=True),
        transforms.ConvertImageDtype(t_float),
    ]
)

resize_normalize_transf = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(size=SHAPE, antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.ConvertImageDtype(t_float),
    ]
)


def pred_to_label(pred: Tensor, device: str = "cuda") -> Tensor:
    colors = stack([color[1].flatten() for color in CLASSES])
    colors = colors.to(device)
    pred_label = colors[pred]
    pred_label = pred_label.squeeze(1)
    pred_label = pred_label.permute(0, 3, 1, 2)
    return pred_label * 255
