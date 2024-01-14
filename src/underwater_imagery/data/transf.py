from torch import float as t_float, Tensor, empty, flatten
from torchvision import transforms

from underwater_imagery.data.constants import SHAPE, CLASSES

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

def pred_to_label(pred: Tensor) -> Tensor:
    _, h, w = pred.shape
    pred_label = empty((3, h, w))
    for y in range(h):
        for x in range(w):
            rgb = flatten(CLASSES[pred[0][y][x]][1])
            pred_label[:, y, x] = rgb
    return pred_label
