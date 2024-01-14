from torch import Tensor
from torch import float as t_float
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassRecall, MulticlassPrecision, MulticlassF1Score

def mask_to_int_class(mask: Tensor) -> Tensor:
    return mask.armgax(dim=(1), keepdim=True)

def confusion_matrix(
    pred: Tensor, 
    true: Tensor, 
    num_classes: int = 8
) -> None:
    metric = MulticlassConfusionMatrix(num_classes=num_classes)
    metric(pred, true)
    metric.plot()

def recall(
    pred: Tensor, 
    true: Tensor, 
    num_classes: int = 8,
    average: str = "micro"
) -> float:
    metric = MulticlassRecall(num_classes=num_classes, average=average)
    return metric(pred, true)

def precision(
    pred: Tensor, 
    true: Tensor, 
    num_classes: int = 8,
    average: str = "micro"
) -> float:
    metric = MulticlassPrecision(num_classes=num_classes, average=average)
    return metric(pred, true)

def f1_score(
    pred: Tensor, 
    true: Tensor, 
    num_classes: int = 8,
    average: str = "micro"
) -> float:
    metric = MulticlassF1Score(num_classes=num_classes, average=average)
    return metric(pred, true)

def mean_pixel_acc(
    pred: Tensor, 
    true: Tensor
) -> t_float:
    return (pred == true).to(t_float).mean()

def iou(
    pred_masks: Tensor, 
    true_masks: Tensor, 
) -> t_float:
    pred_masks = pred_masks.to(bool)
    true_masks = true_masks.to(bool)
    intersection = (pred_masks == true_masks).sum()
    union = (pred_masks| true_masks).sum()
    return intersection / union



