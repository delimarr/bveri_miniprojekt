from torch import Tensor
from torch import float as t_float
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassRecall, MulticlassPrecision, MulticlassF1Score
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from underwater_imagery.data.constants import CLASSES

def logits_to_idx_class(logits: Tensor) -> Tensor:
    probs = F.softmax(logits, dim=1)
    pred = probs.argmax(dim=(1), keepdim=True)
    return pred.squeeze(1)

def confusion_matrix(
    pred: Tensor, 
    true: Tensor, 
    num_classes: int = 8
) -> None:
    pred = pred.to('cpu')
    true = true.to('cpu')
    metric = MulticlassConfusionMatrix(num_classes=num_classes)
    matrix = metric(pred, true)
    matrix = matrix / matrix.sum() * 100
    names = [classes[0] for classes in CLASSES]
    sns.heatmap(matrix, annot=True, fmt='.2f', xticklabels=names, yticklabels=names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)

def recall(
    pred: Tensor, 
    true: Tensor, 
    num_classes: int = 8,
    average: str = "micro"
) -> float:
    pred = pred.to('cpu')
    true = true.to('cpu')
    metric = MulticlassRecall(num_classes=num_classes, average=average)
    return metric(pred, true)

def precision(
    pred: Tensor, 
    true: Tensor, 
    num_classes: int = 8,
    average: str = "micro"
) -> float:
    pred = pred.to('cpu')
    true = true.to('cpu')
    metric = MulticlassPrecision(num_classes=num_classes, average=average)
    return metric(pred, true)

def f1_score(
    pred: Tensor, 
    true: Tensor, 
    num_classes: int = 8,
    average: str = "micro"
) -> float:
    pred = pred.to('cpu')
    true = true.to('cpu')
    metric = MulticlassF1Score(num_classes=num_classes, average=average)
    return metric(pred, true)

def mean_pixel_acc(
    pred: Tensor, 
    true: Tensor
) -> t_float:
    pred = pred.to('cpu')
    true = true.to('cpu')
    return (pred == true).to(t_float).mean()

def iou(
    pred_masks: Tensor, 
    true_masks: Tensor, 
) -> t_float:
    pred_masks = pred_masks.to('cpu')
    true_masks = true_masks.to('cpu')
    pred_masks = pred_masks.to(bool)
    true_masks = true_masks.to(bool)
    intersection = (pred_masks == true_masks).sum()
    union = (pred_masks | true_masks).sum()
    return intersection / union



