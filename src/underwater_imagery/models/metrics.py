from torch import Tensor, zeros
from torch.utils.data import DataLoader
from torch import nn
from torch import float as t_float
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassRecall, MulticlassPrecision, MulticlassF1Score
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from underwater_imagery.data.constants import CLASSES
from underwater_imagery.models.iou_loss import SoftIoULoss

def logits_to_idx_class(logits: Tensor) -> Tensor:
    probs = F.softmax(logits, dim=1)
    pred = probs.argmax(dim=(1), keepdim=True)
    return pred.squeeze(1)

def confusion_matrix(
    pred: Tensor, 
    true: Tensor, 
    num_classes: int = 8
) -> Tensor:
    pred = pred.to('cpu')
    true = true.to('cpu')
    metric = MulticlassConfusionMatrix(num_classes=num_classes)
    matrix = metric(pred, true)
    matrix = matrix / matrix.sum() * 100
    return matrix

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

def eval(
    model: nn.Module, 
    data_loader: DataLoader, 
    num_classes: int = 8,
    device: str = 'cuda',
    max_iter: int = 3,
) -> None:
    precision_total = 0
    recall_total = 0
    f1_total = 0
    mean_pixel_acc_total = 0
    iou_loss_total = 0
    entropy_loss_total = 0
    matrix = zeros((num_classes, num_classes))
    iou_loss = SoftIoULoss()
    criterion = nn.CrossEntropyLoss()
    for i, data in enumerate(data_loader):
        images, label_masks, label_images = data
        images: Tensor = images.to(device)
        label_masks: Tensor = label_masks.to(device)
        label_images: Tensor = label_images.to(device)

        logits = model(images)
        pred = logits_to_idx_class(logits)
        true = logits_to_idx_class(label_masks)

        precision_total += precision(pred, true)
        recall_total += recall(pred, true)
        f1_total += f1_score(pred, true)
        mean_pixel_acc_total += mean_pixel_acc(pred, true)
        iou_loss_total += iou_loss(logits, label_masks)
        entropy_loss_total += criterion(logits, label_masks)
        matrix += confusion_matrix(pred, true)
        if i > max_iter:
            break

    i += 1
    print("always micro")
    print(f"precision: {precision_total / i:.3f}")
    print(f"recall: {recall_total / i:.3f}")
    print(f"f1 score: {f1_total / i:.3f}")
    print(f"mean pixel acc: {mean_pixel_acc_total / i:.3f}")
    print(f"soft log IoU loss: {iou_loss_total / i:.3f}")
    print(f"cross entropy loss: {entropy_loss_total / i:.3f}")

    names = [classes[0] for classes in CLASSES]
    sns.heatmap(matrix / i, annot=True, fmt='.2f', xticklabels=names, yticklabels=names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)


