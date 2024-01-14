"""
https://discuss.pytorch.org/t/how-to-implement-soft-iou-loss/15152 [14.01.2024]
https://towardsdatascience.com/efficient-image-segmentation-using-pytorch-part-1-89e8297a0923 [14.01.2024]
"""

from torch import Tensor, log
from torch.nn import functional as F
from torch import nn

class SoftIoULoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
    ) -> None:
        super(SoftIoULoss, self).__init__()
        self.num_classes = num_classes

    def forward(
        self,
        pred_masks: Tensor, 
        true_masks: Tensor,
    ) -> float:
        N = pred_masks.shape[0]
        pred_masks = F.softmax(pred_masks, dim=1) 

        intersection = pred_masks * true_masks
        intersection = intersection.view(N,self.num_classes, -1).sum(2)

        union = pred_masks + true_masks - (pred_masks * true_masks)
        union = union.view(N, self.num_classes, -1).sum(2) + 1e-6

        loss = intersection/union
        mean_log_loss = log(loss.mean() + 1e-6)
        return -mean_log_loss
