from typing import Tuple

from torch import Tensor, nn
from torch.nn import functional as F

from underwater_imagery.models.suim_net.rsb import RSB


class SuimEncoder(nn.Module):
    def __init__(self, channels: int = 3, device: str = "cuda") -> None:
        super().__init__()
        self.channels = channels
        self.device = device

        # encoder block 1
        self.conv_1 = nn.Conv2d(
            in_channels=self.channels,
            out_channels=64,
            kernel_size=(5, 5),
            stride=1,
            device=self.device,
        )

        # encoder block 2
        self.bn_2 = nn.BatchNorm2d(num_features=64, momentum=0.8, device=self.device)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.filters_2 = [64, 64, 128, 128]
        self.rsb_2_1 = RSB(
            kernel_size=3,
            input_dim=64,
            filters=self.filters_2,
            stride=2,
            skip=False,
            device=self.device,
        )
        self.rsb_2_2_1 = RSB(
            kernel_size=3,
            input_dim=128,
            filters=self.filters_2,
            skip=True,
            device=self.device,
        )
        self.rsb_2_2_2 = RSB(
            kernel_size=3,
            input_dim=128,
            filters=self.filters_2,
            skip=True,
            device=self.device,
        )

        # encoder block 3
        self.filters_3 = [128, 128, 256, 256]
        self.rsb_3_1 = RSB(
            kernel_size=3,
            input_dim=128,
            filters=self.filters_3,
            stride=2,
            skip=False,
            device=self.device,
        )
        self.rsb_3_2_1 = RSB(
            kernel_size=3,
            input_dim=256,
            filters=self.filters_3,
            skip=True,
            device=self.device,
        )
        self.rsb_3_2_2 = RSB(
            kernel_size=3,
            input_dim=256,
            filters=self.filters_3,
            skip=True,
            device=self.device,
        )
        self.rsb_3_2_3 = RSB(
            kernel_size=3,
            input_dim=256,
            filters=self.filters_3,
            skip=True,
            device=self.device,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        # encoder block 1
        x = self.conv_1(x)
        enc_1 = x.clone().requires_grad_(True)

        # encoder block 2
        x = self.bn_2(x)
        x = F.relu(x)
        x = self.maxpool_2(x)
        x = self.rsb_2_1.forward(x)
        x = self.rsb_2_2_1.forward(x)
        x = self.rsb_2_2_2.forward(x)
        enc_2 = x.clone().requires_grad_(True)

        # encoder block 3
        x = self.rsb_3_1.forward(x)
        x = self.rsb_3_2_1.forward(x)
        x = self.rsb_3_2_2.forward(x)
        x = self.rsb_3_2_3.forward(x)

        return enc_1, enc_2, x
