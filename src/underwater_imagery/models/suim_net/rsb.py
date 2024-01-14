from typing import Tuple

from torch import Tensor, nn
from torch.nn import functional as F


class RSB(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        input_dim: int,
        filters: Tuple[int, int, int, int],
        stride: int = 1,
        skip: bool = True,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.input_dim = input_dim
        self.filters = filters
        self.stride = stride
        self.skip = skip
        self.device = device

        f1, f2, f3, f4 = self.filters

        # sub-block1
        self.conv1 = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=f1,
            kernel_size=(1, 1),
            stride=self.stride,
            padding=0,
            device=self.device,
        )
        self.bn1 = nn.BatchNorm2d(
            num_features=f1,
            momentum=0.8,
            device=self.device,
        )

        # sub-block2
        self.conv2 = nn.Conv2d(
            in_channels=f1,
            out_channels=f2,
            kernel_size=self.kernel_size,
            padding="same",
            device=self.device,
        )
        self.bn2 = nn.BatchNorm2d(num_features=f2, momentum=0.8, device=self.device)

        # sub-block3
        self.conv3 = nn.Conv2d(
            in_channels=f2,
            out_channels=f3,
            kernel_size=(1, 1),
            padding=0,
            device=self.device,
        )
        self.bn3 = nn.BatchNorm2d(num_features=f3, momentum=0.8, device=self.device)

        # skip-block
        self.conv_skip = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=f4,
            kernel_size=(1, 1),
            padding=0,
            stride=self.stride,
            device=self.device,
        )
        self.bn_skip = nn.BatchNorm2d(num_features=f4, momentum=0.8, device=self.device)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x.clone()

        # sub-block1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # sub-block2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # sub-block3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # skip-block
        if not self.skip:
            shortcut = self.conv_skip(shortcut)
            shortcut = self.bn_skip(shortcut)

        x = x + shortcut
        x = F.relu(x)
        return x
