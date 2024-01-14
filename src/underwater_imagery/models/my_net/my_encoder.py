from torch import Tensor, nn
from torch.nn import functional as F


class EncoderBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, device: str = "cuda"
    ) -> None:
        """Encoder-Block. Containing Conv((3, 3), stride=1), relu, Conv((3, 3), stride=2), relu

        Args:
            in_channels (int): number of input features
            out_channels (int): number of output features
            device (str, optional): "cpu" or "cuda". Defaults to "cuda".
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            device=self.device,
        )
        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=2,
            padding=1,
            device=self.device,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, num_channels: int, device: str = "cuda") -> None:
        super().__init__()
        self.device = device
        self.layers_ = nn.ModuleList()
        for _, (in_channels, out_channels) in enumerate(
            zip(num_channels, num_channels[1:])
        ):
            self.layers_.append(
                EncoderBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    device=self.device,
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers_:
            x = layer(x)
        return x
