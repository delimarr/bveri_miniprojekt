"""Decoder."""
from torch import Tensor, nn
from torch.nn import functional as F


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, device: str = "cuda"
    ) -> None:
        """Decoder-block containing: conv((3, 3), stride=1), relu, conv((3, 3), stride=2), relu.

        Args:
            in_channels (int): number of inputfeatures
            out_channels (int): number of output features
            device (str, optional): "cpu" or "cuda". Defaults to "cuda".
        """
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            device=self.device,
        )
        # Transposed Convolution Layer
        self.conv2 = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=2,
            output_padding=1,
            padding=1,
            device=self.device,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_channels: int, device: str = "cuda") -> None:
        super().__init__()
        self.device = device
        self.layers_ = nn.ModuleList()
        for _, (in_channels, out_channels) in enumerate(
            zip(num_channels, num_channels[1:])
        ):
            self.layers_.append(
                DecoderBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    device=self.device,
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers_:
            x = layer(x)
        return x
