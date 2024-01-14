from torch import Tensor, nn
from torch.nn import functional as F

from underwater_imagery.models.my_net.my_decoder import Decoder
from underwater_imagery.models.my_net.my_encoder import Encoder


class MyModel(nn.Module):
    """Encoder-Decoder"""

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        num_initial_channels: int,
        num_input_channels: int,
        num_output_channels: int,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.device = device
        self.input = nn.Conv2d(
            3,
            num_initial_channels,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            device=self.device,
        )
        self.encoder = encoder
        self.decoder = decoder
        self.output = nn.Conv2d(
            num_input_channels,
            num_output_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            device=self.device,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output(x)
        return x
