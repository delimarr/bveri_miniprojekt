from typing import List, Tuple

from torch import Tensor, cuda, nn

from underwater_imagery.models.suim_net.suim_decoder import SuimDecoder
from underwater_imagery.models.suim_net.suim_encoder import SuimEncoder


class SuimNet(nn.Module):
    def __init__(self, classes: List[Tuple[str, Tensor]], device: str = "cuda") -> None:
        super().__init__()
        if device == "cuda":
            if not cuda.is_available():
                device = "cpu"
                print("no cuda device found, set device to cpu")
        self.device = device
        self.classes = classes
        self.encoder = SuimEncoder(device=self.device)
        self.decoder = SuimDecoder(n_classes=len(self.classes), device=self.device)

    def forward(self, x: Tensor) -> Tensor:
        enc_output = self.encoder.forward(x)
        return self.decoder.forward(enc_output)
