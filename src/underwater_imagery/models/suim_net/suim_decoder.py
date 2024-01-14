from typing import Tuple

from torch import Tensor, cat, nn
from torch.nn import functional as F


class SuimDecoder(nn.Module):
    def __init__(
        self, n_classes: int, channels: int = 256, device: str = "cuda"
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.channels = channels
        self.device = device

        # decoder block 1
        self.conv_1 = nn.Conv2d(
            in_channels=self.channels,
            out_channels=256,
            kernel_size=(3, 3),
            stride=1,
            padding="same",
            device=self.device,
        )
        self.bn_1 = nn.BatchNorm2d(num_features=256, momentum=0.8, device=self.device)

        # decoder block 2
        self.conv_2_1 = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=(3, 3),
            stride=1,
            padding="same",
            device=self.device,
        )
        self.bn_2_1 = nn.BatchNorm2d(num_features=256, momentum=0.8, device=self.device)
        self.conv_2_2 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=(3, 3),
            stride=1,
            padding="same",
            device=self.device,
        )
        self.bn_2_2 = nn.BatchNorm2d(num_features=128, momentum=0.8, device=self.device)

        # decoder block 3
        self.conv_3_1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=(3, 3),
            padding="same",
            device=self.device,
        )
        self.bn_3_1 = nn.BatchNorm2d(num_features=128, momentum=0.8, device=self.device)
        self.conv_3_2 = nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=(3, 3),
            padding="same",
            device=self.device,
        )
        self.bn_3_2 = nn.BatchNorm2d(num_features=64, momentum=0.8, device=self.device)

        # output layer
        self.conv_out = nn.Conv2d(
            in_channels=64,
            out_channels=self.n_classes,
            kernel_size=(3, 3),
            padding="same",
            device=self.device,
        )

    def forward(self, enc_outputs: Tuple[Tensor, ...]) -> Tensor:
        self.enc_1, self.enc_2, self.enc_3 = enc_outputs
        # decoder block 1
        dec_1 = self.conv_1(self.enc_3)
        dec_1 = self.bn_1(dec_1)
        dec_1 = nn.Upsample(scale_factor=2)(dec_1)
        dec_1 = F.pad(dec_1[:, :, :-2, :-2], (1, 1, 1, 1), value=0)
        enc_2 = F.pad(self.enc_2[:, :, :-1, :-1], (1, 1, 1, 1), value=0)
        dec_1s = self.concat_skip(enc_2, dec_1, 256)

        # decoder block 2
        dec_2 = self.conv_2_1(dec_1s)
        dec_2 = self.bn_2_1(dec_2)
        dec_2 = nn.Upsample(scale_factor=2)(dec_2)
        dec_2s = self.conv_2_2(dec_2)
        dec_2s = self.bn_2_2(dec_2s)
        dec_2s = nn.Upsample(scale_factor=2)(dec_2s)
        enc_1 = F.pad(self.enc_1, (2, 2, 2, 2), value=0)
        dec_2s = self.concat_skip(enc_1, dec_2s, 128)

        # decoder block 3
        dec_3 = self.conv_3_1(dec_2s)
        dec_3 = self.bn_3_1(dec_3)
        dec_3s = self.conv_3_2(dec_3)
        dec_3s = self.bn_3_2(dec_3s)

        # output layer
        out = self.conv_out(dec_3s)
        out = F.sigmoid(out)
        return out

    def concat_skip(
        self, layer_input: Tensor, skip_input: Tensor, filters: int, f_size: int = 3
    ) -> Tensor:
        input_conv = nn.Conv2d(
            in_channels=layer_input.shape[1],
            out_channels=filters,
            kernel_size=f_size,
            stride=1,
            padding="same",
            device=self.device,
        )
        input_bn = nn.BatchNorm2d(
            num_features=filters, momentum=0.8, device=self.device
        )
        layer_input = input_conv(layer_input)
        layer_input = F.relu(layer_input)
        layer_input = input_bn(layer_input)
        return cat([layer_input, skip_input], dim=1)
