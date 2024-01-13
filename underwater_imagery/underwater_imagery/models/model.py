from torch import nn
from torch.nn import functional as F

class EncoderDecoder(nn.Module):
    """ Encoder-Decoder """
    def __init__(self, encoder, decoder, num_initial_channels, num_input_channels, num_output_channels):
        super().__init__()
        self.input = nn.Conv2d(3, num_initial_channels, kernel_size=(3, 3), stride=1, padding=1, device='cuda')
        self.encoder = encoder
        self.decoder = decoder
        self.output = nn.Conv2d(num_input_channels,  num_output_channels, kernel_size=(1, 1), stride=1, padding=0, device='cuda')

    def forward(self, x):
        x = self.input(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=1, padding=1, device='cuda')
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=2, padding=1, device='cuda')


    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=1, padding=1, device='cuda')
        # Transposed Convolution Layer
        self.conv2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3, 3), stride=2, output_padding=1, padding=1, device='cuda')

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x

class Encoder(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.layers_ = nn.ModuleList()
        for _, (in_channels, out_channels) in enumerate(zip(num_channels, num_channels[1:])):
            self.layers_.append(EncoderBlock(in_channels=in_channels, out_channels=out_channels))
            
    def forward(self, x):
        for layer in self.layers_:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.layers_ = nn.ModuleList()
        for _, (in_channels, out_channels) in enumerate(zip(num_channels, num_channels[1:])):
            self.layers_.append(DecoderBlock(in_channels=in_channels, out_channels=out_channels))

    def forward(self, x):
        for layer in self.layers_:
            x = layer(x)
        return x