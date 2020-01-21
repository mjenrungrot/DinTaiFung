import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# from collections import OrderedDict
# def unet(pretrained=False, **kwargs):
#     model = UNet(**kwargs)
#     if pretrained:
#         checkpoint = "https://github.com/mateuszbuda/brain-segmentation-pytorch/releases/download/v1.0/unet-e012d006.pt"
#         state_dict = torch.hub.load_state_dict_from_url(checkpoint, progress=False, map_location='cpu')
#         model.load_state_dict(state_dict, strict=False)
#     return model

# class UNet(nn.Module):
#     def __init__(self, in_channels=2, out_channels=2, init_features=32):
#         super(UNet, self).__init__()

#         features = init_features
#         self.encoder1 = UNet._block(in_channels, features, name="enc1_mod")
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder2 = UNet._block(features, features * 2, name="enc2")
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
#         self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

#         self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
#         self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
#         self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
#         self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
#         self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
#         self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
#         self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
#         self.decoder1 = UNet._block(features * 2, features, name="dec1")

#         self.conv_mod = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

#     def forward(self, x): # pylint: disable=arguments-differ
#         enc1 = self.encoder1(x)
#         enc2 = self.encoder2(self.pool1(enc1))
#         enc3 = self.encoder3(self.pool2(enc2))
#         enc4 = self.encoder4(self.pool3(enc3))

#         bottleneck = self.bottleneck(self.pool4(enc4))

#         dec4 = self.upconv4(bottleneck)
#         dec4 = torch.cat((dec4, enc4), dim=1)
#         dec4 = self.decoder4(dec4)
#         dec3 = self.upconv3(dec4)
#         dec3 = torch.cat((dec3, enc3), dim=1)
#         dec3 = self.decoder3(dec3)
#         dec2 = self.upconv2(dec3)
#         dec2 = torch.cat((dec2, enc2), dim=1)
#         dec2 = self.decoder2(dec2)
#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat((dec1, enc1), dim=1)
#         dec1 = self.decoder1(dec1)

#         return torch.sigmoid(self.conv_mod(dec1))

#     def loss(self, prediction, label, reduction="elementwise_mean"):
#         loss_val = F.binary_cross_entropy(prediction, label, reduction=reduction)
#         return loss_val

#     @staticmethod
#     def _block(in_channels, features, name):
#         return nn.Sequential(OrderedDict(
#             [
#                 (
#                     name + "conv1",
#                     nn.Conv2d(
#                         in_channels=in_channels,
#                         out_channels=features,
#                         kernel_size=3,
#                         padding=1,
#                         bias=False,
#                     ),
#                 ),
#                 (name + "norm1", nn.BatchNorm2d(num_features=features)),
#                 (name + "relu1", nn.ReLU(inplace=True)),
#                 (
#                     name + "conv2",
#                     nn.Conv2d(
#                         in_channels=features,
#                         out_channels=features,
#                         kernel_size=3,
#                         padding=1,
#                         bias=False,
#                     ),
#                 ),
#                 (name + "norm2", nn.BatchNorm2d(num_features=features)),
#                 (name + "relu2", nn.ReLU(inplace=True)),
#             ]
#         )
#                              )

class BLSTM(nn.Module):
    def __init__(self, dim, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = nn.Linear(2*dim, dim)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1, 2, 0)
        return x

def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale

def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)

def center_trim(tensor, reference):
    if hasattr(reference, "size"):
        reference = reference.size(-1)
    diff = tensor.size(-1) - reference
    if diff < 0:
        raise ValueError("tensor must be larger than reference")
    if diff:
        tensor = tensor[..., diff // 2:-(diff - diff // 2)]
    return tensor

def upsample(x, stride):
    batch, channels, time = x.size()
    weight = torch.arange(stride, device=x.device, dtype=torch.float) / stride
    x = x.view(batch, channels, time, 1)
    out = x[..., :-1, :] * (1 - weight) + x[..., 1:, :] * weight
    return out.reshape(batch, channels, -1)

def downsample(x, stride):
    return x[:, :, ::stride]

class Demucs(nn.Module):
    def __init__(self, sources=2, 
                 n_audio_channels=2,
                 kernel_size=8,
                 stride=4,
                 context=3,
                 depth=6,
                 channels=64,
                 growth=2.0,
                 lstm_layers=2,
                 rescale=0.1,
                 upsample=False): # pylint: disable=redefined-outer-name
        super().__init__()
        self.sources = sources
        self.n_audio_channels = n_audio_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.context = context
        self.depth = depth
        self.channels = channels
        self.growth = growth
        self.lstm_layers = lstm_layers
        self.rescale = rescale
        self.upsample = upsample

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.loc_decoder = nn.ModuleList()

        self.final = None

        if upsample:
            self.final = nn.Conv1d(channels + n_audio_channels, sources * n_audio_channels, 1)
            stride = 1

        activation = nn.GLU(dim=1)
        ch_scale = 2

        in_channels = n_audio_channels
        for index in range(depth):
            encode = []
            encode += [nn.Conv1d(in_channels, channels, kernel_size, stride), nn.ReLU()]
            encode += [nn.Conv1d(channels, ch_scale * channels, 1), activation]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            if index > 0:
                out_channels = in_channels
            else:
                if upsample:
                    out_channels = channels
                else:
                    out_channels = sources * n_audio_channels

            decode += [nn.Conv1d(channels, ch_scale * channels, context), activation]

            if upsample:
                decode += [nn.Conv1d(channels, out_channels, kernel_size, stride=1)]
            else:
                decode += [nn.ConvTranspose1d(channels, out_channels, kernel_size, stride)]

            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))

            loc_decoder = []
            loc_decoder += [nn.ConvTranspose1d(3, 3, kernel_size, stride)]
            if index > 0:
                loc_decoder.append(nn.ReLU())
            self.loc_decoder.insert(0, nn.Sequential(*loc_decoder))

            in_channels = channels
            channels = int(growth * channels)

        channels = in_channels
        self.lstm = nn.LSTM(bidirectional=True, num_layers=lstm_layers, hidden_size=channels, input_size=channels)
        self.lstm_linear = nn.Linear(2*channels, channels)
        self.loc_prediction = nn.Linear(2*channels, 3)

        rescale_module(self, reference=rescale)


    def forward(self, mix):
        x = mix
        saved = [x]
        # print(x.shape)
        for encode in self.encoder:
            x = encode(x)
            print(x.shape)
            saved.append(x)
            if self.upsample:
                x = downsample(x, self.stride)
        # print("Before LSTM", x.shape)

        x = x.permute(2, 0, 1) # prep input for LSTM
        x = self.lstm(x)[0]
        locs = self.loc_prediction(x)
        locs = locs.permute(1, 2, 0)

        x = self.lstm_linear(x)
        x = x.permute(1, 2, 0)

        # print("After LSTM", x.shape)
        # print(locs.shape)
        # print(x.shape)

        for decode in self.decoder:
            if self.upsample:
                x = upsample(x, stride=self.stride)
            skip = center_trim(saved.pop(-1), x)
            x = x + skip
            x = decode(x)

        for loc_decode in self.loc_decoder:
            locs = loc_decode(locs)

        if self.final:
            skip = center_trim(saved.pop(-1), x)
            x = torch.cat([x, skip], dim=1)
            x = self.final(x)

        x = x.view(x.size(0), self.sources, self.n_audio_channels, x.size(-1))
        locs = locs.view(locs.size(0), self.sources, 3, locs.size(-1))
        return x, locs

    def loss(self, prediction, label):
        loss_val = F.l1_loss(prediction, label)
        return loss_val

    def valid_length(self, length): # pylint: disable=redefined-outer-name
        for _ in range(self.depth):
            if self.upsample:
                length = math.ceil(length / self.stride) + self.kernel_size - 1
            else:
                length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)
            length += self.context - 1

        for _ in range(self.depth):
            if self.upsample:
                length = length * self.stride + self.kernel_size - 1
            else:
                length = (length - 1) * self.stride + self.kernel_size

        return int(length)

if __name__ == '__main__':
    model = Demucs(sources=1, # voice
                   n_audio_channels=2, # left/right mics
                   )

    n_sources = 1
    n_audio_channels = 2
    length = 2**16
    dummy = torch.zeros((n_audio_channels, length))
    print("input shape = {}".format(dummy.shape))

    # padd
    valid_length = model.valid_length(length)
    delta = valid_length - length
    padded = F.pad(dummy, (delta // 2, delta - delta // 2))
    dummy_padded = padded.unsqueeze(0)
    print("input padded shape = {}".format(dummy_padded.shape))
    output, locs = model(dummy_padded)
    output_trimmed = center_trim(output, dummy)
    locs_trimmed = center_trim(locs, dummy)
    print(output_trimmed.shape)
    print("output shape = {}".format(output_trimmed.shape))
    print("locs shape = {}".format(locs_trimmed.shape))
