import math
from typing import Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d

def rescale_conv(conv, reference):
    """
    Rescale a convolutional module with `reference`.
    """
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale

def rescale_module(module, reference):
    """
    Rescale a module with `reference`.
    """
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)

def center_trim(tensor, reference):
    """
    Trim a tensor to match with the dimension of `reference`.
    """
    if hasattr(reference, "size"):
        reference = reference.size(-1)
    diff = tensor.size(-1) - reference
    if diff < 0:
        raise ValueError("tensor must be larger than reference")
    if diff:
        tensor = tensor[..., diff // 2:-(diff - diff // 2)]
    return tensor

def left_trim(tensor, reference):
    """
    Trim a tensor to match with the dimension of `reference`. Trims only the end.
    """
    if hasattr(reference, "size"):
        reference = reference.size(-1)
    diff = tensor.size(-1) - reference
    if diff < 0:
        raise ValueError("tensor must be larger than reference")
    if diff:
        tensor = tensor[..., 0:-diff]
    return tensor

def upsample(x, stride):
    """
    Bi-linearly upsample a tensor with a given stride.
    """
    batch, channels, time = x.size()
    weight = torch.arange(stride, device=x.device, dtype=torch.float) / stride
    x = x.view(batch, channels, time, 1)
    out = x[..., :-1, :] * (1 - weight) + x[..., 1:, :] * weight
    return out.reshape(batch, channels, -1)

def downsample(x, stride):
    """
    Downsample a tensor with a given stride.
    """
    return x[:, :, ::stride]

class Demucs(nn.Module):
    """
    Demucs network for audio source separation.
    """
    def __init__(self,
                 sources: int = 2,
                 n_audio_channels: int = 2, # pylint: disable=redefined-outer-name
                 kernel_size: int = 8,
                 stride: int = 4,
                 context: int = 3,
                 depth: int = 6,
                 channels: int = 64,
                 growth: float = 2.0,
                 lstm_layers: int = 2,
                 rescale: float = 0.1,
                 upsample: bool = False): # pylint: disable=redefined-outer-name
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

        self.encoder = nn.ModuleList() # Source encoder
        self.decoder = nn.ModuleList() # Audio output decoder

        self.final = None

        if upsample:
            self.final = nn.Conv1d(channels + n_audio_channels, sources * n_audio_channels, 1)
            stride = 1

        activation = nn.GLU(dim=1)

        in_channels = n_audio_channels          # Number of input channels

        # Wave U-Net structure
        for index in range(depth):
            encode = []
            encode += [nn.Conv1d(in_channels, channels, kernel_size, stride), nn.ReLU()]
            encode += [nn.Conv1d(channels, 2 * channels, 1), activation]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            if index > 0:
                out_channels = in_channels
            else:
                if upsample:
                    out_channels = channels
                else:
                    out_channels = sources * n_audio_channels

            decode += [nn.Conv1d(channels, 2 * channels, context), activation]

            if upsample:
                decode += [nn.Conv1d(channels, out_channels, kernel_size, stride=1)]
            else:
                decode += [nn.ConvTranspose1d(channels, out_channels, kernel_size, stride)]

            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))

            in_channels = channels
            channels = int(growth * channels)

        # Bi-directional LSTM for the bottleneck layer
        channels = in_channels
        self.lstm = nn.LSTM(bidirectional=True, num_layers=lstm_layers, hidden_size=channels, input_size=channels)
        self.lstm_linear = nn.Linear(2*channels, channels)

        rescale_module(self, reference=rescale)

    def forward(self, mix: torch.Tensor, angle_conditioning: torch.Tensor): # pylint: disable=arguments-differ
        """
        Forward pass. Note that in our current work the use of `locs` is disregarded.

        Args:
            mix (torch.Tensor) - An input recording of size `(batch_size, n_mics, time)`.

        Output:
            x, locs (Tuple[torch.Tensor, torch.Tensor]) where
            `x` is a source separated output at every microphone and `locs` is the corresponding
            location. `x` has dimension of `(batch_size, n_sources, n_mics, time)`, and `locs` has
            dimension of `(batch_size, n_sources, 3, time)`.
        """
        x = mix
        saved = [x]

        # Encoder
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)
            if self.upsample:
                x = downsample(x, self.stride)


        # Bi-directional LSTM at the bottleneck layer
        x = x.permute(2, 0, 1) # prep input for LSTM
        self.lstm.flatten_parameters() # to improve memory usage.
        x = self.lstm(x)[0]
        x = self.lstm_linear(x)
        x = x.permute(1, 2, 0)

        # Source decoder
        for decode in self.decoder:
            if self.upsample:
                x = upsample(x, stride=self.stride)
            skip = center_trim(saved.pop(-1), x)
            x = x + skip
            x = decode(x)


        if self.final:
            skip = center_trim(saved.pop(-1), x)
            x = torch.cat([x, skip], dim=1)
            x = self.final(x)

        # Reformat the output
        x = x.view(x.size(0), self.sources, self.n_audio_channels, x.size(-1))

        return x

    def loss(self,
             input_signal: torch.Tensor,
             voice_signals: torch.Tensor,
             label_voice_signals: torch.Tensor,
             bg_signals: torch.Tensor,
             label_bg_signals: torch.Tensor,
             voice_locs: torch.Tensor, # pylint: disable=unused-argument
             label_voice_locs: torch.Tensor, # pylint: disable=unused-argument
             bg_locs: torch.Tensor, # pylint: disable=unused-argument
             label_bg_locs: torch.Tensor) -> Tuple[torch.Tensor, Any]:      # pylint: disable=unused-argument
        """
        Compute the loss function.

        Args:
            input_signal (torch.Tensor) - an input signal of size `(batch_size, n_mics, time)`.
            voice_signals (torch.Tensor) - an output voice of size `(batch_size, n_sources, n_mics, time)`.
            label_voice_signals (torch.Tensor) a ground truth voice of size the same as `voice_signals`.
            bg_signals (torch.Tensor) - an output background of size `(batch_size, n_sources, n_mics, time)`.
            label_bg_signals (torch.Tensor) - a ground truth background of size the same as `bg_signals`.
            voice_locs (torch.Tensor) - an output location for voice of size `(batch_size, n_sources, 3, time)`.
            label_voice_locs (torch.Tensor) - a ground truth location for voice of size the same as `voice_locs`.
            bg_locs (torch.Tensor) - an output location for background of size `(batch_size, n_sources, 3, time)`.
            label_bg_locs (torch.Tensor) - a ground truth location for background of size the same as `bg_locs`.

        Output:
            loss_val (torch.Tensor) - loss tensor.
            info (Dict[str, torch.Tensor]) - information regarding each individual loss.
        """
        loss_val = 0

        reconstruction_voices_loss = 0
        reconstruction_bg_loss = 0
        reconstruction_combined_loss = 0

        reconstruction_voices_loss += F.l1_loss(voice_signals, label_voice_signals)
        reconstruction_bg_loss += F.l1_loss(bg_signals, label_bg_signals)

        combined_signal = torch.zeros_like(input_signal)
        combined_signal += torch.sum(voice_signals, dim=1)
        combined_signal += torch.sum(bg_signals, dim=1)
        reconstruction_combined_loss += F.l1_loss(input_signal, combined_signal)

        info = {
            'reconstruction_voices_loss': reconstruction_voices_loss,
            'reconstruction_bg_loss': reconstruction_bg_loss,
            'reconstruction_combined_loss': reconstruction_combined_loss,
        }
        loss_val = reconstruction_voices_loss + \
            reconstruction_bg_loss

        return loss_val, info

    def voice_loss(self, voice_signals, gt_voice_signals):
        reconstruction_voices_loss = F.l1_loss(voice_signals, gt_voice_signals)

        info = {
            'reconstruction_voices_loss': reconstruction_voices_loss,
        }

        return reconstruction_voices_loss, info

    def shift_pred_loss(self, pred_locs, gt_locs):
        # gt_locs = gt_locs.detach().cpu().numpy()
        # gt_locs_smoothed = gaussian_filter1d(gt_locs, sigma=3, axis=1)
        # gt_locs_smoothed = torch.Tensor(gt_locs_smoothed).cuda()
        loss_val = F.cross_entropy(pred_locs, torch.argmax(gt_locs, 1) , reduction='elementwise_mean')
        # shift_pred_loss = F.binary_cross_entropy_with_logits(pred_locs, gt_locs_smoothed)

        print("GT: {} Pred: {}".format(np.argmax(gt_locs.detach().cpu().numpy(), axis=1),
            np.argmax(pred_locs.detach().cpu().numpy(), axis=1)))
        # info = {
        #     'shift_pred_loss': shift_pred_loss,
        # }

        # return shift_pred_loss, info
        return loss_val, None

    def valid_length(self, length: int) -> int: # pylint: disable=redefined-outer-name
        """
        Find the length of the input to the network such that the output's length is
        equal to the given `length`.
        """
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

def load_pretrain(model, state_dict): # pylint: disable=redefined-outer-name
    """
    This code load the Demucs model's weight with the actual weight available online.
    Used only for pre-training. The code here is generated.
    """
    loaded_keys = {}
    loaded_keys['decoder.0.0.bias'  ] = 'decoder.0.0.bias'   # torch.Size([4096])
    loaded_keys['decoder.0.0.weight'] = 'decoder.0.0.weight' # torch.Size([4096, 2048, 3])
    loaded_keys['decoder.0.2.bias'  ] = 'decoder.0.2.bias'   # torch.Size([1024])
    loaded_keys['decoder.0.2.weight'] = 'decoder.0.2.weight' # torch.Size([2048, 1024, 8])
    loaded_keys['decoder.1.0.bias'  ] = 'decoder.1.0.bias'   # torch.Size([2048])
    loaded_keys['decoder.1.0.weight'] = 'decoder.1.0.weight' # torch.Size([2048, 1024, 3])
    loaded_keys['decoder.1.2.bias'  ] = 'decoder.1.2.bias'   # torch.Size([512])
    loaded_keys['decoder.1.2.weight'] = 'decoder.1.2.weight' # torch.Size([1024, 512, 8])
    loaded_keys['decoder.2.0.bias'  ] = 'decoder.2.0.bias'   # torch.Size([1024])
    loaded_keys['decoder.2.0.weight'] = 'decoder.2.0.weight' # torch.Size([1024, 512, 3])
    loaded_keys['decoder.2.2.bias'  ] = 'decoder.2.2.bias'   # torch.Size([256])
    loaded_keys['decoder.2.2.weight'] = 'decoder.2.2.weight' # torch.Size([512, 256, 8])
    loaded_keys['decoder.3.0.bias'  ] = 'decoder.3.0.bias'   # torch.Size([512])
    loaded_keys['decoder.3.0.weight'] = 'decoder.3.0.weight' # torch.Size([512, 256, 3])
    loaded_keys['decoder.3.2.bias'  ] = 'decoder.3.2.bias'   # torch.Size([128])
    loaded_keys['decoder.3.2.weight'] = 'decoder.3.2.weight' # torch.Size([256, 128, 8])
    loaded_keys['decoder.4.0.bias'  ] = 'decoder.4.0.bias'   # torch.Size([256])
    loaded_keys['decoder.4.0.weight'] = 'decoder.4.0.weight' # torch.Size([256, 128, 3])
    loaded_keys['decoder.4.2.bias'  ] = 'decoder.4.2.bias'   # torch.Size([64])
    loaded_keys['decoder.4.2.weight'] = 'decoder.4.2.weight' # torch.Size([128, 64, 8])
    loaded_keys['decoder.5.0.bias'  ] = 'decoder.5.0.bias'   # torch.Size([128])
    loaded_keys['decoder.5.0.weight'] = 'decoder.5.0.weight' # torch.Size([128, 64, 3])
    # loaded_keys['decoder.5.2.bias'  ] = 'decoder.5.2.bias'   # torch.Size([8])          # Output layer
    # loaded_keys['decoder.5.2.weight'] = 'decoder.5.2.weight' # torch.Size([64, 8, 8])   # Output layer
    # loaded_keys['encoder.0.0.bias'  ] = 'encoder.0.0.bias'   # torch.Size([64])         # Input layer
    # loaded_keys['encoder.0.0.weight'] = 'encoder.0.0.weight' # torch.Size([64, 2, 8])   # Input layer
    loaded_keys['encoder.0.2.bias'  ] = 'encoder.0.2.bias'   # torch.Size([128])
    loaded_keys['encoder.0.2.weight'] = 'encoder.0.2.weight' # torch.Size([128, 64, 1])
    loaded_keys['encoder.1.0.bias'  ] = 'encoder.1.0.bias'   # torch.Size([128])
    loaded_keys['encoder.1.0.weight'] = 'encoder.1.0.weight' # torch.Size([128, 64, 8])
    loaded_keys['encoder.1.2.bias'  ] = 'encoder.1.2.bias'   # torch.Size([256])
    loaded_keys['encoder.1.2.weight'] = 'encoder.1.2.weight' # torch.Size([256, 128, 1])
    loaded_keys['encoder.2.0.bias'  ] = 'encoder.2.0.bias'   # torch.Size([256])
    loaded_keys['encoder.2.0.weight'] = 'encoder.2.0.weight' # torch.Size([256, 128, 8])
    loaded_keys['encoder.2.2.bias'  ] = 'encoder.2.2.bias'   # torch.Size([512])
    loaded_keys['encoder.2.2.weight'] = 'encoder.2.2.weight' # torch.Size([512, 256, 1])
    loaded_keys['encoder.3.0.bias'  ] = 'encoder.3.0.bias'   # torch.Size([512])
    loaded_keys['encoder.3.0.weight'] = 'encoder.3.0.weight' # torch.Size([512, 256, 8])
    loaded_keys['encoder.3.2.bias'  ] = 'encoder.3.2.bias'   # torch.Size([1024])
    loaded_keys['encoder.3.2.weight'] = 'encoder.3.2.weight' # torch.Size([1024, 512, 1])
    loaded_keys['encoder.4.0.bias'  ] = 'encoder.4.0.bias'   # torch.Size([1024])
    loaded_keys['encoder.4.0.weight'] = 'encoder.4.0.weight' # torch.Size([1024, 512, 8])
    loaded_keys['encoder.4.2.bias'  ] = 'encoder.4.2.bias'   # torch.Size([2048])
    loaded_keys['encoder.4.2.weight'] = 'encoder.4.2.weight' # torch.Size([2048, 1024, 1])
    loaded_keys['encoder.5.0.bias'  ] = 'encoder.5.0.bias'   # torch.Size([2048])
    loaded_keys['encoder.5.0.weight'] = 'encoder.5.0.weight' # torch.Size([2048, 1024, 8])
    loaded_keys['encoder.5.2.bias'  ] = 'encoder.5.2.bias'   # torch.Size([4096])
    loaded_keys['encoder.5.2.weight'] = 'encoder.5.2.weight' # torch.Size([4096, 2048, 1])
    loaded_keys['lstm_linear.bias'                ] = 'lstm.linear.bias'                 # torch.Size([2048])
    loaded_keys['lstm_linear.weight'              ] = 'lstm.linear.weight'               # torch.Size([2048, 4096])
    loaded_keys['lstm.bias_hh_l0'                 ] = 'lstm.lstm.bias_hh_l0'             # torch.Size([8192])
    loaded_keys['lstm.bias_hh_l0_reverse'         ] = 'lstm.lstm.bias_hh_l0_reverse'     # torch.Size([8192])
    loaded_keys['lstm.bias_hh_l1'                 ] = 'lstm.lstm.bias_hh_l1'             # torch.Size([8192])
    loaded_keys['lstm.bias_hh_l1_reverse'         ] = 'lstm.lstm.bias_hh_l1_reverse'     # torch.Size([8192])
    loaded_keys['lstm.bias_ih_l0'                 ] = 'lstm.lstm.bias_ih_l0'             # torch.Size([8192])
    loaded_keys['lstm.bias_ih_l0_reverse'         ] = 'lstm.lstm.bias_ih_l0_reverse'     # torch.Size([8192])
    loaded_keys['lstm.bias_ih_l1'                 ] = 'lstm.lstm.bias_ih_l1'             # torch.Size([8192])
    loaded_keys['lstm.bias_ih_l1_reverse'         ] = 'lstm.lstm.bias_ih_l1_reverse'     # torch.Size([8192])
    loaded_keys['lstm.weight_hh_l0'               ] = 'lstm.lstm.weight_hh_l0'           # torch.Size([8192, 2048])
    loaded_keys['lstm.weight_hh_l0_reverse'       ] = 'lstm.lstm.weight_hh_l0_reverse'   # torch.Size([8192, 2048])
    loaded_keys['lstm.weight_hh_l1'               ] = 'lstm.lstm.weight_hh_l1'           # torch.Size([8192, 2048])
    loaded_keys['lstm.weight_hh_l1_reverse'       ] = 'lstm.lstm.weight_hh_l1_reverse'   # torch.Size([8192, 2048])
    loaded_keys['lstm.weight_ih_l0'               ] = 'lstm.lstm.weight_ih_l0'           # torch.Size([8192, 2048])
    loaded_keys['lstm.weight_ih_l0_reverse'       ] = 'lstm.lstm.weight_ih_l0_reverse'   # torch.Size([8192, 2048])
    loaded_keys['lstm.weight_ih_l1'               ] = 'lstm.lstm.weight_ih_l1'           # torch.Size([8192, 4096])
    loaded_keys['lstm.weight_ih_l1_reverse'       ] = 'lstm.lstm.weight_ih_l1_reverse'   # torch.Size([8192, 4096])
    for key in loaded_keys:
        try:
            _ = model.load_state_dict({key: state_dict[loaded_keys[key]]}, strict=False)
            print("Load {} (shape = {}) from the pretrained model".format(key, state_dict[loaded_keys[key]].shape))
        except:
            print("Failed to load {}".format(key))
            pass

if __name__ == '__main__':
    # Sample network with dummy input
    n_sources = 2
    n_audio_channels = 8
    length = 2**16
    model = Demucs(sources=n_sources, n_audio_channels=n_audio_channels)

    # input
    dummy = torch.zeros((n_audio_channels, length))
    print("input's shape = {}".format(dummy.shape))

    # padded input
    valid_length = model.valid_length(length)
    delta = valid_length - length
    padded = F.pad(dummy, (delta // 2, delta - delta // 2))
    dummy_padded = padded.unsqueeze(0)
    print("padded input's shape = {}".format(dummy_padded.shape))

    # output
    output, locs = model(dummy_padded)
    output_trimmed = center_trim(output, dummy)
    locs_trimmed = center_trim(locs, dummy)
    print("output's shape = {}".format(output_trimmed.shape))
    print("locs' shape = {}".format(locs_trimmed.shape))
