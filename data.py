from pathlib import Path
import os
import json
import torch
import cv2
import librosa
import numpy as np
from scipy.io import wavfile

GLOBAL_SAMPLE_RATE = 22050
NUM_BGS = 20
DIM_DIVISOR = 32  # To go through UNet, must divide this dim
USE_CUDA = True
BATCH_SIZE = 128
TEST_BATCH_SIZE = 128

def save_mask(masks, directory):
    import matplotlib # pylint: disable=import-outside-toplevel
    matplotlib.use("Agg")

    for mask_idx in range(masks.shape[0]):
        mask = masks[mask_idx, :, :]
        cv2.imwrite(os.path.join(directory, "mask{:02d}.png".format(mask_idx)),
        (mask*255).astype(np.uint8))

def read_file(filename, sample_rate=None, trim=False):
    """
    Reads in a wav file and returns it as an np.float32 array in the range [-1,1]
    """
    file_sr, signal = wavfile.read(filename)
    if signal.dtype == np.int16:
        signal = np.float32(signal) / np.iinfo(np.int16).max
    elif signal.dtype != np.float32:
        raise OSError('Encounted unexpected dtype: {}'.format(signal.dtype))
    if sample_rate is not None and sample_rate != file_sr:
        if len(signal) > 0:
            signal = librosa.core.resample(signal, file_sr, sample_rate, res_type='kaiser_fast')
        file_sr = sample_rate
    if trim and len(signal) > 1:
        signal = librosa.effects.trim(signal, top_db=40)[0]
    return signal, file_sr

def log_cqt(fname, sample_rate=None):
    """
    Generates a constant Q transform in dB magnitude
    """
    y, sample_rate = read_file(fname, sample_rate=sample_rate)
    fmin = None
    hop_length = 256
    n_bins = 256
    bins_per_octave = 32
    filter_scale = 0.1

    C = np.abs(librosa.cqt(y, sr=sample_rate,
                           hop_length=hop_length,
                           fmin=fmin,
                           n_bins=n_bins,
                           filter_scale=filter_scale,
                           bins_per_octave=bins_per_octave))
    C_db = librosa.power_to_db(C)
    return C_db

class SpatialAudioDatasetWaveform(torch.utils.data.Dataset):
    """
    Dataset of mixed waveforms and their corresponding ground truth waveforms
    recorded at different microphone.

    Data format is a pair of Tensors containing mixed waveforms and
    ground truth waveforms respectively. The tensor's dimension is formatted
    as (n_microphone, duration).
    """

    def __init__(self, input_path, sr=GLOBAL_SAMPLE_RATE):
        super().__init__()
        self.dirs = list(Path(input_path).glob('[0-9]*'))
        self.sr = sr

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        curr_dir = self.dirs[idx]

        mic_files = sorted(list(Path(curr_dir).rglob('*mixed.wav')))
        # Mixed signals
        mixed_waveforms = []
        for _, mic_file in enumerate(mic_files):
            mixed_waveform, _ = librosa.core.load(mic_file, self.sr, mono=True)
            mixed_waveforms.append(torch.from_numpy(mixed_waveform))
        mixed_data = torch.tensor(np.stack(mixed_waveforms)).float()

        # GT signals
        gt_audio_files = sorted(list(Path(curr_dir).rglob('*source00*.wav')))
        gt_waveforms = []
        for _, gt_audio_file in enumerate(gt_audio_files):
            gt_waveform, _ = librosa.core.load(gt_audio_file, self.sr, mono=True)
            gt_waveforms.append(torch.from_numpy(gt_waveform))
        gt_data = torch.tensor(np.stack(gt_waveforms)).float()

        gt_data = gt_data.unsqueeze(0) # Special case when only have one source separation

        # Get metadata
        with open(Path(curr_dir) / 'metadata.json') as json_file:
            json_data = json.load(json_file)
            locs = json_data['source00']['position']
        locs = torch.tensor(locs)

        return (mixed_data, gt_data, locs)

# class SpatialAudioDatasetSpectrogram(torch.utils.data.Dataset):
#     def __init__(self, input_path):
#         super().__init__()
#         self.dirs = list(Path(input_path).glob('[0-9]*'))

#     def __len__(self):
#         return len(self.dirs)

#     def __getitem__(self, idx):
#         curr_dir = self.dirs[idx]

#         # Get mic recordings
#         mic_files = sorted(list(Path(curr_dir).rglob('*mixed.wav')))

#         # Mixed signals
#         mixed_specgrams = []
#         for _, mic_file in enumerate(mic_files):
#             specgram = log_cqt(str(mic_file), sample_rate=self.sr)
#             mixed_specgrams.append(torch.from_numpy(specgram))

#         mixed_data = np.stack(mixed_specgrams)

#         # Ground truth signals
#         gt_audio_files = sorted(list(Path(curr_dir).rglob('*source00*.wav')))
#         gt_specgrams = []
#         for _, gt_audio_file in enumerate(gt_audio_files):
#             specgram = log_cqt(str(gt_audio_file), sample_rate=self.sr)
#             gt_specgrams.append(torch.from_numpy(specgram))

#         gt_data = torch.stack(gt_specgrams)

#         # Background masks
#         bg_max = np.ones_like(gt_data.numpy()) * np.NINF
#         for bg_source_idx in range(1, NUM_BGS + 1):
#             gt_bg_files = sorted(list(Path(curr_dir) \
#                 .rglob('*sourc{:02d}*.wav'.format(bg_source_idx))))
#             curr_bg_stack = np.zeros_like(bg_max)
#             for bg_mic_idx, gt_bg_file in enumerate(gt_bg_files):
#                 curr_bg_stack[bg_mic_idx, :, :] = log_cqt(gt_bg_file, sample_rate=self.sr)
#             bg_max = np.maximum(bg_max, curr_bg_stack)
#         mask = gt_data.numpy() > bg_max

#         # Padding for u-net
#         time_dim = mask.shape[2]
#         time_dim_padded = np.ceil(time_dim / DIM_DIVISOR) * DIM_DIVISOR

#         input_padded = np.zeros((mixed_data.shape[0], mixed_data.shape[1], time_dim_padded))
#         input_padded[:mixed_data.shape[0], :mixed_data.shape[1], :mixed_data.shape[2]] = mixed_data

#         mask_padded = np.zeros((mask.shape[0], mask.shape[1], time_dim_padded))
#         mask_padded[:mask.shape[0], :mask.shape[1], :mask.shape[2]] = mask
#         save_mask(mask_padded, "./data/")

#         return (torch.tensor(input_padded).float(),
#                 torch.tensor(mask_padded).float())


if __name__ == '__main__':
    data_train = SpatialAudioDatasetWaveform('./data/')
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=4)

    x = None
    for (data, label) in train_loader:
        x = (data, label)
        break
