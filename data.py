from typing import Tuple, Optional
from pathlib import Path
import json
import torch
import numpy as np
from scipy.io import wavfile
import librosa

GLOBAL_SAMPLE_RATE: int = 22050
SPEED_OF_SOUND = 343.0  # m/s

def read_file(filename, sample_rate: Optional[int] = None, trim: bool = False) -> Tuple[np.ndarray, int]:
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

class SpatialAudioDatasetWaveform(torch.utils.data.Dataset):
    """
    Dataset of mixed waveforms and their corresponding ground truth waveforms
    recorded at different microphone.

    Data format is a pair of Tensors containing mixed waveforms and
    ground truth waveforms respectively. The tensor's dimension is formatted
    as (n_microphone, duration).
    """

    def __init__(self, input_path, n_sources=1, n_backgrounds=1, sr=GLOBAL_SAMPLE_RATE):
        super().__init__()
        self.dirs = list(Path(input_path).glob('[0-9]*'))
        self.n_sources = n_sources
        self.n_backgrounds = n_backgrounds
        self.sr = sr

    def __len__(self) -> int:
        return len(self.dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        curr_dir = self.dirs[idx]

        # Get metadata
        with open(Path(curr_dir) / 'metadata.json') as json_file:
            json_data = json.load(json_file)
            locs_voices = json_data['voice']['position']
            locs_bg = json_data['bg']['position']
        locs_voices = torch.tensor(locs_voices)
        locs_bg = torch.tensor(locs_bg)

        # mic_files = sorted(list(Path(curr_dir).rglob('*mixed.wav')))
        if np.random.uniform() < 0.3: # With 30% chance use the non-reverb sources
            mic_files = sorted(list(Path(curr_dir).rglob('*original_mixed.wav')))
        else:
            mic_files = sorted(list(Path(curr_dir).rglob('*reverb_mixed.wav')))

        # Mixed signals
        mixed_waveforms = []
        for _, mic_file in enumerate(mic_files):
            mixed_waveform, _ = librosa.core.load(mic_file, self.sr, mono=True)
            mixed_waveforms.append(torch.from_numpy(mixed_waveform))
        mixed_data = torch.tensor(np.stack(mixed_waveforms)).float()
        mixed_data = self.shift_input(mixed_data, np.array(locs_voices))

        # GT voice signals
        gt_voice_data = []
        for source in range(self.n_sources):
            #gt_audio_files = sorted(list(Path(curr_dir).rglob('*source{:02d}*.wav'.format(source))))
            gt_audio_files = sorted(list(Path(curr_dir).rglob('*original_source{:02d}*.wav'.format(source))))
            gt_waveforms = []
            for _, gt_audio_file in enumerate(gt_audio_files):
                gt_waveform, _ = librosa.core.load(gt_audio_file, self.sr, mono=True)
                gt_waveforms.append(torch.from_numpy(gt_waveform))
            gt_voice_data.append(self.shift_input(torch.tensor(np.stack(gt_waveforms)).float(), np.array(locs_voices)))
        gt_voice_data = torch.stack(gt_voice_data, dim=0)
        
        # GT background signals
        gt_bg_data = []
        for source in range(self.n_sources, self.n_sources + self.n_backgrounds):
            #gt_audio_files = sorted(list(Path(curr_dir).rglob('*source{:02d}*.wav'.format(source))))
            gt_audio_files = sorted(list(Path(curr_dir).rglob('*original_source{:02d}*.wav'.format(source))))
            gt_waveforms = []
            for _, gt_audio_file in enumerate(gt_audio_files):
                gt_waveform, _ = librosa.core.load(gt_audio_file, self.sr, mono=True)
                gt_waveforms.append(torch.from_numpy(gt_waveform))
            gt_bg_data.append(torch.tensor(np.stack(gt_waveforms)).float())
        gt_bg_data = torch.stack(gt_bg_data, dim=0)



        return (mixed_data, gt_voice_data, gt_bg_data, locs_voices, locs_bg)

    def shift_input(self, input_data, input_position):
        """
        Shifts the input according to the voice position. This tried to
        line up the voice samples in the time domain
        """
        radius = 0.145 / 2
        num_channels = input_data.shape[0]
        mic_array = [[radius * np.cos(2 * np.pi / num_channels * i), radius * np.sin(2 * np.pi / num_channels * i), 0] for i in range(6)]

        for channel_idx in range(num_channels):
            distance = np.linalg.norm(mic_array[channel_idx] - input_position)
            shift_time = distance / SPEED_OF_SOUND
            shift_samples = int(GLOBAL_SAMPLE_RATE * shift_time)
            input_data[channel_idx] = torch.roll(input_data[channel_idx], -shift_samples)

        return input_data

if __name__ == '__main__':
    data_train = SpatialAudioDatasetWaveform('/projects/grail/audiovisual/datasets/DinTaiFung/mics_8_radius_3_voice_1_bg_1/train')
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=4)

    x = None
    for x in train_loader:
        print(x)
        break
