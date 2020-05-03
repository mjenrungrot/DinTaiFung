import os
import json
import random

from typing import Tuple, Optional
from pathlib import Path

import torch
import numpy as np
import librosa

from pysndfx import AudioEffectsChain
from scipy.io import wavfile



GLOBAL_SAMPLE_RATE: int = 44100
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


def check_valid_dir(dir, requires_n_voices=1):
    """Checks that there is at least one voice"""
    if len(list(Path(dir).glob('*_voice00.wav'))) < 6:
        return False

    if requires_n_voices == 2:
        if len(list(Path(dir).glob('*_voice01.wav'))) < 6:
            return False

    if requires_n_voices == 3:
        if len(list(Path(dir).glob('*_voice02.wav'))) < 6:
            return False

    if requires_n_voices == 4:
        if len(list(Path(dir).glob('*_voice03.wav'))) < 6:
            return False

    if len(list(Path(dir).glob('metadata.json'))) < 1:
        return False
    return True


class SpatialAudioDatasetWaveform(torch.utils.data.Dataset):
    """
    Dataset of mixed waveforms and their corresponding ground truth waveforms
    recorded at different microphone.

    Data format is a pair of Tensors containing mixed waveforms and
    ground truth waveforms respectively. The tensor's dimension is formatted
    as (n_microphone, duration).
    """

    def __init__(self, input_path,
                 n_mics=1,
                 n_sources=1,
                 n_backgrounds=1, sr=GLOBAL_SAMPLE_RATE, location_shifts=12,
                 target_fg_std=0.03,
                 target_bg_std=0.04,
                 perturb_prob=0.0,
                 angular_specificity_idx=0,
                 dont_shift=False,
                 requires_n_voices=1):
        super().__init__()
        all_dirs = sorted(list(Path(input_path).glob('[0-9]*')))
        self.dirs = [x for x in all_dirs if check_valid_dir(x, requires_n_voices)]

        self.n_sources = n_sources
        self.n_backgrounds = n_backgrounds
        self.sr = sr
        self.target_fg_std = target_fg_std
        self.target_bg_std = target_bg_std
        self.perturb_prob = perturb_prob
        self.n_mics = n_mics
        self.angular_specificity_idx = angular_specificity_idx
        self.dont_shift = dont_shift

    def __len__(self) -> int:
        return len(self.dirs)  # Two different ground truth speakers

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.angular_specificity_idx = -1:
            curr_angular_specificity_idx = np.random.randint(0, 5)
        else:
            curr_angular_specificity_idx = self.angular_specificity_idx

        ALL_ANGULAR_SPECIFICITY = [np.pi / 2,  # 90 degrees
                                   np.pi / 4,  # 45 degrees
                                   np.pi / 8,  # 22.5 degrees
                                   np.pi / 16,  # 11.25 degrees
                                   np.pi / 32,  # 5.625 degrees
                                   ]
        curr_angular_specificity = ALL_ANGULAR_SPECIFICITY[curr_angular_specificity_idx]
        #STARTING_ANGLES = np.array([-3, -1, 1, 3]) * np.pi / 4  # Middle of each quadrant
        #STARTING_ANGLES = np.array([-7, -5, -3, -1, 1, 3, 5, 7]) * np.pi / 8  # Middle of each quadrant

        divisor = 2**(2+curr_angular_specificity_idx)
        starting_angles = np.array(list(range(-divisor + 1, divisor, 2))) * np.pi / divisor

        RADIUS = 2

        curr_dir = self.dirs[idx]

        # Get metadata
        with open(Path(curr_dir) / 'metadata.json') as json_file:
            json_data = json.load(json_file)

        num_voices = len(json_data) - 1

        # target_voice = random.randint(0, num_voices - 1)
        #target_voice_key = 'voice{:02}'.format(target_voice)
        #locs_voice = json_data[target_voice_key]['position']
        #locs_voice = torch.tensor(locs_voice)

        mic_files = sorted(list(Path(curr_dir).rglob('*mixed.wav')))
        random_perturb = RandomAudioPerturbation()

        # All voice signals
        keys = ["voice{:02}".format(i) for i in range(num_voices)]
        # keys.append("bg")

        # Either use a random slice or get one with a voice 
        random_probability = 1. / (2**(curr_angular_specificity_idx + 1))
        if np.random.uniform() < random_probability:
            target_angle = np.random.choice(starting_angles)
            random_pos = np.array([RADIUS * np.cos(target_angle), RADIUS * np.sin(target_angle)])
            target_pos = random_pos
        else:
            random_key = random.choice(keys)
            voice_pos = json_data[random_key]["position"]
            voice_pos = np.array(voice_pos)
            voice_angle = np.arctan2(voice_pos[1], voice_pos[0])

            # Get the sector closest to that voice
            angle_idx = (np.abs(starting_angles - voice_angle)).argmin()
            target_angle = starting_angles[angle_idx]
            target_pos = np.array([RADIUS * np.cos(target_angle), RADIUS * np.sin(target_angle)])

        # Iterate over different sources
        all_sources = []
        target_voice_data = []
        voice_positions = []
        for key in keys:
            gt_audio_files = sorted(list(Path(curr_dir).rglob("*" + key + ".wav")))
            assert(len(gt_audio_files) > 0)
            gt_waveforms = []
            # Iterate over different mics
            for _, gt_audio_file in enumerate(gt_audio_files):
                gt_waveform, _ = librosa.core.load(gt_audio_file, self.sr, mono=True)

                # Normalize volume
                if "bg" not in key:
                    if self.target_fg_std is not None:
                        gt_waveform = gt_waveform / ((gt_waveform.std() + 1e-4) / self.target_fg_std)
                        random_fg_volume = np.random.uniform(0.5, 4.0)
                        gt_waveform *= random_fg_volume
                else:
                    if self.target_bg_std is not None:
                        gt_waveform = gt_waveform / ((gt_waveform.std() + 1e-4) / self.target_bg_std)
                        random_bg_volume = np.random.uniform(0.1, 1.5)
                        gt_waveform *= random_bg_volume

                gt_waveforms.append(torch.from_numpy(gt_waveform))
            

            if self.dont_shift:
                shifted_gt = torch.tensor(np.stack(gt_waveforms)).float()
            else:
                shifted_gt, _ = self.shift_input(torch.tensor(np.stack(gt_waveforms)).float(), target_pos)

            if np.random.uniform() < self.perturb_prob:
                #perturbed_source = torch.tensor(random_perturb(shifted_gt[[0,3]].numpy()))  # 6 mic array going to 2 mics. Hard coded for now
                perturbed_source = torch.tensor(random_perturb(shifted_gt.numpy()))
            else:
                #perturbed_source = shifted_gt[[0,3]]
                perturbed_source = shifted_gt

            all_sources.append(perturbed_source)

            locs_voice = json_data[key]['position']
            voice_angle = np.arctan2(locs_voice[1], locs_voice[0])
            voice_positions.append(locs_voice)

            # Need to save for ground truth
            if abs(voice_angle - target_angle) < (curr_angular_specificity / 2):
                target_voice_data.append(perturbed_source.view(1, perturbed_source.shape[0], perturbed_source.shape[1]))
        
            else:
                target_voice_data.append(torch.zeros((1, perturbed_source.shape[0], perturbed_source.shape[1])))

        all_sources = torch.stack(all_sources, dim=0)
        mixed_data = torch.sum(all_sources, dim=0)

        target_voice_data = torch.stack(target_voice_data, dim=0)
        target_voice_data = torch.sum(target_voice_data, dim=0)

        while len(voice_positions) < 4:
            voice_positions.append([0, 0])

        return (mixed_data, target_voice_data, torch.tensor(voice_positions), torch.tensor(target_pos), curr_angular_specificity_idx)

        # if self.n_mics == 1:
        #     return (mixed_data[[0]], target_voice_data[:,[0]])

        # elif self.n_mics == 2:
        #     return (mixed_data[[0,3]], target_voice_data[:,[0,3]])

        # elif self.n_mics == 6:
        #     return (mixed_data, target_voice_data)

        # else:
        #     raise ValueError("Invalid number of mics {}".format(self.n_mics))

    def shift_input(self, input_data, input_position):
        """
        Shifts the input according to the voice position. This tried to
        line up the voice samples in the time domain
        """
        radius = 0.145 / 2
        num_channels = input_data.shape[0]
        mic_array = [[radius * np.cos(2 * np.pi / num_channels * i), radius * np.sin(2 * np.pi / num_channels * i)] for i in range(6)]

        distance0 = np.linalg.norm(mic_array[0] - input_position)
        shifts = [0]
        for channel_idx in range(1, num_channels):
            distance = np.linalg.norm(mic_array[channel_idx] - input_position)
            distance_diff = distance - distance0
            shift_time = distance_diff / SPEED_OF_SOUND
            shift_samples = int(round(self.sr * shift_time))
            input_data[channel_idx] = torch.roll(input_data[channel_idx], -shift_samples)
            shifts.append(shift_samples)

        return input_data, shifts


class RealDataset(torch.utils.data.Dataset):
    """
    Dataset of synthetic composites of real data
    """

    def __init__(self, input_dir, duration=3.0,
                 sr=GLOBAL_SAMPLE_RATE,
                 num_elements=1000,
                 perturb_prob=0.0,
                 short_data=False,
                 target_fg_std=0.03,
                 target_bg_std=0.04,
                 max_num_voices=3):
        super().__init__()
        self.duration = duration
        self.sr = sr
        self.fgs = []
        self.bgs = []
        self.num_elements = num_elements
        self.perturb_prob = perturb_prob
        self.max_num_voices = max_num_voices

        duration = 60.0 if short_data else None 

        # Read fg files
        all_fg_files = os.listdir(os.path.join(input_dir, "fg"))
        for fg_file in all_fg_files:
            full_path = os.path.join(input_dir, "fg", fg_file)
            data = librosa.core.load(full_path, sr=self.sr, mono=False, duration=duration)[0]
            shift = int(fg_file.split("shift")[-1][:-4])
            print(shift)
            data = data / (data.std() / target_fg_std)
            self.fgs.append((data, shift))

        # Read bg files
        all_bg_files = os.listdir(os.path.join(input_dir, "bg"))
        for bg_file in all_bg_files:
            full_path = os.path.join(input_dir, "bg", bg_file)
            data = librosa.core.load(full_path, sr=self.sr, mono=False, duration=duration)[0]
            data = data / (data.std() / target_bg_std)
            self.bgs.append(data)


    def __len__(self) -> int:
        return self.num_elements

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Choose the number of voices in the sample
        num_voices = random.randint(1, min(self.max_num_voices, len(self.fgs)))
        voices = random.sample(self.fgs, num_voices)

        # Randomly sample n voices
        all_fg = []
        all_shifts = []
        for curr_fg, curr_shift in voices:
            start_fg_sample = np.random.randint(0, curr_fg.shape[1] - (self.sr * self.duration + 1))
            random_fg_volume = np.random.uniform(0.5, 4.0)
            fg_data = curr_fg[:, start_fg_sample:int(start_fg_sample+(self.sr * self.duration))] * random_fg_volume
            all_fg.append(fg_data)
            all_shifts.append(curr_shift)

        # print("-----------------------------")
        # print("Shifts: {}".format(all_shifts))
        fg_data = sum(all_fg)
        
        # Sample 1 bg
        curr_bg = random.choice(self.bgs) 
        start_bg_sample = np.random.randint(0, curr_bg.shape[1] - (self.sr * self.duration + 1))
        random_bg_volume = np.random.uniform(0.1, 1.5)
        bg_data = curr_bg[:, start_bg_sample:int(start_bg_sample+(self.sr * self.duration))] * random_bg_volume
        random_bg_shift = np.random.randint(-12, 12)
        # print("BG Shift: {}".format(random_bg_shift))
        bg_data = self.shift_input(torch.tensor(bg_data), random_bg_shift).numpy()


        # Data augmentation
        if np.random.uniform() < self.perturb_prob:
            random_perturb = RandomAudioPerturbation()
            fg_data = random_perturb(fg_data)
            bg_data = random_perturb(bg_data)

        mixed_data = fg_data + bg_data
        shift = round(all_shifts[0] * self.sr / 44100)

        mixed_data = torch.tensor(mixed_data)
        mixed_data = self.shift_input(mixed_data, shift)

        gt = self.shift_input(torch.tensor(all_fg[0]), shift)

        return (mixed_data, gt.view(1, 2, -1))

    def shift_input(self, data, shift):
        """
        Shifts the input according to the voice position. This tried to
        line up the voice samples in the time domain
        # """
        data[0] = torch.roll(data[0], int(shift))

        return data

if __name__ == '__main__':
    data_train = SpatialAudioDatasetWaveform('/projects/grail/audiovisual/datasets/DinTaiFung/mics_8_radius_3_voice_1_bg_1/train')
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=4)

    x = None
    for x in train_loader:
        print(x)
        break

class MixedDataset(torch.utils.data.Dataset):

    def __init__(self, input_dir_synth, input_dir_real, num_elements_real, sr=GLOBAL_SAMPLE_RATE, perturb_prob=0.0, short_data=False):
        
        self.synth_dataset = SpatialAudioDatasetWaveform(input_dir_synth,
            n_sources=1, n_backgrounds=1, sr=sr, perturb_prob=perturb_prob)

        self.real_dataset = RealDataset(input_dir_real, num_elements=num_elements_real, sr=sr, perturb_prob=perturb_prob, short_data=short_data)

    def __len__(self):
        return len(self.synth_dataset) + len(self.real_dataset)

    def __getitem__(self, idx: int):
        if idx < len(self.real_dataset):
            return ("real", self.real_dataset[idx])
        else:
            return ("synth", self.synth_dataset[idx - len(self.real_dataset)])

class RandomAudioPerturbation(object):
    """Randomly perturb audio samples"""

    def __call__(self, data):
        highshelf_gain = np.random.normal(0, 5)
        lowshelf_gain = np.random.normal(0, 5)
        noise_amount = np.random.uniform(0, 0.003)
        shift = 0 #random.randint(-1, 1)

        fx = (
            AudioEffectsChain()
            .highshelf(gain=highshelf_gain)
            .lowshelf(gain=lowshelf_gain)
        )

        for i in range(data.shape[0]):
            data[i] = fx(data[i])
            data[i] += np.random.uniform(-noise_amount, noise_amount, size=data[i].shape)
            np.roll(data[i], shift, axis=0)
        return data

