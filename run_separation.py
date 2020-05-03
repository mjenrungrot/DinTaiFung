import argparse
import json
import random
from statistics import median
from pathlib import Path

import torch
import scipy.io
import numpy as np
import torch.nn.functional as F
import librosa.output
import mir_eval
import soundfile as sf

#from sdr import GetSDR, compute_measures

np.random.seed(123)
random.seed(123)
USE_CUDA = True

from data import SpatialAudioDatasetWaveform, RealDataset, MixedDataset, check_valid_dir
from network import center_trim, Demucs


ALL_ANGULAR_SPECIFICITY = [np.pi / 2,  # 90 degrees
                           np.pi / 4,  # 45 degrees
                           np.pi / 8,  # 22.5 degrees
                           np.pi / 16,  # 11.25 degrees
                           np.pi / 32,  # 5.625 degrees
                           ]
RADIUS = 2
SPEED_OF_SOUND = 343.0  # m/s


def shift_input(args, input_data, input_position):
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
        shift_samples = int(round(args.sr * shift_time))
        input_data[channel_idx] = torch.roll(input_data[channel_idx], -shift_samples)
        shifts.append(shift_samples)

    return input_data


def get_items(args):
    """
    This is a modified version of the SpatialAudioDataset DataLoader
    """
    curr_dir = args.test_dir
    with open(Path(curr_dir) / 'metadata.json') as json_file:
        json_data = json.load(json_file)

    num_voices = len(json_data) - 1
    mic_files = sorted(list(Path(curr_dir).rglob('*mixed.wav')))

    # All voice signals
    keys = ["voice{:02}".format(i) for i in range(num_voices)]

    """
    Loading the sources
    """
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
            gt_waveform, _ = librosa.core.load(gt_audio_file, args.sr, mono=True)
            gt_waveforms.append(gt_waveform)

        single_source = np.stack(gt_waveforms)
        all_sources.append(single_source)
        locs_voice = json_data[key]['position']
        voice_positions.append(locs_voice)

        sf.write("gt_{}.wav".format(key), single_source[0], args.sr)

    all_sources = np.stack(all_sources)  # n voices x n mics x n samples
    mixed_data = np.sum(all_sources, axis=0)  # n mics x n samples
    sf.write("mixed.wav", mixed_data[0], args.sr)

    import pdb
    pdb.set_trace()

    return mixed_data


def main(args):
    device = torch.device('cuda:0')

    mixed_data = get_items(args)


    model = Demucs(sources=2, n_audio_channels=args.n_channels)


    all_models = [
        args.model_checkpoint0,
        args.model_checkpoint1,
        args.model_checkpoint2,
        args.model_checkpoint3,
        args.model_checkpoint4,
    ]

    for angle_idx in range(5):
        model.load_state_dict(torch.load(all_models[angle_idx]))
        model.train = False
        model.to(device)


        curr_angular_specificity = ALL_ANGULAR_SPECIFICITY[angle_idx]
        divisor = 2**(2 + angle_idx)
        starting_angles = np.array(list(range(-divisor + 1, divisor, 2))) * np.pi / divisor

        for target_angle in starting_angles:
            target_pos = np.array([RADIUS * np.cos(target_angle), RADIUS * np.sin(target_angle)])

            data = shift_input(args, torch.tensor(mixed_data).to(device), target_pos)

            data = data.unsqueeze(0)  # Batch size is 1

            # Normalize input
            data = (data * 2**15).round() / 2**15
            ref = data.mean(1)  # Average across the n microphones
            means = ref.mean(1).unsqueeze(1).unsqueeze(2)
            stds = ref.std(1).unsqueeze(1).unsqueeze(2)
            data_transformed = (data - means) / stds

            # Run through the model
            valid_length = model.valid_length(data_transformed.shape[-1])
            delta = valid_length - data_transformed.shape[-1]
            padded = F.pad(data_transformed, (delta // 2, delta - delta // 2))

            output_signal, output_locs = model(padded)
            output_signal = center_trim(output_signal, data_transformed)
            output_locs = center_trim(output_locs, data_transformed)
            
            output_signal = output_signal * stds.unsqueeze(3) + means.unsqueeze(3)
            output_voices = output_signal[:, 0]  # batch x n_mics x n_samples

            sf.write("out{}_angle{:.2f}.wav".format(angle_idx, target_angle * 180 / np.pi), output_voices.detach().cpu().numpy()[0,0], args.sr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dir', type=str, help="Path to the testing directory")
    parser.add_argument('model_checkpoint0', type=str, help="Path to the model file")
    parser.add_argument('model_checkpoint1', type=str, help="Path to the model file")
    parser.add_argument('model_checkpoint2', type=str, help="Path to the model file")
    parser.add_argument('model_checkpoint3', type=str, help="Path to the model file")
    parser.add_argument('model_checkpoint4', type=str, help="Path to the model file")
    parser.add_argument('--sr', type=int, default=22050, help="Sampling rate")
    parser.add_argument('--n_channels', type=int, default=2, help="Number of channels")
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true', help="Whether to use cuda")
    parser.add_argument('--device', type=str, default='cpu', help="Device for pytorch")
    print(parser.parse_args())
    main(parser.parse_args())
