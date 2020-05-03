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


def get_items(args, curr_dir, starting_angles, curr_angular_specificity):
    """
    This is a modified version of the SpatialAudioDataset DataLoader
    """
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

    all_sources = np.stack(all_sources)  # n voices x n mics x n samples
    mixed_data = np.sum(all_sources, axis=0)  # n mics x n samples

    # To avoid collate issues
    while len(voice_positions) < 4:
        voice_positions.append([0, 0])

    """
    Generating data points
    """
    batched_data = []
    batched_labels = []
    batched_voice_positions = []
    batched_target_positions = []

    # For each voice, return one true and one false sample
    for voice_idx in range(num_voices):
        key = keys[voice_idx]
        voice_pos = json_data[key]["position"]
        voice_pos = np.array(voice_pos)
        voice_angle = np.arctan2(voice_pos[1], voice_pos[0])

        # A true sample

        # Get the sector closest to that voice
        angle_idx = (np.abs(starting_angles - voice_angle)).argmin()
        target_angle = starting_angles[angle_idx]
        target_pos = np.array([RADIUS * np.cos(target_angle), RADIUS * np.sin(target_angle)])

        # Get all voices there
        labels = []
        for idx, key in enumerate(keys):
            potential_voice_pos = json_data[key]["position"]
            potential_voice_angle = np.arctan2(potential_voice_pos[1], potential_voice_pos[0])
            if abs(potential_voice_angle - target_angle) < (curr_angular_specificity / 2):
                labels.append(all_sources[idx])

        labels = np.array(labels).sum(0)
        shifted_mixture = shift_input(args, torch.tensor(mixed_data), target_pos)
        shifted_labels = shift_input(args, torch.tensor(labels), target_pos)

        batched_data.append(shifted_mixture)
        batched_labels.append(shifted_labels)
        batched_voice_positions.append(torch.tensor(voice_pos))
        batched_target_positions.append(torch.tensor(target_pos))


        # A random sample
        target_angle = np.random.choice(starting_angles)
        target_pos = np.array([RADIUS * np.cos(target_angle), RADIUS * np.sin(target_angle)])
        
        # Get all voices there
        labels = all_sources[voice_idx]

        shifted_mixture = shift_input(args, torch.tensor(mixed_data), target_pos)
        shifted_labels = shift_input(args, torch.tensor(labels), target_pos)

        batched_data.append(shifted_mixture)
        batched_labels.append(shifted_labels)
        batched_voice_positions.append(torch.tensor(voice_pos))
        batched_target_positions.append(torch.tensor(target_pos))

    return (torch.stack(batched_data, 0), torch.stack(batched_labels, 0),
            torch.stack(batched_voice_positions, 0), torch.stack(batched_target_positions, 0))


def main(args):
    # input_stats = np.load("input_sdr.npy")
    # output_stats = np.load("output_sdr.npy")
    # draw_plot(input_stats, output_stats, "sdr")

    kwargs = {
        'num_workers': args.n_workers,
        'pin_memory': True
    }

    device = torch.device('cuda:0')


    model = Demucs(sources=2, n_audio_channels=args.n_channels)
    model.load_state_dict(torch.load(args.model_checkpoint))
    model.train = False
    model.to(device)

    all_input_sdr = []
    all_output_sdr = []
    all_input_angles = []

    def debug_signal_handler(signal, frame, vars):
        import pdb
        pdb.set_trace()

    import signal
    signal.signal(signal.SIGINT, debug_signal_handler)

    curr_angular_specificity = ALL_ANGULAR_SPECIFICITY[args.angle_idx]
    divisor = 2**(2 + args.angle_idx)
    starting_angles = np.array(list(range(-divisor + 1, divisor, 2))) * np.pi / divisor

    all_dirs = sorted(list(Path(args.test_dir).glob('[0-9]*')))
    all_dirs = [x for x in all_dirs if check_valid_dir(x)]

    try:
        with torch.no_grad():
            for dir_idx, curr_dir in enumerate(all_dirs):
                print(dir_idx)
                data, label_voice_signals, voice_pos, target_pos = get_items(args, curr_dir, starting_angles, curr_angular_specificity)

                voice_pos = voice_pos.numpy()
                target_pos = target_pos.numpy()

                data = data.to(device)
                label_voice_signals = label_voice_signals.to(device)

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


                channels = [0]
                # if args.n_channels == 2:
                #     channels = [0, 1]
                # elif args.n_channels == 6:
                #     channels = [0, 3]

                # Iterate over every input
                for i in range(label_voice_signals.shape[0]):

                    # Figure out how many foregrounds there are
                    target_angle = np.arctan2(target_pos[i, 1], target_pos[i, 0])
                    final_voice_angle = np.arctan2(voice_pos[i, 1], voice_pos[i, 0])
        
                    angle_diff = (target_angle - final_voice_angle) * 180 / np.pi

                    input_sdr, input_sir, input_sar, input_perm = mir_eval.separation.bss_eval_sources(label_voice_signals[i, channels].cpu().numpy(), data[i, channels].cpu().numpy(), compute_permutation=False)
                    output_sdr, output_sir, output_sar, output_perm = mir_eval.separation.bss_eval_sources(label_voice_signals[i, channels].cpu().numpy(), output_voices[i, channels].cpu().numpy(), compute_permutation=False)

                    all_input_sdr.append(input_sdr[0])
                    all_output_sdr.append(output_sdr[0])
                    all_input_angles.append(angle_diff)

                    #print("idx {} input SDR {} output SDR {}".format(i, input_sdr, output_sdr))
                    # sf.write("labels.wav", label_voice_signals.detach().cpu().numpy()[i, 0, 0], 22050)
                    # sf.write("output.wav", output_voices.detach().cpu().numpy()[i, 0], 22050)

        joint_data = np.stack((np.array(all_input_sdr), np.array(all_output_sdr), np.array(all_input_angles)), axis=0)
        np.save("5angle_sdr_{}mics.npy".format(args.n_channels), joint_data)

    except Exception as e:
        print(e)
        import pdb
        pdb.set_trace()

    import pdb
    pdb.set_trace()

    print(all_input_sdr)
    print(all_output_sdr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dir', type=str, help="Path to the testing dataset")
    parser.add_argument('model_checkpoint', type=str, help="Path to the model file")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--sr', type=int, default=22050, help="Sampling rate")
    parser.add_argument('--n_channels', type=int, default=2, help="Number of channels")
    parser.add_argument('--n_workers', type=int, default=32, help="Number of parallel workers")
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true', help="Whether to use cuda")
    parser.add_argument('--device', type=str, default='cpu', help="Device for pytorch")
    parser.add_argument('--angle_idx', default=0, type=int, help="See data.py for details")
    print(parser.parse_args())
    main(parser.parse_args())
