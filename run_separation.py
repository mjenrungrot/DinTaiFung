import argparse
import json
import random
import os
import shutil

from statistics import median
from pathlib import Path

import torch
import scipy.io
import numpy as np
import torch.nn.functional as F
import librosa.output
import mir_eval
import soundfile as sf

from mir_eval_tests import compute_angle_error, compute_sdr
# from drawing_utils import draw_dia

#from sdr import GetSDR, compute_measures

np.random.seed(123)
random.seed(123)
USE_CUDA = True

ENERGY_CUTOFF = 0.003
#ENERGY_CUTOFF = 0.000
NMS_RADIUS = np.pi / 4
NMS_SIMILARITY_CUTOFF = 0.015
#NMS_SIMILARITY_CUTOFF = 0.000


from data import check_valid_dir
from network import center_trim, Demucs


ALL_ANGULAR_SPECIFICITY = [np.pi / 2,  # 90 degrees
                           np.pi / 4,  # 45 degrees
                           np.pi / 8,  # 22.5 degrees
                           np.pi / 16,  # 11.25 degrees
                           np.pi / 32,  # 5.625 degrees
                           ]
RADIUS = 2
SPEED_OF_SOUND = 343.0  # m/s

def to_categorical(index: int, num_classes: int):
    data = np.zeros((num_classes))
    data[index] = 1

    return data


def shift_input(args, input_data, input_position, inverse=False):
    """
    Shifts the input according to the voice position. This tried to
    line up the voice samples in the time domain
    """
    #radius = 0.145 / 2
    radius = .1016
    num_channels = input_data.shape[0]
    mic_array = [[radius * np.cos(2 * np.pi / num_channels * i), radius * np.sin(2 * np.pi / num_channels * i)] for i in range(num_channels)]

    distance0 = np.linalg.norm(mic_array[0] - input_position)
    shifts = [0]
    for channel_idx in range(1, num_channels):
        distance = np.linalg.norm(mic_array[channel_idx] - input_position)
        distance_diff = distance - distance0
        shift_time = distance_diff / SPEED_OF_SOUND
        shift_samples = int(round(args.sr * shift_time))
        if inverse:
            input_data[channel_idx] = torch.roll(input_data[channel_idx], shift_samples)
        else:
            input_data[channel_idx] = torch.roll(input_data[channel_idx], -shift_samples)
        shifts.append(shift_samples)

    return input_data

def shift_input_np(args, input_data, input_position, inverse=False):
    """
    Shifts the input according to the voice position. This tried to
    line up the voice samples in the time domain
    """
    radius = .1016
    num_channels = input_data.shape[0]
    mic_array = [[radius * np.cos(2 * np.pi / num_channels * i), radius * np.sin(2 * np.pi / num_channels * i)] for i in range(num_channels)]

    distance0 = np.linalg.norm(mic_array[0] - input_position)
    shifts = [0]
    for channel_idx in range(1, num_channels):
        distance = np.linalg.norm(mic_array[channel_idx] - input_position)
        distance_diff = distance - distance0
        shift_time = distance_diff / SPEED_OF_SOUND
        shift_samples = int(round(args.sr * shift_time))
        if inverse:
            input_data[channel_idx] = np.roll(input_data[channel_idx], shift_samples)
        else:
            input_data[channel_idx] = np.roll(input_data[channel_idx], -shift_samples)
        shifts.append(shift_samples)

    return input_data


def draw_outputs(candidate_angles):
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    voice_positions = []
    for angle in candidate_angles:
        voice_positions.append([np.cos(angle) * RADIUS, np.sin(angle) * RADIUS])

    fig, ax = plt.subplots()
    ax.set(xlim=(-5, 5), ylim = (-5, 5))
    ax.set_aspect("equal")
    ax.axhline()
    ax.axvline()

    for pos in voice_positions:
        a_circle = plt.Circle((pos[0], pos[1]), 0.3, color='b', fill=False)
        ax.add_artist(a_circle)
    plt.savefig("output.png")

def get_items(curr_dir, args):
    """
    This is a modified version of the SpatialAudioDataset DataLoader
    """
    with open(Path(curr_dir) / 'metadata.json') as json_file:
        json_data = json.load(json_file)

    #num_voices = min(len(json_data) - 1, 2)
    #num_voices = len(json_data) - 1
    num_voices = 1

    print("Num voices: {}".format(num_voices))
    mic_files = sorted(list(Path(curr_dir).rglob('*mixed.wav')))

    # All voice signals
    keys = ["voice{:02}".format(i) for i in range(num_voices)]
    keys.append("bg")

    """
    Loading the sources
    """
    # Iterate over different sources
    all_sources = []
    target_voice_data = []
    voice_positions = []
    for key in keys:
        if "bg" not in key:
            print("Voice pos {}".format(np.arctan2(json_data[key]["position"][1],
                                        json_data[key]["position"][0])))
        else:
            print("BG pos {}".format(np.arctan2(json_data[key]["position"][1],
                                        json_data[key]["position"][0])))
        gt_audio_files = sorted(list(Path(curr_dir).rglob("*" + key + ".wav")))
        assert(len(gt_audio_files) > 0)
        gt_waveforms = []

        # Iterate over different mics
        for _, gt_audio_file in enumerate(gt_audio_files):
            gt_waveform, _ = librosa.core.load(gt_audio_file, args.sr, mono=True)

            # if "bg" in key:
            #     gt_waveform /= 2.
            gt_waveforms.append(gt_waveform)

        single_source = np.stack(gt_waveforms)
        all_sources.append(single_source)
        locs_voice = np.arctan2(json_data[key]["position"][1],
                                        json_data[key]["position"][0])
        voice_positions.append(locs_voice)

        if args.debug:
            sf.write(os.path.join(args.writing_dir, "gt_{}.wav".format(key)), single_source[0], args.sr)

    all_sources = np.stack(all_sources)  # n voices x n mics x n samples
    mixed_data = np.sum(all_sources, axis=0)  # n mics x n samples

    gt = [(voice_positions[i], all_sources[i]) for i in range(num_voices)]

    if args.debug:
        sf.write(os.path.join(args.writing_dir, "mixed.wav"), mixed_data[0], args.sr)


    if args.n_channels == 2:
        return mixed_data[[0, 3]], gt

    else:
        return mixed_data, gt


def angular_distance(angle1, angle2):
    d1 = abs(angle1 - angle2)
    d2 = abs(angle1 - angle2 + 2 * np.pi)
    d3 = abs(angle2 - angle1 + 2 * np.pi)

    return min(d1, d2, d3)

def nms(candidate_angles, nms_cutoff):
    final_proposals = []
    initial_proposals = candidate_angles

    while len(initial_proposals) > 0:
        new_initial_proposals = []
        sorted_angles = sorted(initial_proposals, key=lambda x: x[1], reverse=True)
        candidate_angle = sorted_angles[0]
        final_proposals.append(candidate_angle)

        for angle in sorted_angles:
            # print(abs(candidate_angle[2] - angle[2]).mean())
            if angular_distance(candidate_angle[0], angle[0]) > NMS_RADIUS or abs(candidate_angle[2] - angle[2]).mean() > nms_cutoff:
                new_initial_proposals.append(angle)

        initial_proposals = new_initial_proposals

    return final_proposals


def run_separation(mixed_data, model, args, energy_cutoff, nms_cutoff):
    divisor = 4
    starting_angles = np.array(list(range(-divisor + 1, divisor, 2))) * np.pi / divisor

    candidate_angles = [(x, None, None) for x in starting_angles]
    for angle_idx in range(5):
        if args.debug:
            print("---------")
        conditioning_label = torch.tensor(to_categorical(angle_idx, 5)).float().to(args.device).unsqueeze(0)

        curr_angular_specificity = ALL_ANGULAR_SPECIFICITY[angle_idx]

        new_candidate_angles = []
        if args.debug:
            print(len(candidate_angles))
        for target_angle, _, _ in candidate_angles:
            target_pos = np.array([RADIUS * np.cos(target_angle), RADIUS * np.sin(target_angle)])

            data = shift_input(args, torch.tensor(mixed_data).to(args.device), target_pos)

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

            output_signal = model(padded, conditioning_label)

            output_signal = center_trim(output_signal, data_transformed)
            
            output_signal = output_signal * stds.unsqueeze(3) + means.unsqueeze(3)
            output_voices = output_signal[:, 0]  # batch x n_mics x n_samples

            output_np = output_voices.detach().cpu().numpy()[0]
            energy = librosa.feature.rms(output_np).mean()

            sf.write(os.path.join(args.writing_dir, "out{}_angle{:.2f}.wav".format(angle_idx, target_angle * 180 / np.pi)), output_np[0], args.sr)

            if args.debug:
                print("Angle {:.2f} energy {}".format(target_angle, energy))

            if energy > energy_cutoff:
                if angle_idx == 4:
                    unshifted_output = shift_input_np(args, output_np, target_pos, inverse=True)
                    new_candidate_angles.append((target_angle, energy, unshifted_output))
                else:
                    new_candidate_angles.append((target_angle + curr_angular_specificity / 4, energy, output_np))
                    new_candidate_angles.append((target_angle - curr_angular_specificity / 4, energy, output_np))

        candidate_angles = new_candidate_angles

    #draw_outputs([angle[0] for angle in candidate_angles])
    candidate_angles = nms(candidate_angles, nms_cutoff)

    if args.debug:
        for angle in candidate_angles:
            sf.write(os.path.join(args.writing_dir, "out{}_angle{:.2f}.wav".format(angle_idx, angle[0] * 180 / np.pi)), angle[2][1], args.sr)

    
    return candidate_angles


def main(args):
    device = torch.device('cuda')

    args.device = device
    model = Demucs(sources=2, n_audio_channels=args.n_channels)
    model.load_state_dict(torch.load(args.model_checkpoint), strict=True)
    model.train = False
    model.to(device)

    all_dirs = sorted(list(Path(args.test_dir).glob('[0-9]*')))[2:3]
    # all_dirs = [x for x in all_Dirs if check_valid_dir(x, 2)]

    all_angle_errors = []
    all_input_sdr = []
    all_output_sdr = []
    for curr_dir in all_dirs:
        print("-------------")

        if args.debug:
            writing_dir = os.path.join(os.getcwd(), str(curr_dir).split("/")[-1])
            print(writing_dir)

            if not os.path.exists(writing_dir):
                os.makedirs(writing_dir)

            args.writing_dir = writing_dir
    
        mixed_data, gt = get_items(curr_dir, args)

        candidate_angles = run_separation(mixed_data, model, args, ENERGY_CUTOFF, NMS_SIMILARITY_CUTOFF)

        if len(candidate_angles) < 2:
            candidate_angles = run_separation(mixed_data, model, args, ENERGY_CUTOFF / 2, NMS_SIMILARITY_CUTOFF)

        if len(candidate_angles) < 2:
            candidate_angles = run_separation(mixed_data, model, args, ENERGY_CUTOFF / 2, NMS_SIMILARITY_CUTOFF / 2)

        if len(candidate_angles) < 2:
            candidate_angles = run_separation(mixed_data, model, args, ENERGY_CUTOFF / 6, NMS_SIMILARITY_CUTOFF / 2)

        if len(candidate_angles) < 2:
            candidate_angles = run_separation(mixed_data, model, args, ENERGY_CUTOFF / 6, NMS_SIMILARITY_CUTOFF / 4)

        
        trimmed_angles = candidate_angles[:2]
        angle_error = compute_angle_error([angle[0] for angle in trimmed_angles],
                                          [item[0] for item in gt])

        all_angle_errors.append(angle_error)

        input_sdr = compute_sdr([mixed_data, mixed_data], [item[1] for item in gt])
        output_sdr = compute_sdr([angle[2] for angle in trimmed_angles], [item[1] for item in gt])

        all_input_sdr.append(input_sdr)
        all_output_sdr.append(output_sdr)

        print(curr_dir)
        print([input_sdr[0].mean(), input_sdr[1].mean()])
        print([output_sdr[0].mean(), output_sdr[1].mean()])

        import pdb
        pdb.set_trace()
        print(all_input_sdr)
        print(all_output_sdr)
        print(all_angle_errors)


    import pdb
    pdb.set_trace()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dir', type=str, help="Path to the testing directory")
    parser.add_argument('model_checkpoint', type=str, help="Path to the model file")
    parser.add_argument('--sr', type=int, default=22050, help="Sampling rate")
    parser.add_argument('--n_channels', type=int, default=2, help="Number of channels")
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true', help="Whether to use cuda")
    parser.add_argument('--device', type=str, default='cpu', help="Device for pytorch")
    parser.add_argument('--debug', action='store_true', help="Save outputs")
    print(parser.parse_args())
    main(parser.parse_args())
