import os
import sys

import argparse
import json
from typing import List
from pathlib import Path
import tqdm
import random

import multiprocessing.dummy as mp


import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import IPython
import pyroomacoustics as pra
import soundfile as sf

N_MICS = 6
FG_TARGET_VOL = 0.4
BG_TARGET_VOL = 0.5

def generate_mic_array(room, radius: float = 0.145 / 2, n_mics: int = N_MICS):
    """
    Generate a list of Microphone objects

    Radius = 50th percentile of men Bitragion breadth
    (https://en.wikipedia.org/wiki/Human_head)
    """
    R = pra.circular_2D_array(center=[0.,0.], M=n_mics,
                              phi0=0, radius=radius)
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

def handle_error(e):
    print(e)

def generate_sample(args: argparse.Namespace,
                    bg: np.ndarray,
                    sr: int,
                    idx: int) -> int:
    """
    Generate a single sample. Return 0 on success.

    Steps:
    - [1] Load voice
    - [2] Sample background with the same length as voice.
    - [3] Pick background location with r in [2.0, 5.0)
    - [4] Create a scene
    - [5] Render sound
    - [6] Save metadata
    """
    # [1]
    output_prefix_dir = os.path.join(args.output_path, '{:05d}'.format(idx))
    Path(output_prefix_dir).mkdir(parents=True, exist_ok=True)

    all_voices = Path(args.input_voice_dir).rglob('*.wav')
    all_voices = list(all_voices)

    num_voices = random.randint(1, 4)  # How many voices to have present
    print("Num voices {}".format(num_voices))
    voice_files = random.sample(all_voices, num_voices)

    voices_data = []
    for voice_file in voice_files:
        voice, _ = librosa.core.load(voice_file, sr=sr, mono=True)
        voices_data.append(voice)

    # [2]
    max_voice_length = max([len(voice) for voice in voices_data])
    bg_length = len(bg)
    bg_start_idx = np.random.randint(bg_length - max_voice_length) # [0, bg_length - voice_length)
    sample_bg = bg[bg_start_idx: bg_start_idx + max_voice_length]

    # Generate room parameters
    left_wall = np.random.uniform(low=-20, high=-6.5)
    right_wall = np.random.uniform(low=20, high=6.5)
    top_wall = np.random.uniform(low=20, high=6.5)
    bottom_wall = np.random.uniform(low=-20, high=-6.5)
    corners = np.array([[left_wall, bottom_wall], [left_wall, top_wall],
                        [right_wall, top_wall], [right_wall, bottom_wall]]).T
    absorption = np.random.uniform(low=0.1, high=0.99)

    


    # FG
    all_fg_signals = []
    voice_positions = []
    for voice_idx in range(num_voices):
        room = pra.Room.from_corners(corners, fs=sr, max_order=10, absorption=absorption)
        mic_array = generate_mic_array(room=room)

        voice_radius = np.random.uniform(low=1.0, high=5.0)
        voice_theta = np.random.uniform(low=0, high=2*np.pi)
        voice_loc = [voice_radius * np.cos(voice_theta), voice_radius * np.sin(voice_theta)]
        voice_positions.append(voice_loc)
        room.add_source(voice_loc, signal=voices_data[voice_idx])

        room.image_source_model(use_libroom=True)
        room.simulate()
        fg_signals = room.mic_array.signals[:, :max_voice_length]
        fg_target = np.clip(np.random.uniform(FG_TARGET_VOL, 0.15), 0, 1)
        fg_signals = fg_signals * fg_target / fg_signals.max()
        all_fg_signals.append(fg_signals)


    # BG
    bg_radius = np.random.uniform(low=2.0, high=6.0)
    bg_theta = np.random.uniform(low=0, high=2*np.pi)
    bg_loc = [bg_radius * np.cos(bg_theta), bg_radius * np.sin(bg_theta)]

    room = pra.Room.from_corners(corners, fs=sr, max_order=10, absorption=absorption)
    mic_array = generate_mic_array(room=room)
    room.add_source(bg_loc, signal=sample_bg)
    
    room.image_source_model(use_libroom=True)
    room.simulate()
    bg_signals = room.mic_array.signals[:, :max_voice_length]
    bg_target = np.clip(np.random.uniform(BG_TARGET_VOL, 0.2), 0, 1)
    bg_signals = bg_signals * bg_target / bg_signals.max()


    # Save
    total_samples = int(args.duration * sr)
    for mic_idx in range(N_MICS):
        output_prefix = str(Path(output_prefix_dir) / "mic{:02d}_".format(mic_idx))

        # Save FG
        all_fg_buffer = np.zeros((total_samples))
        for voice_idx in range(num_voices):
            curr_fg_buffer = np.pad(all_fg_signals[voice_idx][mic_idx], (0, total_samples))[:total_samples]
            sf.write(output_prefix + "voice{:02d}.wav".format(voice_idx), curr_fg_buffer, sr)
            all_fg_buffer += curr_fg_buffer

        bg_buffer = np.pad(bg_signals[mic_idx], (0, total_samples))[:total_samples]
        sf.write(output_prefix + "bg.wav", bg_buffer, sr)

        sf.write(output_prefix + "mixed.wav", all_fg_buffer + bg_buffer, sr)

    # fig, ax = room.plot()
    # ax.set_xlim([-30, 30])
    # ax.set_ylim([-30, 30])

    # plt.savefig("fig.png")

    # [6]
    metadata = {}
    for voice_idx in range(num_voices):
        metadata['voice{:02d}'.format(voice_idx)] = {
            'position': voice_positions[voice_idx]
        }
    metadata['bg'] = {
        'position': bg_loc
    }

    metadata_file = str(Path(output_prefix_dir) / "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

def main(args: argparse.Namespace):
    np.random.seed(args.seed)
    base_background, _ = librosa.core.load(args.input_background_path, sr=args.sr, mono=True)

    pbar = tqdm.tqdm(total=args.n_outputs)
    pool = mp.Pool(args.n_workers)
    callback_fn = lambda _: pbar.update()
    for i in range(args.n_outputs):
        pool.apply_async(generate_sample,
                         args=(args, base_background, args.sr, i),
                         callback=callback_fn,
                         error_callback=handle_error)
    pool.close()
    pool.join()
    pbar.close()


# sf.write("output.wav", np.transpose(room.mic_array.signals) / 2**15, 48000)
# fig, ax = room.plot()
# ax.set_xlim([-30, 30])
# ax.set_ylim([-30, 30])

# plt.savefig("fig.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_voice_dir', type=str, default="base_voice.wav")
    parser.add_argument('--input_background_path', type=str, default="base_background.wav")
    parser.add_argument('--output_path', type=str, default='./outputs')
    parser.add_argument('--n_outputs', type=int, default=10000)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sr', type=int, default=44100)
    parser.add_argument('--duration', type=float, default=3.0)
    main(parser.parse_args())
