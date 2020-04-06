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
    voice_files = random.sample(all_voices, 1)
    voice, _ = librosa.core.load(voice_files[0], sr=sr, mono=True)

    # [2]
    voice_length = len(voice)
    bg_length = len(bg)
    bg_start_idx = np.random.randint(bg_length - voice_length) # [0, bg_length - voice_length)
    sample_bg = bg[bg_start_idx: bg_start_idx + voice_length]

    # [3]
    bg_radius = np.random.uniform(low=2.0, high=6.0)
    bg_theta = np.random.uniform(low=0, high=2*np.pi)

    voice_radius = np.random.uniform(low=1.0, high=5.0)
    voice_theta = np.random.uniform(low=0, high=2*np.pi)

    # [4]
    # Generate room
    left_wall = np.random.uniform(low=-20, high=-6.5)
    right_wall = np.random.uniform(low=20, high=6.5)
    top_wall = np.random.uniform(low=20, high=6.5)
    bottom_wall = np.random.uniform(low=-20, high=-6.5)
    corners = np.array([[left_wall, bottom_wall], [left_wall, top_wall],
                        [right_wall, top_wall], [right_wall, bottom_wall]]).T
    absorption = np.random.uniform(low=0.1, high=0.99)
    voice_loc = [voice_radius * np.cos(voice_theta), voice_radius * np.sin(voice_theta)]
    bg_loc = [bg_radius * np.cos(bg_theta), bg_radius * np.sin(bg_theta)]

    # FG
    room = pra.Room.from_corners(corners, fs=sr, max_order=10, absorption=absorption)
    mic_array = generate_mic_array(room=room)
    room.add_source(voice_loc, signal=voice)
    
    room.image_source_model(use_libroom=True)
    room.simulate()
    fg_signals = room.mic_array.signals[:, :voice_length]
    fg_target = np.clip(np.random.uniform(FG_TARGET_VOL, 0.15), 0, 1)
    fg_signals = fg_signals * fg_target / fg_signals.max()

    # BG
    room = pra.Room.from_corners(corners, fs=sr, max_order=10, absorption=absorption)
    mic_array = generate_mic_array(room=room)
    room.add_source(bg_loc, signal=sample_bg)
    
    room.image_source_model(use_libroom=True)
    room.simulate()
    bg_signals = room.mic_array.signals[:, :voice_length]
    bg_target = np.clip(np.random.uniform(BG_TARGET_VOL, 0.2), 0, 1)
    bg_signals = bg_signals * bg_target / bg_signals.max()

    total_samples = int(args.duration * sr)

    for mic_idx in range(N_MICS):
        fg_buffer = np.pad(fg_signals[mic_idx], (0, total_samples))[:total_samples]
        bg_buffer = np.pad(bg_signals[mic_idx], (0, total_samples))[:total_samples]

        output_prefix = str(Path(output_prefix_dir) / "mic{:02d}_".format(mic_idx))
        sf.write(output_prefix + "source00_gt.wav", fg_buffer, sr)
        sf.write(output_prefix + "source01_gt.wav", bg_buffer, sr)
        sf.write(output_prefix + "mixed.wav", bg_buffer + fg_buffer, sr)

    # fig, ax = room.plot()
    # ax.set_xlim([-30, 30])
    # ax.set_ylim([-30, 30])

    # plt.savefig("fig.png")

    # [6]
    metadata = {}
    metadata['voice'] = {
        'position': voice_loc
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
