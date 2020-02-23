import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import argparse
import json
from typing import List
from pathlib import Path
import numpy as np
import librosa
import tqdm
import random
import multiprocessing.dummy as mp

from classes import Microphone, SoundSource, Scene

def generate_mic_array(radius: float = 0.145 / 2, n_mics: int = 6) -> List[Microphone]:
    """
    Generate a list of Microphone objects

    Radius = 50th percentile of men Bitragion breadth
    (https://en.wikipedia.org/wiki/Human_head)
    """
    mic_array = []
    for i in range(n_mics):
        position_x = radius * np.cos(2 * np.pi / n_mics * i)
        position_y = radius * np.sin(2 * np.pi / n_mics * i)
        position_z = 0
        mic_array.append(Microphone([position_x, position_y, position_z]))
    return mic_array

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
    output_prefix_dir = os.path.join(args.output_path, '{:05d}'.format(idx))

    # [1]
    all_voices = Path(args.input_voice_dir).rglob('*.wav')
    voice_files = random.sample(list(all_voices), 1)
    voice, _ = librosa.core.load(voice_files[0], sr=sr, mono=True)

    # [2]
    voice_length = len(voice)
    bg_length = len(bg)
    bg_start_idx = np.random.randint(bg_length - voice_length) # [0, bg_length - voice_length)
    sample_bg = bg[bg_start_idx: bg_start_idx + voice_length] * args.bg_factor

    # [3]
    r = np.random.uniform(low=2.0, high=5.0)
    theta = np.random.uniform(low=0, high=2*np.pi)

    # [4]
    mic_array = generate_mic_array()
    voice_loc = [1.0 * np.cos(0), 1.0 * np.sin(0), 0]
    bg_loc = [r * np.cos(theta), r * np.sin(theta), 0]
    voice = SoundSource(position=voice_loc, data=voice, sr=sr)
    bg = SoundSource(position=bg_loc, data=sample_bg, sr=sr)
    scene = Scene([voice, bg], mic_array)

    # [5]
    scene.render(cutoff_time=3.0)
    Path(output_prefix_dir).mkdir(parents=True, exist_ok=True)
    for i, mic in enumerate(mic_array):
        output_prefix = str(Path(output_prefix_dir) / "mic{:02d}_".format(i))
        mic.save(output_prefix)
        mic.reset()

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

    return 0

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_voice_dir', type=str, default="base_voice.wav")
    parser.add_argument('--input_background_path', type=str, default="base_background.wav")
    parser.add_argument('--output_path', type=str, default='./outputs')
    parser.add_argument('--n_outputs', type=int, default=10000)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--bg_factor', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sr', type=int, default=22050)
    main(parser.parse_args())

"""
python generate.py --output_path /projects/grail/audiovisual/datasets/DinTaiFung/simple_1_voice_1_bg/train --n_outputs 10000
python generate.py --output_path /projects/grail/audiovisual/datasets/DinTaiFung/simple_1_voice_1_bg/test --n_outputs 2000

python generate.py \
    --input_voice_dir /projects/grail/audiovisual/datasets/VCTK-Corpus/wav48/ \
    --output_path /projects/grail/audiovisual/datasets/DinTaiFung/1_voice_1_bg_hard/train \
    --bg_factor 10.0 \
    --n_outputs 10000
python generate.py \
    --input_voice_dir /projects/grail/audiovisual/datasets/VCTK-Corpus/wav48/ \
    --output_path /projects/grail/audiovisual/datasets/DinTaiFung/1_voice_1_bg_hard/test \
    --bg_factor 10.0 \
    --n_outputs 2000
"""