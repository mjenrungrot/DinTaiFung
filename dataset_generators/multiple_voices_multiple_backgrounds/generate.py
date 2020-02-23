"""
To generate a dataset, run

```
python generate.py \
    /projects/grail/audiovisual/datasets/VCTK-Corpus/wav48/p239 \
    /projects/grail/audiovisual/datasets/AudioSet/no_voices_train \
    /projects/grail/audiovisual/datasets/DinTaiFung/mics_8_radius_3_voice_1_bg_1/train \
    --n-samples 10000 \
    --n-voices-per-sample 1 \
    --n-backgrounds-per-sample 1 \
    --scene-duration 6.0 \
    --n-workers 16 \
    --radius 3.0 \
    --n-mics 8

python generate.py \
    /projects/grail/audiovisual/datasets/VCTK-Corpus/wav48/p239 \
    /projects/grail/audiovisual/datasets/AudioSet/no_voices_test \
    /projects/grail/audiovisual/datasets/DinTaiFung/mics_8_radius_3_voice_1_bg_1/test \
    --n-samples 2000 \
    --n-voices-per-sample 1 \
    --n-backgrounds-per-sample 1 \
    --scene-duration 6.0 \
    --n-workers 16 \
    --radius 3.0 \
    --n-mics 8
```
"""
import argparse
from typing import List
import multiprocessing as mp
import os
import json
import random
from pathlib import Path
import tqdm
import numpy as np
import librosa

from classes import Microphone, SoundSource, Scene, INPUT_OUTPUT_TARGET_SAMPLE_RATE

RADIUS_RANGE_FOR_SPEAKER = (3.5, 5)
RADIUS_RANGE_FOR_BG = (3.5, 10)

def generate_mic_array(args: argparse.Namespace) -> List[Microphone]:
    """
    Generate a list of Microphone objects
    """
    mic_array = []
    for i in range(args.n_mics):
        position_x = args.radius * np.cos(2 * np.pi / args.n_mics * i)
        position_y = args.radius * np.sin(2 * np.pi / args.n_mics * i)
        position_z = 0
        mic_array.append(Microphone([position_x, position_y, position_z]))
    return mic_array

def generate_sample(args: argparse.Namespace, idx: int) -> int:
    """
    Generate a single sample.

    Return 0 on success.
    """
    # Generate mic array
    mic_array = generate_mic_array(args)
    metadata = {}
    all_sources = []
    output_prefix_dir = os.path.join(args.outputs_dir, '{:05d}'.format(idx))

    # Pick the voices
    all_voices = Path(args.voices_dir).rglob('*.' + args.voice_format)
    voice_files = random.sample(list(all_voices), args.n_voices_per_sample)
    for i, voice_file in enumerate(voice_files):
        random_r = 1.5 # np.random.uniform(*RADIUS_RANGE_FOR_SPEAKER)
        random_theta = 0 # np.random.uniform(0, 2*np.pi)
        random_x = random_r * np.cos(random_theta)
        random_y = random_r * np.sin(random_theta)
        sound_source_voice = SoundSource(position=[random_x, random_y, 0],
                                         filename=str(voice_file))
        all_sources.append(sound_source_voice)
        metadata["source{:02d}".format(i)] = {
            "position": [random_x, random_y, 0],
            "filename": os.path.join(args.outputs_dir, '{:05d}'.format(idx), 'gt_voice_{:02d}.wav'.format(i))
        }

    # Select the background
    all_backgrounds = Path(args.backgrounds_dir).rglob('*.' + args.background_format)
    background_files = random.sample(list(all_backgrounds), args.n_backgrounds_per_sample)
    for i, background_file in enumerate(background_files):
        random_r = np.random.uniform(*RADIUS_RANGE_FOR_BG)
        random_theta = np.random.uniform(0, 2*np.pi)
        random_x = random_r * np.cos(random_theta)
        random_y = random_r * np.sin(random_theta)
        bg_data, bg_sr = librosa.core.load(str(background_file),
                                           sr=INPUT_OUTPUT_TARGET_SAMPLE_RATE,
                                           mono=True)

        sound_source_bg = SoundSource(position=[random_x, random_y, 0],
                                      data=bg_data,
                                      sr=bg_sr)
        all_sources.append(sound_source_bg)
        metadata["background{:02d}".format(i)] = {
            "position": [random_x, random_y, 0],
            "filename": str(background_file)
        }

    # Construct the scene
    scene = Scene(all_sources, mic_array)
    scene.render(cutoff_time=args.scene_duration,
                 geometric_attenuation=args.geometric_attenuation,
                 atmospheric_attenuation=args.atmospheric_attenuation)

    # Write all mics to output
    Path(output_prefix_dir).mkdir(parents=True, exist_ok=True)
    for i, mic in enumerate(mic_array):
        output_prefix = str(Path(output_prefix_dir) / "mic{:02d}_".format(i))
        mic.save(output_prefix)
        mic.reset()

    # # Write metadata
    metadata_file = str(Path(output_prefix_dir) / "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    return 0

def handle_error(error):
    print(error)

def main(args: argparse.Namespace):
    pbar = tqdm.tqdm(total=args.n_samples)
    update = lambda x: pbar.update()
    pool = mp.Pool(args.n_workers)
    for i in range(args.n_samples):
        pool.apply_async(generate_sample, args=(args, i,), callback=update, error_callback=handle_error)
    pool.close()
    pool.join()
    pbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('voices_dir', type=str, help="Path to the voices directory")
    parser.add_argument('backgrounds_dir', type=str, help="Path to the background sounds directory")
    parser.add_argument('outputs_dir', type=str, help="Path to the generated output directory")
    parser.add_argument('--voice-format', type=str, default='wav', help="File format of voice files")
    parser.add_argument('--background-format', type=str, default='wav', help="File format of background sounds")
    parser.add_argument('--n-samples', type=int, default=10, help="Number of samples")
    parser.add_argument('--n-voices-per-sample', type=int, default=1, help="Number of voices per sample")
    parser.add_argument('--n-backgrounds-per-sample', type=int, default=1, help="Number of backgrounds per sample")
    parser.add_argument('--scene-duration', type=float, default=6.0, help="Duration of the scene")
    parser.add_argument('--n-workers', type=int, default=4, help="Number of multiprocessor workers")
    parser.add_argument('--radius', type=float, default=3.0, help="Radius for mic array")
    parser.add_argument('--n-mics', type=int, default=8, help="Number of mics in the mic array")
    parser.add_argument('--no-geometric-attenuation', action='store_false', dest='geometric_attenuation')
    parser.add_argument('--no-atmospheric-attenuation', action='store_false', dest='atmospheric_attenuation')
    main(parser.parse_args())
