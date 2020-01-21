import argparse
import multiprocessing as mp
from pathlib import Path
import numpy as np
import random
import librosa
import tqdm
import os
import json

import pprint
import time


from classes import Microphone, SoundSource, Scene, INPUT_OUTPUT_TARGET_SAMPLE_RATE

def generate_mic_array(args):
    mic_array = []
    radius = 0.145 / 2 # 50th percentile of Bitragion breadth (From Wikipedia)
    for i in range(2):
        position_x = radius * np.cos(2 * np.pi / 2 * i)
        position_y = radius * np.sin(2 * np.pi / 2 * i)
        position_z = 0
        mic_array.append(Microphone([position_x, position_y, position_z]))
    return mic_array

def generate_sample(args, idx):
    # Generate mic array
    mic_array = generate_mic_array(args)
    metadata = {}
    all_sources = []
    output_prefix_dir = os.path.join(args.outputs_dir, '{:05d}'.format(idx))

    # Pick the voices
    all_voices = Path(args.voices_dir).rglob('*.' + args.voice_format)
    voice_files = random.sample(list(all_voices), args.n_voices_per_sample)
    for i, voice_file in enumerate(voice_files):
        random_x = np.random.uniform(-1.0, 1.0)
        random_y = np.random.uniform(0, 1.0)
        sound_source_voice = SoundSource(position=[random_x, random_y, 0],
                                         filename=str(voice_file))
        all_sources.append(sound_source_voice)
        metadata["source{:02d}".format(i)] = {
            "position": [random_x, random_y, 0],
            "filename": os.path.join(args.outputs_dir, '{:05d}'.format(idx), 'gt_voice_{:02d}.wav'.format(i))
        }

    # Select the background
    all_backgrounds = Path(args.backgrounds_dir).rglob('*.' + args.background_format)
    unique_background_stems = list(set(map(lambda x: '_'.join(x.stem.split('_')[:-1]), list(all_backgrounds))))

    background_stem = random.choice(unique_background_stems)
    possible_background_files = list(Path(args.backgrounds_dir).rglob('{}*.{}'.format(background_stem, args.background_format)))
    background_files = random.sample(list(possible_background_files), args.n_backgrounds_per_sample)


    for i, background_file in enumerate(background_files):
        random_r = np.random.uniform(0.5, 5.0)
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
    scene.render(cutoff_time=args.scene_duration)

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

def main(args):
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
    parser.add_argument('--n-backgrounds-per-sample', type=int, default=20, help="Number of backgrounds per sample")
    parser.add_argument('--scene_duration', type=float, default=6.0, help="Duration of the scene")
    parser.add_argument('--n-workers', type=int, default=4, help="Number of multiprocessor workers")

    main(parser.parse_args())
