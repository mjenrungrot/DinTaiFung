import argparse
from pathlib import Path
import tqdm
import os
import multiprocessing as mp
import librosa
import soundfile as sf
import numpy as np

from classes import INPUT_OUTPUT_TARGET_SAMPLE_RATE

def generate_sample(args, stem, data, sr, idx):
    output_path = os.path.join(args.outputPath, '{:}_{:04d}.wav'.format(stem, idx))
    random_start_idx = np.random.randint(0, data.shape[0] - int(np.floor(args.duration * sr)))
    random_end_idx = random_start_idx + int(np.floor(args.duration * sr))
    sample = data[random_start_idx:random_end_idx]
    sf.write(output_path, sample, sr)

def handle_error(e):
    print(e)

def main(args):
    ambient_files = Path(args.path).rglob('*.wav')
    os.makedirs(args.outputPath, exist_ok=True)
    for ambient_file in tqdm.tqdm(ambient_files):
        data, sr = librosa.core.load(str(ambient_file),
                                    sr=INPUT_OUTPUT_TARGET_SAMPLE_RATE,
                                    mono=True)

        print("Data shape = {}".format(data.shape))
        pbar = tqdm.tqdm(total=args.n_samples_per_input)
        update = lambda x: pbar.update()
        pool = mp.Pool(args.n_workers)
        for i in range(args.n_samples_per_input):
            pool.apply_async(generate_sample, args=(args, str(ambient_file.stem), data, sr, i), callback=update, error_callback=handle_error)
        pool.close()
        pool.join()
        pbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Path to background files")
    parser.add_argument('outputPath', type=str, help="Path to the outputs")
    parser.add_argument('--n-samples-per-input', type=int, default=1000)
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--n-workers', type=int, default=4)

    main(parser.parse_args())