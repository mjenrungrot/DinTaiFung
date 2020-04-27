import argparse
import random
from statistics import median

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

from data import SpatialAudioDatasetWaveform, RealDataset, MixedDataset
from network import center_trim, Demucs


def draw_plot(input_stats, output_stats, name, label, step_size=1.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ave_inputs = input_stats
    ave_outputs = output_stats

    #ave_inputs = np.array([x.mean() for x in input_stats])
    #ave_outputs = np.array([x.mean() for x in input_stats])

    x_coords = []
    y_coords = []
    for base_bin in np.arange(ave_inputs.min(), ave_inputs.max(), step_size):
        output_sum = []
        for i in range(len(ave_inputs)):
            if base_bin < ave_inputs[i] < (base_bin + step_size):
                output_sum.append(ave_outputs[i])

        if len(output_sum) > 0:
            #output_mean = sum(output_sum) / len(output_sum)
            output_med = median(output_sum)
            x_coords.append(base_bin + step_size/2)
            y_coords.append(output_med)

    plt.plot(x_coords, y_coords, label=label)
    plt.xlabel("Input {} db".format(name))
    plt.ylabel("Output {} db".format(name))
    plt.savefig("{} Comparison.png".format(name))


def main(args):
    # input_stats = np.load("input_sdr.npy")
    # output_stats = np.load("output_sdr.npy")
    # draw_plot(input_stats, output_stats, "sdr")

    kwargs = {
        'num_workers': args.n_workers,
        'pin_memory': True
    }

    #data_test = RealDataset(args.test_dir, sr=args.sr)
    data_test = SpatialAudioDatasetWaveform(args.test_dir,
        n_mics=args.n_channels,
        sr=args.sr, target_fg_std=None,
        target_bg_std=None, perturb_prob=0.0)
    device = torch.device('cuda:0')

    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              **kwargs)

    model = Demucs(sources=2, n_audio_channels=args.n_channels)
    model.load_state_dict(torch.load(args.model_checkpoint))
    model.train = False
    model.to(device)

    all_input_sdr = []
    all_output_sdr = []

    all_input_sar = []
    all_output_sar = []

    all_input_sir = []
    all_output_sir = []

    try:
        with torch.no_grad():
            for batch_idx, (data, label_voice_signals) in enumerate(test_loader):
                print(batch_idx)

                data = data.to(device)
                label_voice_signals = label_voice_signals.to(device)

                # Normalize input
                data_transformed = (data * 2**15).round() / 2**15
                ref = data_transformed.mean(0)
                data_transformed = (data_transformed - ref.mean()) / (ref.std())

                # Run through the model
                valid_length = model.valid_length(data_transformed.shape[-1])
                delta = valid_length - data_transformed.shape[-1]
                padded = F.pad(data_transformed, (delta // 2, delta - delta // 2))

                output_signal, output_locs = model(padded)
                output_signal = center_trim(output_signal, data_transformed)
                output_locs = center_trim(output_locs, data_transformed)
                
                output_signal = output_signal * ref.std() + ref.mean()
                output_voices = output_signal[:, 0]

                if args.n_channels == 2:
                    channels = [0, 1]
                elif args.n_channels == 6:
                    channels = [0, 3]

                for i in range(label_voice_signals.shape[0]):
                    input_sdr, input_sir, input_sar, input_perm = mir_eval.separation.bss_eval_sources(label_voice_signals[i, 0, channels].cpu().numpy(), data[i, channels].cpu().numpy(), compute_permutation=False)
                    output_sdr, output_sir, output_sar, output_perm = mir_eval.separation.bss_eval_sources(label_voice_signals[i, 0, channels].cpu().numpy(), output_voices[i, channels].cpu().numpy(), compute_permutation=False)

                    all_input_sdr.append(input_sdr)
                    all_input_sir.append(input_sir)
                    all_input_sar.append(input_sar)

                    all_output_sdr.append(output_sdr)
                    all_output_sir.append(output_sir)
                    all_output_sar.append(output_sar)
                    #print("idx {} input SDR {} output SDR {}".format(i, input_sdr, output_sdr))
        joint_data = np.stack((np.array(all_input_sdr), np.array(all_output_sdr)), axis=2)
        np.save("ours_sdr_{}mics.npy".format(args.n_channels), joint_data)

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
    print(parser.parse_args())
    main(parser.parse_args())
