import time

import librosa
import numpy as np

from mir_eval.separation import bss_eval_sources
import math

from itertools import permutations


def compute_angle_error(output_angles, gt_angles):
    assert(len(gt_angles) == len(output_angles))

    loss1 = abs(gt_angles[0] - output_angles[0])
    loss2 = abs(gt_angles[1] - output_angles[1])
    loss3 = abs(gt_angles[0] - output_angles[1])
    loss4 = abs(gt_angles[1] - output_angles[0])

    if (loss1 + loss2) < (loss3 + loss4):
        return [loss1, loss2]
    else:
        return [loss3, loss4]

def compute_sdr(estimated_signal, gt_signals):
    """
    Only handles two sources for now
    estimated and gt signals need to be 2 x mics x time
    """
    loss1 = compute_sdr_helper(estimated_signal[0], gt_signals[0])
    loss2 = compute_sdr_helper(estimated_signal[1], gt_signals[1])
    loss3 = compute_sdr_helper(estimated_signal[1], gt_signals[0])
    loss4 = compute_sdr_helper(estimated_signal[0], gt_signals[1])

    if (loss1.mean() + loss2.mean()) > (loss3.mean() + loss4.mean()):
        return [loss1, loss2]
    else:
        return [loss3, loss4]

def compute_sdr_helper(estimated_signal, gt_signals):
    assert(estimated_signal.shape == gt_signals.shape)
    all_sdr = []
    for channel_idx in range(estimated_signal.shape[0]):
        all_sdr.append(compute_measures(estimated_signal[channel_idx], np.expand_dims(gt_signals[channel_idx], 1), 0)[0])

    return np.array(all_sdr)

def compute_measures(estimated_signal, reference_signals, j, scaling=True):
    Rss= np.dot(reference_signals.transpose(), reference_signals)
    this_s= reference_signals[:,j]

    if scaling:
        # get the scaling factor for clean sources
        a= np.dot( this_s, estimated_signal) / Rss[j,j]
    else:
        a= 1

    e_true= a * this_s
    e_res= estimated_signal - e_true

    Sss= (e_true**2).sum()
    Snn= (e_res**2).sum()

    SDR= 10 * math.log10(Sss/Snn)
    
    # Get the SIR
    Rsr= np.dot(reference_signals.transpose(), e_res)
    b= np.linalg.solve(Rss, Rsr)

    e_interf= np.dot(reference_signals , b)
    e_artif= e_res - e_interf
    
    SIR= 10 * math.log10(Sss / (e_interf**2).sum())
    SAR= 10 * math.log10(Sss / (e_artif**2).sum())

    return SDR, SIR,SAR

# def SDR(estimated_signal, reference_signals)


# mixed_file = "/projects/grail/audiovisual/datasets/DinTaiFung/librispeech_6mics_multiplevoices/test/00004/mic00_mixed.wav"
# voice_file = "/projects/grail/audiovisual/datasets/DinTaiFung/librispeech_6mics_multiplevoices/test/00004/mic00_voice00.wav"

# mixed, _ = librosa.core.load(mixed_file, sr=22050)
# voice, _ = librosa.core.load(voice_file, sr=22050)

# # import pdb
# # pdb.set_trace()

# t0 = time.time()

# sdr1, _, _, _ = bss_eval_sources(voice, mixed)
# sdr2, _, _, _ = bss_eval_sources(10 * voice, mixed)
# sdr3, _, _, _ = bss_eval_sources(voice, 10 * mixed)

# print(time.time() - t0)
# t0 = time.time()

# print("Mir Eval Results:")
# print(sdr1[0])
# print(sdr2[0])
# print(sdr3[0])

# mixed = np.expand_dims(mixed, 1)


# sdr1, _, _ = compute_measures(voice, mixed, j=0)
# sdr2, _, _ = compute_measures(10 * voice, mixed, j=0)
# sdr3, _, _ = compute_measures(voice, 10 * mixed, j=0)

# print(time.time() - t0)

# print("Custom Results:")
# print(sdr1)
# print(sdr2)
# print(sdr3)
