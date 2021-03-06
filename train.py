import argparse
from typing import Dict, List, Tuple, Optional # pylint: disable=unused-import
from pathlib import Path
import os
import multiprocessing
import numpy as np
import tqdm # pylint: disable=unused-import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.tensorboard as tensorboard

from data import SpatialAudioDatasetWaveform
from network import Demucs, center_trim, load_pretrain # pylint: disable=unused-import

SAMPLING_RATE = 22050
USE_CUDA = True

def train_epoch(model: nn.Module,
                device: torch.device,
                optimizer: optim.Optimizer,
                train_loader: torch.utils.data.dataloader.DataLoader,
                epoch: int,
                log_interval: int = 20,
                data_parallel: bool = False,
                writer: Optional[tensorboard.writer.SummaryWriter] = None) -> float:
    """
    Train a single epoch.
    """
    # Set the model to training.
    model.train()

    # Training loop
    losses = []
    interval_losses = []
    voice_losses = []
    bg_losses = []
    combined_losses = []
    for batch_idx, (data, label_voice_signals, label_bg_signals, label_voice_locs, label_bg_locs) in enumerate(train_loader):
        data = data.to(device)
        label_voice_signals = label_voice_signals.to(device)
        label_bg_signals = label_bg_signals.to(device)
        label_voice_locs = label_voice_locs.to(device)
        label_bg_locs = label_bg_locs.to(device)

        # Normalize input
        data = (data * 2**15).round() / 2**15
        ref = data.mean(0)
        data = (data - ref.mean()) / ref.std()

        # Reset grad
        optimizer.zero_grad()

        # Run through the model
        if data_parallel:
            valid_length = model.module.valid_length(data.shape[-1])
        else:
            valid_length = model.valid_length(data.shape[-1])
        delta = valid_length - data.shape[-1]
        padded = F.pad(data, (delta // 2, delta - delta // 2))

        output_signal, output_locs = model(padded)
        output_signal = center_trim(output_signal, data)
        output_locs = center_trim(output_locs, data)

        # Un-normalize
        output_signal = output_signal * ref.std() + ref.mean()
        output_voices = output_signal[:, :label_voice_signals.shape[1]]
        output_voice_locs = output_locs[:, :label_voice_locs.shape[1]]
        output_backgrounds = output_signal[:, label_voice_signals.shape[1] : label_voice_signals.shape[1] + label_bg_signals.shape[1]]
        output_background_locs = output_locs[:, label_voice_locs.shape[1] : label_voice_locs.shape[1] + label_bg_locs.shape[1]]

        if data_parallel:
            loss, info = model.module.loss(data,
                                     output_voices, label_voice_signals,
                                     output_backgrounds, label_bg_signals,
                                     output_voice_locs, label_voice_locs,
                                     output_background_locs, label_bg_locs)
        else:
            loss, info = model.loss(data,
                              output_voices, label_voice_signals,
                              output_backgrounds, label_bg_signals,
                              output_voice_locs, label_voice_locs,
                              output_background_locs, label_bg_locs)
        interval_losses.append(loss.item())
        voice_losses.append(info['reconstruction_voices_loss'].item())
        bg_losses.append(info['reconstruction_bg_loss'].item())
        combined_losses.append(info['reconstruction_combined_loss'].item())

        # Backpropagation
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        # Update the weights
        optimizer.step()

        if batch_idx % log_interval == 0:
            print("Loss {}".format(loss))
            print("Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(interval_losses)))

            if writer:
                writer.add_scalar('Loss/train_batch', loss.item())
                writer.flush()

            losses.extend(interval_losses)
            interval_losses = []

    # Write loss to the Tensorboard
    if writer:
        writer.add_scalar('Loss/train', np.mean(losses), epoch)
        writer.add_scalar('Loss_voice/train', np.mean(voice_losses), epoch)
        writer.add_scalar('Loss_bg/train', np.mean(bg_losses), epoch)
        writer.add_scalar('Loss_combined/train', np.mean(combined_losses), epoch)
        writer.flush()

    return np.mean(losses)

def test_epoch(model: nn.Module,
               device: torch.device,
               test_loader: torch.utils.data.dataloader.DataLoader,
               epoch: int,
               log_interval: int = 20,
               data_parallel: bool = False,
               writer: Optional[tensorboard.writer.SummaryWriter] = None) -> float:
    """
    Evaluate the network.
    """
    model.eval()
    test_loss = 0

    voice_losses = []
    bg_losses = []
    combined_losses = []
    with torch.no_grad():
        for batch_idx, (data, label_voice_signals, label_bg_signals, label_voice_locs, label_bg_locs) in enumerate(test_loader):
            data = data.to(device)
            label_voice_signals = label_voice_signals.to(device)
            label_bg_signals = label_bg_signals.to(device)
            label_voice_locs = label_voice_locs.to(device)
            label_bg_locs = label_bg_locs.to(device)

            # Normalize input
            transformed_data = (data * 2**15).round() / 2**15
            ref = transformed_data.mean(0)
            transformed_data = (transformed_data - ref.mean()) / ref.std()

            # Run through the model
            if data_parallel:
                valid_length = model.module.valid_length(transformed_data.shape[-1])
            else:
                valid_length = model.valid_length(transformed_data.shape[-1])
            delta = valid_length - transformed_data.shape[-1]
            padded = F.pad(transformed_data, (delta // 2, delta - delta // 2))

            output_signal, output_locs = model(padded)
            output_signal = center_trim(output_signal, transformed_data)
            output_locs = center_trim(output_locs, transformed_data)

            # Un-normalize
            output_signal = output_signal * ref.std() + ref.mean()
            output_voices = output_signal[:, :label_voice_signals.shape[1]]
            output_voice_locs = output_locs[:, :label_voice_locs.shape[1]]
            output_backgrounds = output_signal[:, label_voice_signals.shape[1] : label_voice_signals.shape[1] + label_bg_signals.shape[1]]
            output_background_locs = output_locs[:, label_voice_locs.shape[1] : label_voice_locs.shape[1] + label_bg_locs.shape[1]]

            if data_parallel:
                loss, info = model.module.loss(data,
                                         output_voices, label_voice_signals,
                                         output_backgrounds, label_bg_signals,
                                         output_voice_locs, label_voice_locs,
                                         output_background_locs, label_bg_locs)
            else:
                loss, info = model.loss(data,
                                  output_voices, label_voice_signals,
                                  output_backgrounds, label_bg_signals,
                                  output_voice_locs, label_voice_locs,
                                  output_background_locs, label_bg_locs)
            test_loss += loss
            voice_losses.append(info['reconstruction_voices_loss'].item())
            bg_losses.append(info['reconstruction_bg_loss'].item())
            combined_losses.append(info['reconstruction_combined_loss'].item())

            if batch_idx % log_interval == 0:
                print("Loss: {}".format(loss))
                if writer:
                    for data_id in range(min(2, data.size(0))): # Output only maximum 2 audios
                        for mic_id in range(data.size(1)):
                            writer.add_audio("{}_input_{}".format(data_id, mic_id), data[data_id, mic_id, :], epoch, sample_rate=SAMPLING_RATE)
                            writer.add_audio("{}_gt_{}".format(data_id, mic_id), label_voice_signals[data_id, 0, mic_id, :], epoch, sample_rate=SAMPLING_RATE)
                            writer.add_audio("{}_pred_{}".format(data_id, mic_id), output_voices[data_id, 0, mic_id, :], epoch, sample_rate=SAMPLING_RATE)

        test_loss /= len(test_loader)
        print("\nTest set: Average Loss: {:.4f}\n".format(test_loss))

        # Write loss to the Tensorboard
        if writer:
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Loss_voice/test', np.mean(voice_losses), epoch)
            writer.add_scalar('Loss_bg/test', np.mean(bg_losses), epoch)
            writer.add_scalar('Loss_combined/test', np.mean(combined_losses), epoch)
            writer.flush()

        return test_loss

def train(args: argparse.Namespace):
    """
    Train the network.
    """
    # Load dataset
    data_train = SpatialAudioDatasetWaveform(args.train_dir, n_sources=1, n_backgrounds=1)
    data_test = SpatialAudioDatasetWaveform(args.test_dir, n_sources=1, n_backgrounds=1)

    # Set up the device and workers.
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else 'cpu')
    num_workers = min(multiprocessing.cpu_count(), args.n_workers)
    kwargs = {'num_workers': num_workers,
              'pin_memory': True} if use_cuda else {}

    # Set up data loaders
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=args.batch_size,
                                               shuffle=args.shuffle,
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(data_test,
                                               batch_size=args.batch_size,
                                               **kwargs)

    # Set up model
    model = Demucs(sources=args.n_sources, n_audio_channels=args.n_channels)

    # Data parallel
    data_parallel = False
    if args.multiple_GPUs:
        data_parallel = True
        model = nn.DataParallel(model, device_ids=args.device_ids)
    model.to(device)

    # Set up checkpoints
    save_prefix = args.name
    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)

    # Set up the optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.decay)

    # Load pretrain
    if args.pretrain:
        pretrain_state_dict = torch.load('models/demucs_extra_state_dict.pt')
        load_pretrain(model, pretrain_state_dict)

    # Load the model if `args.start_epoch` is greater than 0. This will load the model from
    # epoch = `args.start_epoch - 1`
    if args.start_epoch is not None:
        assert args.start_epoch > 0, "start_epoch must be greater than 0."
        start_epoch = args.start_epoch
        checkpoint_path = Path(args.checkpoints_dir) / "{}_{}.pt".format(save_prefix, start_epoch - 1)
        state_dict = torch.load(checkpoint_path)
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
    else:
        start_epoch = 0

    # Set up tensorboard writer
    if args.tensorboard:
        writer = tensorboard.SummaryWriter(log_dir=args.log_dir + '/' + args.name, flush_secs=10)
        writer.add_hparams(hparam_dict={
            'batch_size': args.batch_size,
            'n_channels': args.n_channels,
            'epochs': args.epochs,
            'lr': args.lr,
            'decay': args.decay,
            'n_workers': args.n_workers
        }, metric_dict={})
        writer.flush()
    else:
        writer = None

    # Loss values
    train_losses = []
    test_losses = []

    # Training loop
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss = train_epoch(model, device, optimizer, train_loader, epoch, args.print_interval, data_parallel=data_parallel, writer=writer)
            test_loss = test_epoch(model, device, test_loader, epoch, args.print_interval, data_parallel=data_parallel, writer=writer)
            train_losses.append((epoch, train_loss))
            test_losses.append((epoch, test_loss))
            if args.multiple_GPUs:
                torch.save(model.module.state_dict(), os.path.join(args.checkpoints_dir, "{}_{}.pt".format(save_prefix, epoch)))
            else:
                torch.save(model.state_dict(), os.path.join(args.checkpoints_dir, "{}_{}.pt".format(save_prefix, epoch)))
    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as _: # pylint: disable=broad-except
        import traceback # pylint: disable=import-outside-toplevel
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dir', type=str, help="Path to the training dataset")
    parser.add_argument('test_dir', type=str, help="Path to the testing dataset")
    parser.add_argument('--name', type=str, default="DinTaiFung", help="Name of the experiment")
    parser.add_argument('--log_dir', type=str, default='./logs', help="Path to the log directory")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--n_sources', type=int, default=1, help="Number of sources")
    parser.add_argument('--n_channels', type=int, default=2, help="Number of channels")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=3e-4, help="learning rate")
    parser.add_argument('--sr', type=int, default=22050, help="Sampling rate")
    parser.add_argument('--decay', type=float, default=0, help="Weight decay")
    parser.add_argument('--n_workers', type=int, default=16, help="Number of parallel workers")
    parser.add_argument('--print_interval', type=int, default=20, help="Logging interval")
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help="Path to the checkpoints")
    parser.add_argument('--start_epoch', type=int, default=None, help="Start epoch")
    parser.add_argument('--pretrain', dest='pretrain', action='store_true', help="Whether to load pretrained weights from the original implementaiton")
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', help="Whether to shuffle the training split")
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true', help="Whether to use cuda")
    parser.add_argument('--device', type=str, default='cpu', help="Device for pytorch")
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help="Whether to use tensorboard")
    parser.add_argument('--multiple_GPUs', dest='multiple_GPUs', action='store_true', help="Whether to use multiple GPUs")
    parser.add_argument('--device_ids', default=[], nargs='+', help="IDs for multiple GPUs")
    print(parser.parse_args())
    train(parser.parse_args())

"""
CUDA_VISIBLE_DEVICES=1,2 python train.py \
    /projects/grail/audiovisual/datasets/DinTaiFung/1_voice_1_bg_hard/train \
    /projects/grail/audiovisual/datasets/DinTaiFung/1_voice_1_bg_hard/test \
    --name 1_voice_1_bg_hard \
    --n_sources 2 \
    --lr 1e-4 \
    --n_channels 6 \
    --batch_size 40 \
    --epochs 50 \
    --use_cuda \
    --shuffle \
    --device cuda:0 \
    --tensorboard \
    --multiple_GPUs --device_ids cuda:0 cuda:1
""" # pylint: disable=pointless-string-statement
