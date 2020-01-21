import argparse
import os
import multiprocessing
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data import SpatialAudioDatasetWaveform
from network import Demucs, center_trim

NUM_BGS = 20
USE_CUDA = True

def train_epoch(model, device, optimizer, train_loader, lr, epoch, log_interval=20):
    model.train()
    losses = []
    for batch_idx, (data, label) in enumerate(tqdm.tqdm(train_loader)):
        data = data.to(device)
        label = label.to(device)

        # Normalize input
        data = (data * 2**15).round() / 2**15
        ref = data.mean(0)
        data = (data - ref.mean()) / ref.std()

        # Reset grad
        optimizer.zero_grad()

        # Run through the model
        valid_length = model.valid_length(data.shape[-1])
        delta = valid_length - data.shape[-1]
        padded = F.pad(data, (delta // 2, delta - delta // 2))

        output_signal, output_locs = model(padded)
        output_signal_trimmed = center_trim(output_signal, data)
        output_locs_trimmed = center_trim(output_locs, data)

        import pdb; pdb.set_trace()

        # Un-normalize
        output = output_signal_trimmed * ref.std() + ref.mean()

        loss = model.loss(output, label)
        losses.append(loss.item())
        loss.backward()

        optimizer.step()

        if batch_idx % log_interval == 0:
            print("Loss {}".format(loss))
            print("Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    return np.mean(losses)

def test_epoch(model, device, test_loader, log_interval=20):
    return 0
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data = data.to(device)
            label = label.to(device)

            # Normalize input
            data = (data * 2**15).round() / 2**15
            ref = data.mean(0)
            data = (data - ref.mean()) / ref.std()

            # Run through the model
            valid_length = model.valid_length(data.shape[-1])
            delta = valid_length - data.shape[-1]
            padded = F.pad(data, (delta // 2, delta - delta // 2))

            output_signal, output_locs = model(padded)
            output_signal_trimmed = center_trim(output_signal, data)
            output_locs_trimmed = center_trim(output_locs, data)

            # Un-normalize
            output = output_signal_trimmed * ref.std() + ref.mean()

            loss = model.loss(output, label)
            test_loss += loss

            if batch_idx % log_interval == 0:
                print("Loss: {}".format(loss))

        test_loss /= len(test_loader)
        print("\nTest set: Average Loss: {:.4f}\n".format(test_loss))
        return test_loss

def train(args):
    data_train = SpatialAudioDatasetWaveform(args.train_dir)
    data_test = SpatialAudioDatasetWaveform(args.test_dir)

    use_cuda = args.use_cuda and torch.cuda.is_available()

    device = torch.device(args.device if use_cuda else 'cpu')

    num_workers = min(multiprocessing.cpu_count(), args.n_workers)

    kwargs = {'num_workers': num_workers,
              'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=args.batch_size,
                                               shuffle=args.shuffle,
                                               **kwargs)

    test_loader = torch.utils.data.DataLoader(data_test,
                                               batch_size=args.batch_size,
                                               **kwargs)

    model = Demucs(sources=1, n_audio_channels=args.n_channels)

    # Data parallel
    if args.multiple_GPUs:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    model.to(device)
    save_prefix = args.name
    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.decay)

    start_epoch = 0
    train_losses = []
    test_losses = []

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss = train_epoch(model, device, optimizer, train_loader, None, epoch, args.print_interval)
            test_loss = test_epoch(model, device, test_loader, args.print_interval)
            train_losses.append((epoch, train_loss))
            test_losses.append((epoch, test_loss))
            torch.save(model, os.path.join(args.checkpoints_dir, "{}_{}.pt".format(save_prefix, epoch)))
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
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--n_channels', type=int, default=2, help="Number of channels")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=3e-4, help="learning rate")
    parser.add_argument('--decay', type=float, default=0, help="Weight decay")
    parser.add_argument('--n_workers', type=int, default=16, help="Number of parallel workers")
    parser.add_argument('--print_interval', type=int, default=20, help="Logging interval")
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help="Path to the checkpoints")
    parser.add_argument('--shuffle', dest='shuffle', action='store_true')
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--multiple_GPUs', dest='multiple_GPUs', action='store_true')
    parser.add_argument('--device_ids', type=list, default=None, nargs='+')
    train(parser.parse_args())



# python train.py /projects/grail/audiovisual/datasets/DinTaiFung/mics_8_radius_3/train /projects/grail/audiovisual/datasets/DinTaiFung/mics_8_radius_3/test --name mics8_radius3_demucs --n_channels 8 --batch_size 4 --use_cuda --device cuda:2
