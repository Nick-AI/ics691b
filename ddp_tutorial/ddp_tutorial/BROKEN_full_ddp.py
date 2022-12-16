import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from typing import Optional

from torch.nn.parallel import DistributedDataParallel as DDP

from .model import FFNet
from .dataset import ToyDataset


def ddp_setup(rank: int,
              world_size: int,
              shared_file: Optional[str] = '/tmp/sharedfile'):
    """Initilize the process group.

    Args:
        rank (int): rank of the process within the process group
        world_size (int): number of processes within the process group
        shared_file (str, optional): Shared file for IPC. Defaults to '/tmp/sharedfile'.
    """
    dist.init_process_group(
        init_method=f'file://{shared_file}',
        rank=rank,
        world_size=world_size)


def ddp_cleanup(shared_file: Optional[str] = '/tmp/sharedfile'):
    """Run after training is completed by rank 0 process.

    Args:
        shared_file (str, optional): Shared file for IPC. Defaults to '/tmp/sharedfile'.
    """
    dist.destroy_process_group()
    try:
        os.remove(shared_file)
    except FileNotFoundError:
        pass


def run_demo(rank: int,
             world_size: int,
             shared_file: Optional[str] = '/tmp/sharedfile'):
    """Script executed by each process in the DDP process group.

    Script for neural network training.
    Notice how it's very similar to the basic example with some coordination with other processes mixed in. 

    Args:
        rank (int): rank of the process within the process group
        world_size (int): number of processes within the process group
        shared_file (str, optional): Shared file for IPC. Defaults to '/tmp/sharedfile'.
    """
    print(f'Training process started on rank {rank}')
    ddp_setup(rank, world_size, shared_file=shared_file)

    # Data parameters
    batch_size = 8
    input_size = 100
    num_samples = 1_024

    train_ds = ToyDataset(input_size=input_size, num_samples=num_samples)
    # Using the torch DataLoader is good practice as it allows easy shuffling and scaling to parallel data loading by increasing the number of workers
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, num_workers=0, shuffle=False)

    # Model parameters
    num_hl = 0
    hl_size = 10

    nn_mdl = FFNet(input_size=input_size, num_hl=num_hl, hl_size=hl_size)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(nn_mdl.parameters(), lr=0.001, momentum=0.9)

    # how many times we want to loop over the training data
    nb_epochs = 10

    for epoch in range(nb_epochs):
        running_loss = 0.
        for batch_idx, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            samples, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = nn_mdl(samples)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        # we expect loss to remain pretty consistent since data and labels are random
        print(f'Epoch: {epoch + 1} - total loss: {running_loss / 2000:.3f}')

    print('Finished Training')
