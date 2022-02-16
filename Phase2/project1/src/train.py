import argparse
import os
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import shutil

from torch.utils.data import DataLoader
from dataio import HomographyDataset
from network.HomographyNet import HomographyNet


def count_params(model):
    params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params, train_params


def l2_loss(out, gt):
    return torch.mean((out - gt) ** 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bp', '--base_path', default='../data/Patches', help='Base path for images. Default: \'../data/Patches\'')
    parser.add_argument('-cb', '--checkpoint_base', default='../checkpoints', help='Base path to save checkpoints. Default: \'../checkpoints\'')
    parser.add_argument('-e', '--epochs', type=int, default=200, help='Number of epochs to train200Default: 200')
    parser.add_argument('-etc', '--epochs_till_chkpt', type=int, default=10, help='Checkpoint is saved after these many epochs. Default: 10')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Size of batch. Default: 32')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.005, help='Starting learning rate. Default: 0.005')
    parser.add_argument('-w', '--workers', type=int, default=0, help='Number of workers. Default: 0')
    parser.add_argument('-n', '--name', required=True, help='Name of the experiment. Used to save checkpoints.')
    args = parser.parse_args()
    base_path = args.base_path
    checkpoint_base = args.checkpoint_base
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    workers = args.workers
    name = args.name
    epochs_till_chkpt = args.epochs_till_chkpt

    train_base_path = os.path.join(base_path, 'Train')
    val_base_path = os.path.join(base_path, 'Val')
    checkpoint_path = os.path.join(checkpoint_base, name)

    train_dataset = HomographyDataset(base_path=train_base_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_dataset = HomographyDataset(base_path=val_base_path)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    model = HomographyNet()
    model.train()
    # loss_fn = l2_loss
    loss_fn = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.1)

    if os.path.isdir(checkpoint_path):
        shutil.rmtree(checkpoint_path)
    os.makedirs(checkpoint_path)

    params, train_params = count_params(model)
    print('===========================================================')
    print('Starting training experiment: ' + name)
    print('Number of model parameters: ' + str(params))
    print('Number of trainable model parameters: ' + str(train_params))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print('CUDA enabled GPU found!')
        model = model.to(device)
    else:
        print('CUDA enabled GPU not found! Using CPU.')
    print('===========================================================')

    epoch_losses = []
    val_losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for data in train_dataloader:
            inp, gt = data[0].to(device), data[1].to(device)
            # print(inp)
            # print(gt)
            optimizer.zero_grad()
            out = model(inp)
            # print(out)
            loss = loss_fn(out, gt)
            # print(loss)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # print(epoch_loss)
        
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            for data in val_dataloader:
                inp, gt = data[0].to(device), data[1].to(device)
                out = model(inp)
                loss = loss_fn(out, gt)
                val_loss += loss.item()
            model.train()

        scheduler.step()
        epoch_losses.append(epoch_loss)
        print(f'Epoch {epoch + 1} (finished) loss: {epoch_loss}')

        val_losses.append(val_loss)
        print(f'Epoch {epoch + 1} (finished) Validation loss: {val_loss}')

        if epoch % epochs_till_chkpt == 0 and epoch != epochs - 1:
            torch.save(model.state_dict(), os.path.join(checkpoint_path, f'model_{epoch}.pth'))
        elif epoch == epochs - 1:
            torch.save(model.state_dict(), os.path.join(checkpoint_path, 'model_final.pth'))


if __name__ == '__main__':
    main()

