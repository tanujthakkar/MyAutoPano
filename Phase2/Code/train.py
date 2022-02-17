import argparse
import os
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import shutil
import numpy as np
from tqdm import tqdm
import cv2

from torch.utils.data import DataLoader
from dataio import HomographyDataset
from Network.HomographyNet import HomographyNet
from GenerateData import generateResult
from Helper import readImageSet, preprocess_H4_data, deprocess_H4_data

def count_params(model):
    params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params, train_params


def l2_loss(out, gt):
    return torch.mean((out - gt) ** 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bp', '--base_path', default='../Data/Patches', help='Base path for images. Default: \'../Data/Patches\'')
    parser.add_argument('-cb', '--checkpoint_base', default='../checkpoints', help='Base path to save checkpoints. Default: \'../checkpoints\'')
    parser.add_argument('--dataset_path', default='../Data/', help='Path to the full training dataset')
    parser.add_argument('-e', '--epochs', type=int, default=200, help='Number of epochs to train Default: 200')
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
    results_path = os.path.join(checkpoint_base, name, 'Results')

    train_dataset = HomographyDataset(base_path=train_base_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_dataset = HomographyDataset(base_path=val_base_path)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    training_data = readImageSet(os.path.join(args.dataset_path,'Train'))
    validation_data = readImageSet(os.path.join(args.dataset_path,'Val'))
    H_AB_training_data = np.load(os.path.join(train_base_path,'H_AB.npy'))
    H_AB_validation_data = np.load(os.path.join(val_base_path, 'H_AB.npy'))

    model = HomographyNet()
    model.train()
    # loss_fn = l2_loss
    loss_fn = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 38, 0.1)

    if os.path.isdir(checkpoint_path):
        shutil.rmtree(checkpoint_path)
    os.makedirs(checkpoint_path)
    if os.path.isdir(os.path.join(results_path, 'Train')):
        shutil.rmtree(os.path.join(results_path, 'Train'))
    os.makedirs(os.path.join(results_path, 'Train'))
    if os.path.isdir(os.path.join(results_path, 'Val')):
        shutil.rmtree(os.path.join(results_path, 'Val'))
    os.makedirs(os.path.join(results_path, 'Val'))

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
    iters_train = len(train_dataset) / batch_size
    iters_val = len(val_dataset) / batch_size
    for epoch in range(epochs):
        epoch_loss = 0.0
        for data in tqdm(train_dataloader):
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

        result = generateResult(training_data[0], deprocess_H4_data(out[0].cpu().detach().numpy()), H_AB_training_data[0], [240, 320], [128,128], 32, False)
        cv2.imwrite(os.path.join(results_path, 'Train', '%d.png'%epoch), result)
        # input('q')
        
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            for data in tqdm(val_dataloader):
                inp, gt = data[0].to(device), data[1].to(device)
                out = model(inp)
                loss = loss_fn(out, gt)
                val_loss += loss.item()

            result = generateResult(validation_data[0], deprocess_H4_data(out[0].cpu().detach().numpy()), H_AB_validation_data[0], [240, 320], [128,128], 32, False)
            cv2.imwrite(os.path.join(results_path, 'Val', '%d.png'%epoch), result)

            model.train()

        scheduler.step()
        epoch_losses.append(epoch_loss / iters_train)
        print(f'Epoch {epoch + 1} (finished) loss: {epoch_loss / iters_train}')

        val_losses.append(val_loss / iters_val)
        print(f'Epoch {epoch + 1} (finished) Validation loss: {val_loss / iters_val}')

        if epoch % epochs_till_chkpt == 0 and epoch != epochs - 1:
            torch.save(model.state_dict(), os.path.join(checkpoint_path, f'model_{epoch}.pth'))
        elif epoch == epochs - 1:
            torch.save(model.state_dict(), os.path.join(checkpoint_path, 'model_final.pth'))


if __name__ == '__main__':
    main()

