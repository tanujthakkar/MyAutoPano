import os
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from skimage.io import imread


class HomographyDataset(Dataset):
    
    def __init__(self, base_path):
        super().__init__()
        self.base_path = base_path
        self.patchA_path = os.path.join(self.base_path, 'PatchA')
        self.patchB_path = os.path.join(self.base_path, 'PatchB')
        self.gt_path = os.path.join(self.base_path, 'H4.npy')

        if not (os.path.isdir(self.patchA_path) and os.path.isdir(self.patchB_path)):
            raise ValueError('Supply a valid path to initialize the dataset!')

        self.patchA_files = [os.path.join(self.patchA_path, file) for file in os.listdir(self.patchA_path)]
        self.patchB_files = [os.path.join(self.patchB_path, file) for file in os.listdir(self.patchB_path)]
        self.patchA = np.array([imread(file) for file in self.patchA_files])
        self.patchB = np.array([imread(file) for file in self.patchB_files])
        self.gt = np.load(self.gt_path)

        if (self.patchA.shape[0] != self.patchB.shape[0]) and (self.gt.shape[0] != self.patchA.shape[0]):
            raise ValueError('There should be equal number of patches in base path!')
    
    def __len__(self):
        return self.patchA.shape[0]
    

    def __remap(self, x, oMin, oMax, iMin, iMax):
        result = np.add(np.divide(np.multiply(x - iMin, oMax - oMin), iMax - iMin), oMin)
        return result


    def __preprocess_h4_data(self, h4, rho=32):
        h4 = self.__remap(h4, -1.0, 1.0, -rho, rho)
        return h4.reshape(8)


    def deprocess_h4_data(self, h4, rho=32):
        h4 = self.__remap(h4, -rho, rho, -1.0, 1.0)
        return np.int32(h4.reshape(4,2))

    
    def __getitem__(self, index):
        patchA_img = (self.patchA[index] - 127.5) / 127.5
        patchB_img = (self.patchB[index] - 127.5) / 127.5
        inp = np.stack((patchA_img, patchB_img))
        gt = self.gt[index]
        gt = self.__preprocess_h4_data(gt)
        inp_tensor = torch.from_numpy(inp).float()
        gt_tensor = torch.from_numpy(gt).float()
        return inp_tensor, gt_tensor
        

if __name__ == '__main__':
    base_path = '../data/Patches/Train'
    dataset = HomographyDataset(base_path=base_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    dataiter = iter(dataloader)
    inp, gt = dataiter.next()
    print(inp.shape)
    print(gt.shape)

