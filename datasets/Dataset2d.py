import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
 
class ImageDataset(Dataset):
    def __init__(self, raw_values, period, height=7, kernel_size=(3, 9)):
        self.period = period
        self.height = height
        self.x = self.process(raw_values)
           
    def process(self, raw_values):
        raw_values = raw_values.reshape(-1)
        raw_values = raw_values.reshape(-1, self.period)
        image_data = []
        for i in range(raw_values.shape[0] - self.height + 1):
            nt2d_win = raw_values[i: i + self.height, :]
            nt2d_win = torch.tensor(nt2d_win)
            var_pool = VarPool2d()
            var2d_win = var_pool.pooling(nt2d_win)
            std2d_win = np.sqrt(var2d_win)
            std2d_win = np.array(std2d_win) 
            
            image_data.append(std2d_win)
        image_data = np.array(image_data)
        x = torch.tensor(image_data).double()
        x = x.unsqueeze(1)
        return x
        
    def __len__(self):
        return len(self.x)

    def get_data(self):
        return self.x

    def __getitem__(self, idx):
        x = self.x[idx].double()
        return x
    
class VarPool2d():
    def __init__(self, kernel_size=(3, 9)):
        dim1 = kernel_size[0]
        dim2 = kernel_size[1]
        
        if dim1 % 2 == 0:
            dim1 += 1        
        if dim2 % 2 == 0:
            dim2 += 1
        
        self.kernel_size = (dim1, dim2)
        self.pool_padding = ((dim1 - 1) // 2, (dim2 - 1) // 2)
        self.avg_pool = nn.AvgPool2d(kernel_size = self.kernel_size, stride=1, padding=self.pool_padding)
        
    def pooling(self, nt2d):
        var2d = torch.pow(nt2d, 2)
        var2d = var2d.unsqueeze(0)
        var2d_pool = self.avg_pool(var2d)
        var2d_pool = var2d_pool.squeeze(0)
        return var2d_pool
