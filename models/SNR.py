# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from datasets.Dataset1d import get_loaders
from datasets.Dataset2d import ImageDataset
from utils.seasonal_noise_std import get_noise_normal_std

def get_cycle_date_code(idx, total_len):
    code = [0, 0]
    code[0] = np.sin(2 * np.pi * idx / total_len)
    code[1] = np.cos(2 * np.pi * idx / total_len)
    return code
    
class SeasonalNoiseRepresentation():
    def __init__(self, nt, period, config={}):
        self.period = period
        self.config = config
        self.device = config['device'] if "device" in config else "cpu"
        self.nt = nt
        return
    
    def run(self):
        self.nt_std = get_noise_normal_std(self.nt, self.period, **self.config)
        self.build_dataset()

        self.model = ConvAE2d(self.period, kernel_size=self.config['SNR_kernel_size'])
        self.train()
        test_loss, test_result = self.test()
        
        rec_data = test_result[:, :, 0, :]
        rec_data = rec_data.transpose(1, 0, 2)
        rec_data_tail = test_result[-1, :, 1:, :]
        
        rec_data = np.concatenate((rec_data, rec_data_tail), axis=1)
        rec_data = rec_data.reshape(rec_data.shape[0], -1).T
        return rec_data, test_loss    
    
    def build_dataset(self, train_len=2880):
        raw_data = self.nt_std
        min_max_scaler = MinMaxScaler()
        raw_data = min_max_scaler.fit_transform(raw_data)
        
        train_data = raw_data
        test_data = raw_data
        self.train_dataset = ImageDataset(train_data, self.period, kernel_size=self.config['VarPool_kernel_size'])
        self.test_dataset = ImageDataset(test_data, self.period, kernel_size=self.config['VarPool_kernel_size'])
        self.train_dataloader, self.val_dataloader = get_loaders(self.train_dataset, self.config['seed'], self.config['batch'], val_ratio = self.config['val_ratio'])  
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.config['batch'], shuffle=False, num_workers=0)
        return self.train_dataloader, self.val_dataloader, self.test_dataloader

    
    def train(self, verbose=False):
        lr = self.config['lr'] if "lr" in self.config else 1e-3
        epoch = self.config['epoch'] if "epoch" in self.config else 5
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.config['decay'] if 'decay' in self.config else 0)
        train_loss_list = []
        for i_epoch in range(epoch):
            acu_loss = 0
            for x in self.train_dataloader:
                x = x.float().to(self.device)
                predicted = self.model(x)
                loss = F.mse_loss(predicted, x, reduction='sum')
                loss = loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acu_loss += loss.item()
                train_loss_list.append(loss.item())
   
            if verbose:
                print('ConvAE2d: epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
                            i_epoch+1, epoch, 
                            acu_loss/len(self.train_dataloader), acu_loss), flush=True
                )             
        return train_loss_list   

    def test(self):
        t_test_predicted_list = []
        
        test_loss_list = []
        for x in self.test_dataloader:
            x = x.float().to(self.device)
            x_shape = x.shape
            with torch.no_grad():
                predicted = self.model(x)
                loss = F.mse_loss(predicted, x, reduction='sum')

            predicted = predicted.reshape(x_shape)     
            x = x.reshape(x_shape)
            
            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                
            test_loss_list.append(loss.item())  
        avg_loss = sum(test_loss_list)/len(test_loss_list)   
        
        test_predicted_list = t_test_predicted_list    
        return avg_loss, np.array(test_predicted_list)      
       
class ConvAE2d(nn.Module):
    def __init__(self, input_size, 
                 h_dim=400, 
                 z_dim=60, 
                 mem_dim=50, 
                 shrink_thres=0.0025,
                 kernel_size=(3, 9)):
        super(ConvAE2d, self).__init__()
        dim1 = kernel_size[0]
        dim2 = kernel_size[1]
        self.kernel_size = kernel_size
        self.conv_padding = (dim1 - 1), (dim2 - 1)
        
        self.input_size = input_size
        
        self.mu_conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size = self.kernel_size, stride=1, padding = self.conv_padding, padding_mode='circular')
        self.mu_conv2.weight.data = torch.ones((1, 1, self.kernel_size[0], self.kernel_size[1])) / (self.kernel_size[0] * self.kernel_size[1])
        self.mu_conv2.bias.data = torch.zeros((1))
        
        self.mu_convtrans2 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size = self.kernel_size, stride=1, padding = self.conv_padding)
        self.mu_convtrans2.weight.data = torch.ones((1, 1, self.kernel_size[0], self.kernel_size[1])) / (self.kernel_size[0] * self.kernel_size[1])
        self.mu_convtrans2.bias.data = torch.zeros((1))

        self.relu = nn.ReLU()
        
    def forward(self, x):
        self.mu_conv2.bias.data = torch.zeros((1))
        self.mu_convtrans2.bias.data = torch.zeros((1))
        x = self.mu_conv2(x)
        x = self.mu_convtrans2(x)    
        out = self.relu(x)
        return out 
