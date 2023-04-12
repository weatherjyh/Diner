# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.Dataset1d import DateDataset, get_loaders
from models.DateMemoryUnit import DateMemoryUnit

class DTMAE():
    def __init__(self, input_size, train_data, test_data, cycle_length, time_interval, with_trend, config={}):
        self.cycle_length = cycle_length
        self.time_interval = time_interval
        self.period = cycle_length // time_interval
        
        self.config = config
        self.device = config['device'] if "device" in config else "cpu"
        train_dataset = DateDataset(train_data, self.cycle_length, self.time_interval)
        test_dataset = DateDataset(test_data, self.cycle_length, self.time_interval)
        
        self.test_dataset = test_dataset
        
        self.train_dataloader, self.val_dataloader = get_loaders(train_dataset, self.config['seed'], self.config['batch'], val_ratio = self.config['val_ratio'])  
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.config['batch'], shuffle=False, num_workers=0)        
        
        self.model = MemConvAEGroup(input_size, 1, with_trend = with_trend, kernel_size=config['DTMAE_kernel_size'])
    
    def run(self):
        self.train()
        sr, sr_weight, sr_bias, sr_loss= self.test()
        return sr, sr_weight, sr_bias, sr_loss
    
    def get_data(self):
        return self.test_dataset.get_data() 
    

    def train(self, verbose=False):
        lr = self.config['lr'] if "lr" in self.config else 1e-3
        epoch = self.config['epoch'] if "epoch" in self.config else 5
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.config['decay'] if 'decay' in self.config else 0)
        train_loss_list = []
        for i_epoch in range(epoch):
            acu_loss = 0
            for rec_idx, t, x,labels in self.train_dataloader:

                x = x.float().to(self.device)

                predicted, weight, bias = self.model(x, t)
                loss1 = F.mse_loss(predicted,x, reduction='sum')
                loss = loss1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acu_loss += loss.item()
                train_loss_list.append(loss.item())   
            if verbose:
                print('VAE: epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
                            i_epoch+1, epoch, 
                            acu_loss/len(self.train_dataloader), acu_loss), flush=True
                )             
        return train_loss_list   
     
    def test(self):
        test_predicted_list = []
        test_weight_list = []
        test_bias_list = []
        test_loss_list = []
        for rec_idx, t, x, labels in self.test_dataloader:
            x = x.float().to(self.device)
            with torch.no_grad():
                predicted, weight, bias = self.model(x, t)
                reconst_loss = F.mse_loss(predicted, x, reduction='sum')
                loss = reconst_loss

            if len(test_predicted_list) <= 0:
                test_predicted_list = predicted
                test_weight_list = weight
                test_bias_list = bias
            else:
                test_predicted_list = torch.cat((test_predicted_list, predicted), dim=0)   
                test_weight_list = torch.cat((test_weight_list, weight), dim=0)  
                test_bias_list = torch.cat((test_bias_list, bias), dim=0)  
                
            test_loss_list.append(loss.item())

        avg_loss = sum(test_loss_list)/len(test_loss_list)     
        return np.array(test_predicted_list), np.array(test_weight_list), np.array(test_bias_list), avg_loss

class MemConvAE(nn.Module):
    def __init__(self, input_size, 
                 h_dim=400, 
                 z_dim=60, 
                 mem_dim=50, 
                 shrink_thres=0.0025, 
                 with_trend=True,
                 kernel_size=5):
        super(MemConvAE, self).__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1
        
        self.mu_conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size = self.kernel_size, padding=self.padding, padding_mode='circular')
        self.mu_convtrans1 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size = self.kernel_size, padding=self.padding)

        zero_padding = self.kernel_size // 2
        self.mem_fea_dim = input_size  + (self.padding - zero_padding) * 2
        self.mem_rep = DateMemoryUnit(mem_dim=mem_dim, fea_dim=self.mem_fea_dim , shrink_thres=shrink_thres)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.fc_w = nn.Linear(input_size, 1)
        self.fc_b = nn.Linear(input_size, 1)
        self.with_trend = with_trend
        
    def forward(self, x, date_vector, verbose=False):
        self.mu_conv1.bias.data = torch.zeros((1))
        self.mu_convtrans1.bias.data = torch.zeros((1))
        
        st = x
        st = st.unsqueeze(1)
        st = self.mu_conv1(st)
        st_mem = st.view(-1, self.mem_fea_dim)
        res_mem = self.mem_rep(st_mem, date_vector)
        st_mem = res_mem['output']
        st = st_mem.view(st.shape)
        st = self.mu_convtrans1(st)
        st = st.squeeze(1) 
        
            
        weight = self.fc_w(x)    
        bias = self.fc_b(x)  

        if self.with_trend:
            max_trend_factor = 0.5 # 0.5
            
            weight = self.tanh(weight)
            weight = torch.mul(weight, max_trend_factor)
            weight = torch.add(weight, 1)
            st = torch.mul(st, weight)

            bias = self.fc_b(x)
            bias = self.tanh(bias)
            bias = torch.mul(bias, max_trend_factor)
            st = torch.add(st, bias)
        return st, weight, bias

class MemConvAEGroup(nn.Module):
    def __init__(self, input_size, 
                  feature_size,
                  h_dim=400, 
                  z_dim=60, 
                  mem_dim=50, 
                  shrink_thres=0.0025,
                  with_trend=True,
                  kernel_size=5):
        super(MemConvAEGroup, self).__init__()
        self.input_size = input_size
        self.model_group = []
        self.feature_size = feature_size
        for i in range(self.feature_size):
            model = MemConvAE(input_size, h_dim, z_dim, mem_dim, shrink_thres, with_trend, kernel_size=kernel_size)
            self.model_group.append(model)
        self.model_group = nn.ModuleList(self.model_group)
        
    def forward(self, x, date_vector, verbose=False):
        output = []
        weight = []
        bias = []
        for i in range(self.feature_size):
            model_output, model_weight, model_bias = self.model_group[i](x[:, i, :], date_vector, verbose)
            model_output = model_output.unsqueeze(1)
            model_weight = model_weight.unsqueeze(1)
            model_bias = model_bias.unsqueeze(1)
            output.append( model_output )
            weight.append( model_weight )
            bias.append( model_bias )
            
        output = torch.cat(output, dim=1)
        weight = torch.cat(weight, dim=1)
        bias = torch.cat(bias, dim=1)
        return output, weight, bias