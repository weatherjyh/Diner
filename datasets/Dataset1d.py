import random

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

from torch.utils.data import Subset


WIN_LEN = 20

def get_loaders(train_dataset, seed, batch, val_ratio=0.1, repeat=True, shuffle=True):
    dataset_len = int(len(train_dataset))
    train_use_len = int(dataset_len * (1 - val_ratio))
    val_use_len = int(dataset_len * val_ratio)
    val_start_index = random.randrange(train_use_len)
    indices = torch.arange(dataset_len)

    train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
    train_subset = Subset(train_dataset, train_sub_indices)

    val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
    val_subset = Subset(train_dataset, val_sub_indices)

    if repeat:
        train_dataloader = DataLoader(train_dataset, batch_size=batch,
                                shuffle=shuffle)
    else:
        train_dataloader = DataLoader(train_subset, batch_size=batch,
                                shuffle=shuffle)

    val_dataloader = DataLoader(val_subset, batch_size=batch,
                            shuffle=False)
    return train_dataloader, val_dataloader

    
def get_cycle_date_code(idx, total_len):
    code = [0, 0]
    code[0] = np.sin(2 * np.pi * idx / total_len)
    code[1] = np.cos(2 * np.pi * idx / total_len)
    return code

def build_cycle_date_vector(total_len):
    date_matrix = np.zeros((total_len, 2))
    for idx in range(total_len):
        date_matrix[idx][0] = np.sin(2 * np.pi * idx / total_len)
        date_matrix[idx][1] = np.cos(2 * np.pi * idx / total_len)
    return date_matrix
 
class DateDataset(Dataset):
    def __init__(self, dataset, cycle_length, time_interval):
        self.cycle_length = cycle_length
        self.time_interval = time_interval
        self.period = self.cycle_length // self.time_interval
        self.raw_values = dataset['values']
        self.raw_labels = dataset['labels']
        self.timestamps = dataset['timestamps']
        self.rec_idxs, self.t, self.x, self.labels = self.process(self.raw_values, self.raw_labels, self.timestamps)
        
        
    def process(self, raw_values, raw_labels, timestamps):
        raw_values = np.array(raw_values).T
        raw_labels = np.array(raw_labels).T

        rec_idxs = []
        t_arr, x_arr = [], []
        labels_arr = []
        for i in range(self.period, raw_values.shape[1] + 1):
            WEEK_LEN = 7
            week_idx = timestamps[i - self.period] // self.time_interval % (self.period * WEEK_LEN)
            day_idx = timestamps[i - self.period] // self.time_interval % (self.period)
            
            rec_idx = (self.period - 1 + day_idx - WIN_LEN) % self.period
            rec_idxs.append(rec_idx)
            
            win_data = raw_values[:, i - self.period: i]
            base = np.concatenate((win_data, win_data), axis=1)
            win_data = torch.tensor(win_data).double()

            cycle_win_data = base[:, self.period - day_idx: 2 * self.period - day_idx]
            cycle_win_data = torch.tensor(cycle_win_data).double()

            win_label = raw_labels[:, i - self.period: i]
            label_base = np.concatenate((win_label, win_label), axis=1)
            cycle_win_label = label_base[:, self.period-day_idx: 2 * self.period-day_idx]
            win_label = torch.tensor(win_label).double()
            cycle_win_label = torch.tensor(cycle_win_label).double()
            
            cycle_date_code = get_cycle_date_code(week_idx, self.period * WEEK_LEN)
            
            date_para = np.array([cycle_date_code])
            
            date_vector = date_para.reshape(-1)
            date_vector = torch.tensor(date_vector).float()
            
            cycle_mean = np.mean(cycle_win_data.detach().numpy(), axis=1).reshape(-1, 1)
            cycle_mean = torch.tensor(cycle_mean).float()
            
            t_arr.append(date_vector)
            x_arr.append(cycle_win_data)
            labels_arr.append(cycle_win_label)
            
        rec_idxs = torch.tensor(rec_idxs).int()
        t = torch.stack(t_arr).contiguous()
        x = torch.stack(x_arr).contiguous()
        labels = torch.stack(labels_arr).contiguous()
        return rec_idxs, t, x, labels
    
    def __len__(self):
        return len(self.x)

    def get_data(self):
        return self.rec_idxs, self.t, self.x, self.labels

    def __getitem__(self, idx):
        t = self.t[idx].float()
        rec_idx = self.rec_idxs[idx].int()
        x = self.x[idx].double()
        label = self.labels[idx].double()
        return rec_idx, t, x, label

def reconstruct(data, rec_idxs, period=288, full_dim=False):
    rec_idxs = [(idx + WIN_LEN) % period for idx in rec_idxs]
    
    if full_dim:
        rec_data = np.zeros((data.shape[0], data.shape[1], 1))
        for i in range(data.shape[0]):
            rec_data[i, :, 0] = data[i, :, rec_idxs[i]]
    else:
        rec_data = np.zeros((data.shape[0], data.shape[1]))
        for i in range(data.shape[0]):
            rec_data[i, :] = data[i, :, rec_idxs[i]]
    
    period = data.shape[2]
    head = data[0, :, :]
    base = np.concatenate((head, head), axis=1)
    idx = rec_idxs[0]
    cycle_head = base[:, idx + 1: idx + period]
    cycle_head = cycle_head.transpose(1, 0)
    rec_data = np.concatenate((cycle_head, rec_data), axis=0)
    return rec_data

