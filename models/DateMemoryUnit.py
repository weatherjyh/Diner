# -*- coding: utf-8 -*-
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def build_cycle_date_vector(total_len):
    date_matrix = np.zeros((total_len, 2))
    for idx in range(total_len):
        date_matrix[idx][0] = np.sin(2 * np.pi * idx / total_len)
        date_matrix[idx][1] = np.cos(2 * np.pi * idx / total_len)
    return date_matrix

class DateMemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(DateMemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres= shrink_thres
        
        self.mem_unit = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C, Memory Matrix
        self.date_map_unit = build_cycle_date_vector(self.mem_dim) # M x 2, Date Map
        self.date_map_unit = torch.tensor(self.date_map_unit).float()
        
        self.reset_parameters()

    def reset_parameters(self):
        self.stdv = 1. / math.sqrt(self.mem_unit.size(1))
        self.mem_unit.data.uniform_(-self.stdv, self.stdv)

    def forward(self, input_data, date_vector, verbose=False):
        # input_data TxC
        value_att_weight = F.linear(input_data, self.mem_unit)  # T x M 

        # T x M
        date_att_weight = F.linear(date_vector, self.date_map_unit)  # T x M
        date_att_weight = F.relu(date_att_weight)
        
        att_weight = torch.mul(value_att_weight, date_att_weight)
        att_weight = F.softmax(att_weight, dim=1)  
        
        if (self.shrink_thres > 0):
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
        att_weight = F.normalize(att_weight, p=1, dim=1)


        mem_trans = self.mem_unit.permute(1, 0)  # M x C
        output = F.linear(att_weight, mem_trans)  # T x C
        
        return {'output': output, 'att': att_weight}

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(
            self.mem_dim, self.fea_dim is not None
        )
    
def hard_shrink_relu(input_data, lambd=0, epsilon=1e-12):
    output = (F.relu(input_data-lambd) * input_data) / (torch.abs(input_data - lambd) + epsilon)
    return output