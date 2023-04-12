# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt

def denoise(sample, H=3, dn1=1., dn2=1.):
    def get_denoise_value(idx):
        start_idx, end_idx = get_neighbor_idx(len(sample), idx, H)
        idxs = np.arange(start_idx, end_idx)
        weight_sample = sample[idxs]

        weights = np.array(list(map(lambda j: bilateral_filter(j, idx, sample[j], sample[idx], dn1, dn2), idxs)))
        return np.sum(weight_sample * weights)/np.sum(weights)

    idx_list = np.arange(len(sample))
    denoise_sample = np.array(list(map(get_denoise_value, idx_list)))
    return denoise_sample

def get_neighbor_idx(total_len, target_idx, H=3):
    '''
    Let i = target_idx.
    Then, return i-H, ..., i, ..., i+H, (i+H+1)
    '''
    return [np.max([0, target_idx-H]), np.min([total_len, target_idx+H+1])]

def bilateral_filter(j, t, y_j, y_t, delta1=1.0, delta2=1.0):
    idx1 = -1.0 * (math.fabs(j-t) **2.0)/(2.0*delta1**2)
    idx2 = -1.0 * (math.fabs(y_j-y_t) **2.0)/(2.0*delta2**2)
    weight = (math.exp(idx1)*math.exp(idx2))
    return weight


if __name__ == "__main__":
    def gen_wave():
        N = 100
        t = np.linspace(0,10,N)
        y = np.sign(np.sin(t))
        y += np.random.random((N))*0.5
        return y
    ts = gen_wave()
    
    # plt.figure(figsize=(10,5))
    # plt.plot(ts, label='original')
    
    # def bilateral_filter(j, t, y_j, y_t, delta1=1.0, delta2=1.0):
    #     idx1 = -1.0 * (math.fabs(j-t) **2.0)/(2.0*delta1**2)
    #     idx2 = 0  												# gaussian filter
    #     weight = (math.exp(idx1)*math.exp(idx2))
    #     return weight
    
    # ts1 = denoise(ts, H=5, dn1=10, dn2=1)
    # plt.plot(ts1, label='gaussian')
    

    plt.figure(figsize=(10,5))
    plt.plot(ts, label='original')    
    ts1 = denoise(ts, H=5, dn1=10, dn2=1)
    plt.plot(ts1, label='bilateral')
    plt.legend()
    plt.show()