# -*- coding: utf-8 -*-
import numpy as np

def noise_resample_1dim(nt, period, **args):
    per = args['per'] if 'per' in args else 95
    th = np.percentile(nt, per)
    nt_shape = nt.shape
    nt = nt.reshape(-1, period)
    nt_normal = np.where(nt >= th, np.nan, nt)
    nt_normal = np.sort(nt_normal, axis=0)
    
    for i in range(period):
        column = nt_normal[:, i]
        nan_idxs = np.argwhere(column != column)
        if len(nan_idxs) != 0: 
            nan_idx = nan_idxs[0, 0]
            if nan_idx != 0:
                column[nan_idx: ] = np.random.choice(column[: nan_idx], len(nt_normal) - nan_idx)
            else:
                column[:] = np.random.choice(nt[:, i], len(nt_normal))
            
        np.random.shuffle(column)
    nt_normal = nt_normal.reshape(nt_shape)
    return nt_normal

def get_noise_normal_std(nt, period, **args):
    N = nt.shape[1]
    nt_normal = np.zeros(nt.shape)
    for i in range(N):
        nt_normal[:, i] = noise_resample_1dim(nt[:, i], period, **args)    
    return nt_normal