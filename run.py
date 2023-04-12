# -*- coding: utf-8 -*-
import os
import copy
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from evaluate import evaluate_1dim_prob_best

from models.SNR import SeasonalNoiseRepresentation
from models.DTMAE import DTMAE
from datasets.Dataset1d import reconstruct

from utils.load_data import load_dataset
from utils.bilateral_filter import denoise

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['font.size'] = '20'

def bi_filter(data, H=30, dn1=30, dn2=1):
    res = np.zeros(data.shape)
    for i in range(data.shape[1]):
        x = data[:, i]
        smoothed_x = denoise(x, H=H, dn1=dn1, dn2=dn2)
        res[:, i] = smoothed_x
    return res

class AnomalyDetection():
    def __init__(self, config, train_data, test_data, kpi_name):
        self.kpi_name = kpi_name
        self.device = 'cpu'
        self.config = config
        self.slide_win = config['slide_win']
        self.cycle_length = config['cycle_length'] 
        self.time_interval = config['time_interval'] 
        self.period = self.cycle_length // self.time_interval
        self.input_size = self.period
        self.seed = config['seed']

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)    
        
        self.train_data = train_data
        self.test_data = test_data
        
        self.results = {}
        self.results['metrics'] = {}
        
    def get_data(self):
        self.x = self.test_data['values']
        self.labels = self.test_data['labels']
        return self.x, self.labels     
    
    def run(self):
        x, labels = self.get_data()
        rec_data1, test_loss1 = self.run_dtmae(with_trend=True)
        scores1 = detector.get_anomaly_score1()
        nt = self.nt
        rec_data2, test_loss2 = self.run_snr(nt)
        scores2 = detector.get_anomaly_score2() 
        self.results['rec_data1'] = rec_data1
        self.results['rec_data2'] = rec_data2
        
        self.evaluate_score1()
        self.evaluate_score2()
        return x, labels, rec_data1, scores1, rec_data2, scores2, nt
    
    def run_dtmae(self, with_trend = False):
        sr_train_data = copy.deepcopy(self.train_data)
        sr_test_data = copy.deepcopy(self.test_data)
        self.sr_x = sr_test_data['values']
        
        sr_class = DTMAE(self.input_size, sr_train_data, sr_test_data, self.cycle_length, self.time_interval, with_trend, self.config)
        sr, sr_weight, sr_bias, sr_loss = sr_class.run()
        rec_idxs, t, x, labels = sr_class.get_data()
        sr = reconstruct(sr, rec_idxs)
        sr = sr.reshape(-1, 1)
        sr_weight = sr_weight.reshape(-1, 1)
        sr_bias = sr_bias.reshape(-1, 1)
        
        self.sr = sr
        self.sr_weight = sr_weight
        self.sr_bias = sr_bias
        return sr, sr_loss
        
    def run_snr(self, nt):
        snr_class = SeasonalNoiseRepresentation(nt, self.period, self.config)
        snr, snr_loss = snr_class.run()  
        self.snr = snr 
        return snr, snr_loss
    
    def get_anomaly_score1(self):
        scores = self.sr_x - self.sr
        scores = abs(scores)
        scaler = MinMaxScaler()
        scores = scaler.fit_transform(scores)  
        sr = bi_filter(self.sr, H=3, dn1=2, dn2=1) 
        self.nt = abs(self.sr_x - sr)
        self.results['scores1'] = scores
        return scores    
    
    def get_anomaly_score2(self):
        epsilon = 0.001
        scores = self.nt / (self.snr + epsilon)
        scores = abs(scores)        
        scaler = MinMaxScaler()
        scores = scaler.fit_transform(scores)
        self.results['scores2'] = scores
        return scores
    
    def evaluate_score1(self):
        x, labels = self.get_data()
        scores = self.results['scores1']
        pre, rec, f1, th = evaluate_1dim_prob_best(labels[period:, 0], scores[period:, 0], 'affiliation', **self.config)
        print('{} res1_DTMAE: precision: {}, recall: {}, f1-score: {}'.format(kpi_name, pre, rec, f1))
        self.results['metrics']['pre1'] = pre
        self.results['metrics']['rec1'] = rec
        self.results['metrics']['f1_1'] = f1
        self.results['metrics']['th1'] = th
        return pre, rec, f1, th
    
    def evaluate_score2(self):
        x, labels = self.get_data()
        scores = self.results['scores2']
        pre, rec, f1, th = evaluate_1dim_prob_best(labels[period:, 0], scores[period:, 0], 'affiliation', **self.config)
        print('{} res2_DTMAE_SNR: precision: {}, recall: {}, f1-score: {}'.format(kpi_name, pre, rec, f1))
        self.results['metrics']['pre2'] = pre
        self.results['metrics']['rec2'] = rec
        self.results['metrics']['f1_2'] = f1
        self.results['metrics']['th2'] = th
        return pre, rec, f1, th   
    
    def plot_results_DTMAE(self):
        scores = self.results['scores1']
        rec_data = self.results['rec_data1']
        pre = self.results['metrics']['pre1']
        rec = self.results['metrics']['rec1']
        f1 = self.results['metrics']['f1_1']
        th = self.results['metrics']['th1']
        
        period = self.period
        ths = np.ones(scores.shape) * th

        x, labels = self.get_data()
        anomalies_gt = np.full(x.shape, None)
        anomalies_gt[labels > 0.5] = x[labels > 0.5]

        anomalies_pred = np.full(x.shape, None)
        anomalies_pred[scores >= ths] = x[scores >= ths] 
        
        labels_pred = np.zeros(x.shape)
        labels_pred[scores >= ths] = 1
        
        m = period
        n = None
        plt.figure(figsize=(20, 20))
        plt.suptitle('{} res1_DTMAE: precision: {}, recall: {}, f1-score: {}'.format(kpi_name, pre, rec, f1))
        plt.subplot(5, 1, 1)
        plt.plot(x[m:n], 'g', label='original values')
        plt.plot(anomalies_gt[m:n], 'r', marker='o', alpha=0.6, label = 'ground truth')
        plt.plot(self.sr_bias[m:n], 'orange', alpha=0.6, label='additive trend')
        plt.legend()
        
        plt.subplot(5, 1, 2)
        plt.plot(self.sr_weight[m:n], 'orange', label='multiplicative trend')
        plt.ylim(0, 2)
        plt.legend()
        
        plt.subplot(5, 1, 3)
        plt.plot(x[m:n], 'g', alpha=0.5, label='original values')
        plt.plot(rec_data[m:n], 'b', label ='reconstructed values')
        
        
        plt.subplot(5, 1, 4)
        plt.plot(scores[m:n], 'orange', label = 'anomaly scores')
        plt.plot(ths[m:n], 'r', label='threshold')
        plt.legend()

        plt.subplot(5, 1, 5)
        plt.plot(ths[m:n], 'r', label='threshold')
        plt.plot(scores[m:n], 'orange', label = 'anomaly scores')
        plt.plot(labels_pred[m:n], 'b', label = 'predicted anomalies')
        plt.plot(-labels[m:n], 'g', label = 'ground truth')
        plt.legend()
        plt.savefig(f'./plts/{kpi_name}_res1.png')
        plt.tight_layout()
        # plt.show()
        plt.close()
    
    def plot_results_DTMAE_SNR(self):
        scores = self.results['scores2']
        rec_data = self.results['rec_data2']
        pre = self.results['metrics']['pre2']
        rec = self.results['metrics']['rec2']
        f1 = self.results['metrics']['f1_2']
        th = self.results['metrics']['th2']
        nt = self.nt
        
        period = self.period
        ths = np.ones(scores.shape) * th

        x, labels = self.get_data()
        anomalies_gt = np.full(x.shape, None)
        anomalies_gt[labels > 0.5] = nt[labels > 0.5]

        anomalies_pred = np.full(x.shape, None)
        anomalies_pred[scores >= ths] = nt[scores >= ths] 
        
        labels_pred = np.zeros(x.shape)
        labels_pred[scores >= ths] = 1
        
        m = period
        n = None
        plt.figure(figsize=(20, 16))
        plt.suptitle('{} res2_DTMAE_SNR: precision: {}, recall: {}, f1-score: {}'.format(kpi_name, pre, rec, f1))
        plt.subplot(4, 1, 1)
        plt.plot(nt[m:n], 'g', label='nt (resid of DTMAE)')
        plt.plot(anomalies_gt[m:n], 'r', marker='o', alpha=0.6, label = 'ground truth')
        plt.legend()
        
        plt.subplot(4, 1, 2)
        plt.plot(rec_data[m:n], 'b', label ='reconstructed nt')
        plt.legend()
        
        plt.subplot(4, 1, 3)
        plt.plot(scores[m:n], 'orange', label = 'adjusted anomaly scores')
        plt.plot(ths[m:n], 'r', label='threshold')
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.plot(ths[m:n], 'r', label='threshold')
        plt.plot(scores[m:n], 'orange', label = 'adjusted anomaly scores')
        plt.plot(labels_pred[m:n], 'b', label = 'predicted anomalies')
        plt.plot(-labels[m:n], 'g', label = 'ground truth')
        plt.legend()
    
        plt.tight_layout()
        plt.savefig(f'./plts/{kpi_name}_res2.png')
        # plt.show()  
        plt.close()
             
# In[]  
    
if __name__ == "__main__":                                                                                                                                                                                   
    config={
        'cycle_length': 86400000,
        'time_interval': 300000,
        'slide_win': 288,
        'batch': 128,
        'epoch': 20,
        'seed': 100,
        'decay': 0,
        'val_ratio': 0.2,
        "per": 95,
        'bound_mode': 18,
        'max_detect_delay': 18,
        'DTMAE_kernel_size': 5,
        'SNR_kernel_size': (3, 9),
        'VarPool_kernel_size': (3, 9),
    }
    
    period = 288
    for dataset in ['GAIA_periodic_data', 'AIOps2018_seasonal', 'NAB_realTweets']:
    # for dataset in ['AIOps2018_seasonal']:
    
        feature_map, train_data_total, test_data_total = load_dataset(dataset)
        scores1_total = []
        scores2_total = []
        labels_total = []
        
        for kpi_name in feature_map: 
            train_data = train_data_total[kpi_name]
            test_data = test_data_total[kpi_name]
        
            detector = AnomalyDetection(config, train_data, test_data, kpi_name)
            x, labels, rec_data1, scores1, rec_data2, scores2, resid = detector.run()

            # detector.plot_results_DTMAE()
            # detector.plot_results_DTMAE_SNR()
            
            # values = x
            # scores_df = {'values': values.reshape(-1), 'labels': labels.reshape(-1), 'scores': scores1.reshape(-1), 'rec_data': rec_data1.reshape(-1)}
            # scores_df = pd.DataFrame(scores_df)
            # scores_df.to_csv(os.path.join('scores', f'{kpi_name}_scores1.csv'))        
    
            # scores_df = {'values': values.reshape(-1), 'labels': labels.reshape(-1), 'scores': scores2.reshape(-1), 'rec_data': rec_data2.reshape(-1)}
            # scores_df = pd.DataFrame(scores_df)
            # scores_df.to_csv(os.path.join('scores', f'{kpi_name}_scores2.csv'))  
