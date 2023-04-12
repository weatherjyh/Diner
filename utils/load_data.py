# -*- coding: utf-8 -*-
import pandas as pd
def get_feature_list(dataset):
    feature_file = open(f'./data/{dataset}/list.txt', 'r', encoding='utf-8')
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())
    return feature_list

def load_dataset(dataset):
    feature_map = get_feature_list(dataset)
    train_data = {}
    test_data = {}
    for kpi_name in feature_map:
        kpi_train_data = pd.read_csv(f'./data/{dataset}/{kpi_name}_train.csv', sep=',')
         
        kpi_train_data = {
             'timestamps': kpi_train_data['timestamps'].values,
             'values': kpi_train_data['values'].values.reshape(-1, 1),
             'labels': kpi_train_data['labels'].values.reshape(-1, 1),
             }
         
        train_data[kpi_name] = kpi_train_data
        kpi_test_data = pd.read_csv(f'./data/{dataset}/{kpi_name}_test.csv', sep=',')
        kpi_test_data = {
             'timestamps': kpi_test_data['timestamps'].values,
             'values': kpi_test_data['values'].values.reshape(-1, 1),
             'labels': kpi_test_data['labels'].values.reshape(-1, 1),
             } 
        test_data[kpi_name] = kpi_test_data
    return feature_map, train_data, test_data