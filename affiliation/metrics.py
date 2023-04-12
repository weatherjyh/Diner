#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from affiliation.generics import (
        infer_Trange,
        has_point_anomalies, 
        _len_wo_nan, 
        _sum_wo_nan,
        read_all_as_events)
from affiliation._affiliation_zone import (
        get_all_E_gt_func,
        E_rest_func,
        affiliation_partition)
from affiliation._single_ground_truth_event import (
        affiliation_precision_distance,
        affiliation_recall_distance,
        affiliation_precision_proba,
        affiliation_recall_proba)

def test_events(events):
    """
    Verify the validity of the input events
    :param events: list of events, each represented by a couple (start, stop)
    :return: None. Raise an error for incorrect formed or non ordered events
    """
    if type(events) is not list:
        raise TypeError('Input `events` should be a list of couples')
    if not all([type(x) is tuple for x in events]):
        raise TypeError('Input `events` should be a list of tuples')
    if not all([len(x) == 2 for x in events]):
        raise ValueError('Input `events` should be a list of couples (start, stop)')
    if not all([x[0] <= x[1] for x in events]):
        raise ValueError('Input `events` should be a list of couples (start, stop) with start <= stop')
    if not all([events[i][1] < events[i+1][0] for i in range(len(events) - 1)]):
        raise ValueError('Couples of input `events` should be disjoint and ordered')


# precision = TP/TP+FP，
# 只有在pred为1的地方才是有意义的，所以应该用pred为1的长度来算，而不是用全部的区间长度来算
# precision长度按照pred为1的点数来计算
def aff_part_len_not_none(aff_partition, bound_mode=None):
    """
    计算每个gt区间对应的各个pred区间的长度的和
    """
    interval_lengths = []
    for aff in aff_partition:
        length = 0
        for e in aff:
            if e is not None:
                length += e[1] - e[0]
        if bound_mode is None:
            interval_lengths.append(1 if length !=0 else 0)
        else:
            interval_lengths.append(length)
    return interval_lengths   
    
def pr_from_events(events_pred, events_gt, Trange, bound_mode='half_gt_avg_len'):
    """
    Compute the affiliation metrics including the precision/recall in [0,1],
    along with the individual precision/recall distances and probabilities
    
    :param events_pred: list of predicted events, each represented by a couple
    indicating the start and the stop of the event
    :param events_gt: list of ground truth events, each represented by a couple
    indicating the start and the stop of the event
    :param Trange: range of the series where events_pred and events_gt are included,
    represented as a couple (start, stop)
    :param bound_mode: {'half_len', 'half_gt_avg_len', numeric}
    'half_len' means every event (both gt and pred event) impacts the additional range (except for itself) of half its lenghth bilaterally. 
    'half_gt_avg_len' means every event impacts the additional range of half the averange gt length bilaterally. 
    A numeric means every event impacts the additional range of the length "bound_mode" bilaterally. 
    Other values mean the length of the impact area is unlimited
    :return: dictionary with precision, recall, and the individual metrics
    """
    # testing the inputs
    test_events(events_pred)
    test_events(events_gt)
    
    # other tests
    minimal_Trange = infer_Trange(events_pred, events_gt)
    if not Trange[0] <= minimal_Trange[0]:
        raise ValueError('`Trange` should include all the events')
    if not minimal_Trange[1] <= Trange[1]:
        raise ValueError('`Trange` should include all the events')
    
    if len(events_gt) == 0:
        raise ValueError('Input `events_gt` should have at least one event')

    if has_point_anomalies(events_pred) or has_point_anomalies(events_gt):
        raise ValueError('Cannot manage point anomalies currently')

    if Trange is None:
        # Set as default, but Trange should be indicated if probabilities are used
        raise ValueError('Trange should be indicated (or inferred with the `infer_Trange` function')
        
    gt_lengths = [p[1]-p[0] for p in events_gt]
    if bound_mode == "half_gt_avg_len":
        gt_avg_len = sum(gt_lengths) / len(gt_lengths)
        bound_mode = gt_avg_len / 2
    E_gt = get_all_E_gt_func(events_gt, Trange, bound_mode) # 附属区间
    aff_partition = affiliation_partition(events_pred, E_gt) # 将预测事件分配到附属区间
    
    E_rest = E_rest_func(E_gt, Trange) # 非附属区间
    aff_partition_rest = affiliation_partition(events_pred, E_rest) # 将预测时间分配到非附属区间

    # Computing precision distance, recall distance, precision, recall
    d_precision = [affiliation_precision_distance(Is, J) for Is, J in zip(aff_partition, events_gt)]
    # 计算 d_recall (recall距离) 时需要考虑长度限制(也可不考虑，没有影响，即从pred看gt事件时不需要考虑pred的延伸距离，因为过远的pred不会存在于该affiliation区间)
    d_recall = [affiliation_recall_distance(Is, J, None) for Is, J in zip(aff_partition, events_gt)]
    p_precision = [affiliation_precision_proba(Is, J, E) for Is, J, E in zip(aff_partition, events_gt, E_gt)]
    # 计算 p_recall (recall概率)时需要考虑长度限制
    p_recall = [affiliation_recall_proba(Is, J, E, None) for Is, J, E in zip(aff_partition, events_gt, E_gt)]
    
    # gt没有1, pred也没有1的部分，不参与precision的计算
    # 这里的pred为1的长度，考虑pred为1的点数量，或者pred为1的区间数量两种可能，pred为1的点数量更合理
    # 非附属区间的 单区间 precision 计算
    rest_p_precision = [math.nan for _ in range(len(E_rest))]
    for i in range(len(aff_partition_rest)):
        if any(aff_partition_rest[i]):
            rest_p_precision[i] = 0    
            
    # 计算 weighted_p_precision_average      
    interval_lengths = aff_part_len_not_none(aff_partition, bound_mode)    # 附属区间中的非空pred点数量
    rest_interval_lengths = aff_part_len_not_none(aff_partition_rest, bound_mode) # 非附属区间中的非空pred点数量
    valid_interval_lengths = interval_lengths + rest_interval_lengths    # 全部有效pred长度，gt=0he pred=0的非附属区间不参与precision计算
    total_p_precision = p_precision + rest_p_precision # 各个区间的 precision
    valid_precision_weights = valid_interval_lengths
    # 按照pred长度对precision进行加权。
    weighted_p_precision = [total_p_precision[i] * valid_precision_weights[i] for i in range(len(total_p_precision))]
    
    if _len_wo_nan(weighted_p_precision) > 0:
        weighted_p_precision_average = _sum_wo_nan(weighted_p_precision) / _sum_wo_nan(valid_precision_weights)
    else:
        weighted_p_precision_average = math.nan
          
    d_precision_add = d_precision + [float("inf") if rest_p_precision[i] == 0 else math.nan for i in range(len(rest_p_precision))]
    
    # 计算 p_recall_average 
    if _len_wo_nan(p_precision) > 0:
        p_precision_average = _sum_wo_nan(p_precision) / _len_wo_nan(p_precision)
    else:
        p_precision_average = p_precision[0] # math.nan
        
    # 计算 p_recall_average     
    p_recall_average = sum(p_recall) / len(p_recall)

    if bound_mode is None:
        dict_out = dict({'precision': p_precision_average,
                         'recall': p_recall_average,
                         'individual_precision_probabilities': p_precision,
                         'individual_recall_probabilities': p_recall,
                         'individual_precision_distances': d_precision,
                         'individual_recall_distances': d_recall})
    else:
        dict_out = dict({'precision': weighted_p_precision_average,
                         'recall': p_recall_average,
                         'individual_precision_probabilities': total_p_precision,
                         'individual_recall_probabilities': p_recall,
                         'individual_precision_distances': d_precision_add,
                         'individual_recall_distances': d_recall})
    return(dict_out)

def produce_all_results():
    """
    Produce the affiliation precision/recall for all files
    contained in the `data` repository
    :return: a dictionary indexed by data names, each containing a dictionary
    indexed by algorithm names, each containing the results of the affiliation
    metrics (precision, recall, individual probabilities and distances)
    """
    datasets, Tranges = read_all_as_events() # read all the events in folder `data`
    results = dict()
    for data_name in datasets.keys():
        results_data = dict()
        for algo_name in datasets[data_name].keys():
            if algo_name != 'groundtruth':
                results_data[algo_name] = pr_from_events(datasets[data_name][algo_name],
                                                         datasets[data_name]['groundtruth'],
                                                         Tranges[data_name])
        results[data_name] = results_data
    return(results)
