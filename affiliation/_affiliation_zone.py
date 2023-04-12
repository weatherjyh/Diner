#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numbers
from affiliation._integral_interval import interval_intersection

def t_start(j, Js = [(1,2),(3,4),(5,6)], Trange = (1,10)):
    """
    Helper for `E_gt_func`
    
    :param j: index from 0 to len(Js) (included) on which to get the start
    :param Js: ground truth events, as a list of couples
    :param Trange: range of the series where Js is included
    :return: generalized start such that the middle of t_start and t_stop 
    always gives the affiliation zone
    """
    b = max(Trange)
    n = len(Js)
    if j == n:
        return(2*b - t_stop(n-1, Js, Trange))
    else:
        return(Js[j][0])

def t_stop(j, Js = [(1,2),(3,4),(5,6)], Trange = (1,10)):
    """
    Helper for `E_gt_func`
    
    :param j: index from 0 to len(Js) (included) on which to get the stop
    :param Js: ground truth events, as a list of couples
    :param Trange: range of the series where Js is included
    :return: generalized stop such that the middle of t_start and t_stop 
    always gives the affiliation zone
    """
    if j == -1:
        a = min(Trange)
        return(2*a - t_start(0, Js, Trange))
    else:
        return(Js[j][1])

def E_gt_func(j, Js, Trange, bound_mode):
    """
    Get the affiliation zone of element j of the ground truth
    
    :param j: index from 0 to len(Js) (excluded) on which to get the zone
    :param Js: ground truth events, as a list of couples
    :param Trange: range of the series where Js is included, can 
    be (-math.inf, math.inf) for distance measures
    :return: affiliation zone of element j of the ground truth represented
    as a couple
    """
    range_left = (t_stop(j-1, Js, Trange) + t_start(j, Js, Trange))/2
    range_right = (t_stop(j, Js, Trange) + t_start(j+1, Js, Trange))/2
    
    if bound_mode == "half_len":
        gt_len = t_stop(j, Js, Trange) - t_start(j, Js, Trange)
        range_left_bound = t_start(j, Js, Trange) - gt_len/2
        range_right_bound = t_stop(j, Js, Trange) + gt_len/2
    elif isinstance(bound_mode, numbers.Number):
        range_left_bound = t_start(j, Js, Trange) - bound_mode
        range_right_bound = t_stop(j, Js, Trange) + bound_mode
    else:
        range_left_bound = range_left
        range_right_bound = range_right 
    
    range_left = max(range_left, range_left_bound)
    range_right = min(range_right, range_right_bound)
    return((range_left, range_right))

float_diff = 1e-2
def E_rest_func(E_gt, Trange):
    E_rest = []
    if E_gt[0][0] - Trange[0] > float_diff:
        E_rest.append((Trange[0], E_gt[0][0]))
    for j in range(1, len(E_gt)):
        if E_gt[j][0] - E_gt[j-1][1] > float_diff:
            E_rest.append((E_gt[j-1][1], E_gt[j][0]))
    if Trange[1] - E_gt[-1][1] > float_diff:
        E_rest.append((E_gt[-1][1], Trange[1]))   
    return E_rest
        
def get_all_E_gt_func(Js, Trange, bound_mode=None):
    """
    Get the affiliation partition from the ground truth point of view
    
    :param Js: ground truth events, as a list of couples
    :param Trange: range of the series where Js is included, can 
    be (-math.inf, math.inf) for distance measures
    :return: affiliation partition of the events
    """
    # E_gt is the limit of affiliation/attraction for each ground truth event
    E_gt = [E_gt_func(j, Js, Trange, bound_mode) for j in range(len(Js))]
    return E_gt

def affiliation_partition(Is = [(1,1.5),(2,5),(5,6),(8,9)], E_gt = [(1,2.5),(2.5,4.5),(4.5,10)]):
    """
    Cut the events into the affiliation zones
    The presentation given here is from the ground truth point of view,
    but it is also used in the reversed direction in the main function.
    
    :param Is: events as a list of couples
    :param E_gt: range of the affiliation zones
    :return: a list of list of intervals (each interval represented by either 
    a couple or None for empty interval). The outer list is indexed by each
    affiliation zone of `E_gt`. The inner list is indexed by the events of `Is`.
    """
    out = [None] * len(E_gt)
    for j in range(len(E_gt)):
        E_gt_j = E_gt[j]
        discarded_idx_before = [I[1] < E_gt_j[0] for I in Is]  # end point of predicted I is before the begin of E
        discarded_idx_after = [I[0] > E_gt_j[1] for I in Is] # start of predicted I is after the end of E
        kept_index = [not(a or b) for a, b in zip(discarded_idx_before, discarded_idx_after)]
        Is_j = [x for x, y in zip(Is, kept_index)]
        out[j] = [interval_intersection(I, E_gt[j]) for I in Is_j]
    # print()
    # print('---affiliation_partition---')
    # print('Is', Is)
    # print('E_gt', E_gt)
    # print('out', out)
    return(out)
