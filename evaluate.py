import numpy as np

from affiliation.generics import convert_vector_to_events
from affiliation.metrics import pr_from_events

def cal_f1_score(precision, recall):
    return 2 * precision * recall / (precision + recall) if (precision + recall)!=0 else 0.0

# In[] affiliation metrics
def evaluate_affiliation_1dim_pred(y_true, y_pred, **args):
    bound_mode = args.get('bound_mode')
    events_pred = convert_vector_to_events(y_pred)
    events_gt = convert_vector_to_events(y_true)
    Trange = (0, len(y_pred))
    # bound_mode = 18
    res = pr_from_events(events_pred, events_gt, Trange, bound_mode)
    
    precision, recall = res['precision'], res['recall']
    f1_score = cal_f1_score(precision, recall)
    
    return precision, recall, f1_score  
# In[] point_wise metrics
def evaluate_pw_1dim_pred(y_true, y_pred, **args):
    TP, FP, FN = 0, 0, 0

    TP = np.sum((y_true + y_pred > 1.5))
    FN = np.sum((y_true - y_pred > 0.5))
    FP = np.sum((y_pred - y_true > 0.5))

    precision = TP / (TP + FP) if (TP + FP)!=0 else 0.0
    recall = TP / (TP + FN) if (TP + FN)!=0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall)!=0 else 0.0
    
    precision = round(precision, 4)
    recall = round(recall, 4)
    f1_score = round(f1_score, 4)
    return precision, recall, f1_score 

# In[] point_adjustment metrics
def interval_sample_1dim_pred(y_true, y_pred,**args):
    max_detect_delay = args.get('max_detect_delay')
    TP, FP, FN = 0, 0, 0
    i = 0
    j = 0
    while i < len(y_true):
        if y_true[i] > 0.5:
            j = i
            while j < len(y_true) and y_true[j] > 0.5:
                j += 1
            if sum(y_pred[i: j][:max_detect_delay]) > 0.5:
                TP += j - i
            else:
                FN += j - i
            i = j
        else:
            if y_pred[i] > 0.5:
                j = i 
                while j < len(y_true) and y_pred[j] > 0.5 and y_true[j] < 0.5:
                    j += 1
                FP += j - i
                i = j
            else:
                i += 1
    return TP, FP, FN
                
def evaluate_iw_1dim_pred(y_true, y_pred, **args):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP, FP, FN =  interval_sample_1dim_pred(y_true, y_pred, **args)
            
    precision = TP / (TP + FP) if (TP + FP)!=0 else 0.0
    recall = TP / (TP + FN) if (TP + FN)!=0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall)!=0 else 0.0
    
    precision = round(precision, 4)
    recall = round(recall, 4)
    f1_score = round(f1_score, 4)
    return precision, recall, f1_score    
 
# In[] aggregate metrics
def evaluate_1dim_prob_prc(y_true, y_prob, metric_name, prc_points=100, **args):
    min_prob = np.min(y_prob)
    max_prob = np.max(y_prob)
    thresholds = np.linspace(min_prob, max_prob, prc_points)
    precisions, recalls, f1_scores = [], [], []
    for th in thresholds:
        y_pred = [1 if y_prob[i] >= th else 0 for i in range(len(y_prob))]
        
        precision, recall, f1_score = eval(f"evaluate_{metric_name}_1dim_pred")(y_true, y_pred, **args)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)          
    return precisions, recalls, f1_scores, thresholds    

def evaluate_1dim_prob_best(y_true, y_prob, metric_name, **args):
    precisions, recalls, f1_scores, thresholds = evaluate_1dim_prob_prc(y_true, y_prob, metric_name, **args)
    f1_scores = np.array(f1_scores)    
    
    try:
        best_idx = np.argmax(f1_scores[np.isfinite(f1_scores)])
        best_f1_score = round(f1_scores[best_idx], 4)
        best_precision = round(precisions[best_idx], 4)
        best_recall = round(recalls[best_idx], 4)
        best_threshold = thresholds[best_idx]
    except:
        best_f1_score, best_precision, best_recall, best_threshold = 0.0, 0.0, 0.0, 0.0
    return best_precision, best_recall, best_f1_score, best_threshold    

def evaluate_muldim_prob_best(y_true, y_prob, metric_name, **args):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    precisions, recalls, f1_scores, thresholds = [], [], [], []
    
    for i in range(y_true.shape[0]):
        precision, recall, f1_score, th = evaluate_1dim_prob_best(y_true[i], y_prob[i], metric_name, **args)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        thresholds.append(th)
    
    total_precision = np.mean(precisions)
    total_recall = np.mean(recalls)
    total_f1_score = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall)!=0 else 0.0
    
    total_precision = round(total_precision, 4)
    total_recall = round(total_recall, 4)
    total_f1_score = round(total_f1_score, 4)
    return precisions, recalls, f1_scores, thresholds, total_precision, total_recall, total_f1_score 