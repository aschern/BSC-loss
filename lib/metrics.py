import numpy as np
import pandas as pd


def MRR(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

def precision_at_k(r, k):
    r = np.array(r)
    return np.mean(r[:k])


def P_k(rs, k):
    return np.mean([precision_at_k(r, k) for r in rs])


def average_precision(r):
    r = np.array(r)
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if r.sum() > 0:
        return np.sum(out)/r.sum()
    else:
        return 0


def MAP(rs):
    return np.mean([average_precision(r) for r in rs])

def dcg_at_k(r, k, method=0):
    '''
    method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    '''
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max
    

def nDCG_k(rs, k, method=0):
    return np.mean([ndcg_at_k(r, k, method) for r in rs])


def calculate_metrics(pairs, preds, labs, labs_bin):
    test_question_ids = np.unique([x[0] for x in pairs])
    retrieve_results = []
    retrieve_results_multi = []
    for el in test_question_ids:
        answer_vars = []
        for i in range(len(pairs)):
            if pairs[i][0] == el:
                answer_vars.append([preds[i], labs_bin[i], labs[i]])
        retrieve_results.append([x[1] for x in sorted(answer_vars, key=lambda k: k[0], reverse=True)])
        retrieve_results_multi.append([x[2] for x in sorted(answer_vars, key=lambda k: k[0], reverse=True)])
        
    metrics = pd.DataFrame()
    metrics['metric'] = ['MAP', 'MRR', 'P@1', 'P@3', 'P@9', 'nDCG@1', 'nDCG@3', 'nDCG@9']
    metrics['value'] = [MAP(retrieve_results),
                         MRR(retrieve_results),
                         P_k(retrieve_results, 1),
                         P_k(retrieve_results, 3),
                         P_k(retrieve_results, 9),
                         nDCG_k(retrieve_results_multi, 1, 1),
                         nDCG_k(retrieve_results_multi, 3, 1),
                         nDCG_k(retrieve_results_multi, 9, 1)]
    metrics['value'] = metrics['value'].apply(lambda x: round(x, 4))
    return metrics
