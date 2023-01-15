# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 15:50:27 2022

@author: 123
"""

import numpy as np
import torch
import torch.nn.functional as F



def topK_hit(test_set, y_pred, y_true, n_top=20):
    assert len(test_set.shape) == 1
    all_ = test_set.shape[0]
    hit_mask = np.isin(y_pred, y_true)
    hit_num = y_pred[hit_mask].shape[0]
    
    return hit_num/all_

def MRR(test_set, y_pred, y_true, n_top=20):
    all_ = test_set.shape[0]
    hit_mask = np.isin(y_pred, y_true)
    MRR_score = 0.0
    for group in hit_mask:
        index = 0
        while True:
            if group[index] == True:
                break
            index+1
        score = 1/index
        MRR_score+=score
    return MRR_score/all_


def validate(valid_loader, model, topk):
    model.eval()
    recalls = []
    mrrs = []
    with torch.no_grad():
        for seq, target, lens in valid_loader:
            seq = seq
            target = target
            outputs = model(seq, lens)
            logits = F.softmax(outputs, dim = 1)
            recall, mrr = evaluate(logits, target, k = topk)
            recalls.append(recall)
            mrrs.append(mrr)
    
    mean_recall = np.mean(recalls)
    mean_mrr = np.mean(mrrs)
    return mean_recall, mean_mrr        




def get_recall(indices, targets):
    """
    Calculates the recall score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.
    Returns:
        recall (float): the recall score
    """

    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices).nonzero()
    if len(hits) == 0:
        return 0
    n_hits = (targets == indices).nonzero()[:, :-1].size(0)
    recall = float(n_hits) / targets.size(0)
    return recall


def get_mrr(indices, targets):
    """
    Calculates the MRR score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.
    Returns:
        mrr (float): the mrr score
    """

    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    hits = (targets == indices).nonzero()
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks).data / targets.size(0)
    return mrr.item()


def evaluate(indices, targets, k=20):
    """
    Evaluates the model using Recall@K, MRR@K scores.
    Args:
        logits (B,C): torch.LongTensor. The predicted logit for the next items.
        targets (B): torch.LongTensor. actual target indices.
    Returns:
        recall (float): the recall score
        mrr (float): the mrr score
    """
    _, indices = torch.topk(indices, k, -1)
    recall = get_recall(indices, targets)
    mrr = get_mrr(indices, targets)
    return recall, mrr
