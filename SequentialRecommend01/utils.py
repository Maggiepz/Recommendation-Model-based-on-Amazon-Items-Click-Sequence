# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 11:12:39 2022

@author: 123
"""

import pandas as pd
import numpy as np
import random

import torch
from sklearn.model_selection import train_test_split


def to_tensor(list_):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tensor_ = torch.tensor(list_).long().to(device)
    return tensor_

def train_test_split(X, y, test_size = 0.15, random_state = 42):
    X_train,X_test, y_train, y_test = train_test_split(X, y, test_size, random_state)
    return X_train, X_test, y_train, y_test
    


def batch_loader(data, batch_size=64):
    
    if isinstance(data, list):
        index = -(batch_size)
        while index<len(data)-batch_size:
            #print('???', index)
            
            index+=batch_size
            batch = data[index:index+batch_size]
            if len(batch) == batch_size:
                yield data[index:index+batch_size]
            else:
                
                
                delta = batch_size-(len(data) - index)
                delta_batch = random.choices(data, k = delta)
                batch += delta_batch
                
                yield batch



def nagetive_sample(unique_items, seqs, next_, n_sample = 1):
    
    
    nagetive_samples = []
    count = 0
    while count<n_sample:
        
        pass_ = random.uniform(0,1)
        if pass_>0.5:
            nagetive = random.choices(unique_items, k = 1)[0]
            if nagetive not in seqs and nagetive!=next_:
                nagetive_samples.append(nagetive)
                count+=1
        else:
            continue
        
        
    
    return nagetive_samples


def stat(df, user_key = 'UserId', items_key = 'itemsId', session_key='sessionId'):
    n_users = df[user_key].unique()
    items_unique = df[items_key].unique()
    seq_len = df.groupby([session_key])[items_key].count()
    
    return n_users, items_unique, seq_len


class ColdLauncher(object):
    def __init__(self, method = 'base'):
        self.method = method
        
    def base(self, K):
        return np.array([-1]*K)
    
    def run(self, K):
        if self.method == 'base':
            return self.base(K)


                
            