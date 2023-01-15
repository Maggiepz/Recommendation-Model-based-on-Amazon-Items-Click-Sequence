# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 17:13:53 2022

@author: 123
"""

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from utils import *
import random


#增加购物篮子序列函数

def ordered(df, session_key = 'session', label_key = 'label', user_key = None, neg_item_key = None):
    #print(df.shape[0])
    session_id = 0
    df_ordered = []
    for i in range(df.shape[0]):
        df_session = {}
        if user_key:
            userId = df[user_key].values[i]
        if neg_item_key:
            neg_items = df[neg_item_key].values[i]
            neg_items_next = neg_items[1:] + [(df[label_key].values[i]//2+1)]
        try:
            items = [int(item) for item in df[session_key].values[i].replace('[','').replace(']','').split(',')]
        except:
            items = df[session_key].values[i]
        next_ = items[1:]+[df[label_key].values[i]]
        df_session['sessionId'] = [session_id]*len(items)
        df_session['itemsId'] = items
        df_session['Next'] = next_
        df_session['orderId'] = list(range(0,len(items)))
        try:
            df_session['UserId'] = [userId]*len(items)
        except:
            pass
        try:
            df_session['Neg_itemsId'] = neg_items
            df_session['Neg_itemsId_Next'] = neg_items_next
        except:
            pass
        #for k, v in df_session.items():
        #    print(k, len(v))
        df_session = pd.DataFrame(df_session)
        df_ordered.append(df_session)
        
        session_id+=1
    result = pd.concat(df_ordered, ignore_index=True)
    result.loc[:,'itemsId']+=1
    assert result.itemsId.min()!=0
    return result

def basket_order(df, user_key = 'UserId', basket_key='session'):
    try:
        
        baskets = df[basket_key].apply(lambda x:set([int(i) for i in x.replace('[','').replace(']','').split(',')]))
    except:
        baskets = df[basket_key].values
    basket_dict = []
    index=0
    for basket in baskets:
        if basket not in [i[1] for i in basket_dict]:
            basket_dict.append((index, basket))
            index+=1
    basket_ids = []
    for basket in baskets:
        for tup in basket_dict:
            if basket == tup[1]:
                basket_ids.append(tup[0])
                break
    df.loc[:,'basketId'] = basket_ids
    
    df_basket = pd.DataFrame(df.groupby([user_key])['basketId'].apply(list)).reset_index()
    df_basket.loc[:,'next_basket'] = df_basket.basketId.apply(lambda x:x[-1])
    df_basket.loc[:,'basketId'] = df_basket.basketId.apply(lambda x:x[0:-1])
    df_basket = df_basket.loc[df_basket.basketId.apply(lambda x:len(x)>0),:]
    
    return df_basket , basket_dict
        


def generate_users(df, n_user = 8000, user_key = 'UserId'):
    df.loc[:, user_key] = np.random.uniform(0, n_user, df.shape[0]).astype(int)
    return df

def sequence(df_ordered=None, 
             cold_launcher = None, 
             session_key = 'sessionId', 
             item_key='itemsId', 
             next_key = 'Next', 
             seq_len = 5):
    sequence_data = []
    for session_id in df_ordered[session_key].unique():
        input_items = df_ordered.loc[df_ordered[session_key] == session_id, item_key].values
        output_items = df_ordered.loc[df_ordered[session_key] == session_id, next_key].values
        
        #input_items = np.append(input_items, output_items[0])
        output_items = np.concatenate([[input_items[0]], output_items])
        #print(len(input_items), len(output_items))
        if cold_launcher:
            fill_sequence = cold_launcher(seq_len-1)
            #print(fill_sequence)
            input_items = np.concatenate([fill_sequence[0:seq_len], input_items])
            output_items = np.concatenate([fill_sequence[1:], output_items])
            #print(input_items, output_items)
            assert len(input_items) == len(output_items)
            index=0
            
            while index<=input_items.shape[0]-seq_len:
                input_seq = input_items[index:(index+seq_len)]
                output_seq = output_items[index:(index+seq_len)]
                #print('???', input_seq, output_seq)
                sequence_data.append((input_seq, output_seq))
                index+=1
        else:
            index=1
            
            while index<=input_items.shape[0]:
                input_seq = input_items[0:index]
                output_seq = output_items[0:index+1]
                #print('???', input_seq, output_seq)
                sequence_data.append((input_seq, output_seq))
                index+=1
    
    return sequence_data


def sequence_(df, 
             item_key='itemsId', 
             next_key = 'label',
             max_len=None):
    
    try:
        df.loc[:,item_key] = df[item_key].apply(lambda x:[int(i)+1 for i in x.replace('[','').replace(']','').split(',')])
    except:
        pass
    
    df.loc[:,item_key+'_len'] = df[item_key].apply(lambda x:len(x))
    if not max_len:
        max_len = df[item_key+'_len'].max()
    else:
        pass
    #max_len_list = [max_len]*df.shape[0]
    df.loc[:,item_key+'_pad'] = df.apply(lambda x:x[item_key]+[0]*(max_len-x[item_key+'_len']), axis=1)
    
    return df[item_key+'_pad'].tolist(), df[next_key].tolist(), df[item_key+'_len'].tolist(), max_len


def make_graph(df_ordered, item_key = 'itemsId', next_key = 'Next', weights = False):
    graph = defaultdict(list)
    for i in range(df_ordered.shape[0]):
        sub = df_ordered[item_key].values[i]
        obj = df_ordered[next_key].values[i]
        graph[sub].append(obj)
    
    graph_new = []
    counter = Counter()
    if weights:
        for k, v in graph.items():
            for point in v:
                counter[point]+=1
            sum_ = sum(counter.values())+1
            weights = list(map(lambda x:x/sum_, counter.values())) 
            points = list(counter.keys())
            graph_new.append([(k,1/sum_)] + [(point, weight) for point, weight in zip(points, weights)])
        
        return graph_new
    
    else:
        for k, v in graph.items():
            graph_new.append([k] + list(set(v)))
        return graph_new


def local_graph(df_ordered = None,
                kernel = None,
                window = 5,
                session_key = 'sessionId',
                item_key = 'itemsId',
                next_key = 'Next', weights = False):
    n_step = window//2
    kernel_index = list(df_ordered[df_ordered[item_key] == kernel].index())
    index_local = []
    for index in kernel_index:
        index_chunk = list(range(index-n_step, index)) + [index] + list(range(index, index+n_step+1))    
        index_local += index_chunk
    
    index_local = [index for index in index_local if index>=0]
    df_local = df_ordered.loc[index_local,:]
    
    graph_local = make_graph(df_local, weights=weights)
    
    return graph_local





            
            
            
    