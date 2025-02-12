# -*- coding: utf-8 -*-
"""
@author: yy
"""
import torch
import torch.utils
import torch.utils.data
import numpy as np
import copy
import collections
import random


def split_train_test(n, nfolds, seed):
    idx = [i for i in range(n)]
    random.seed(seed)
    random.shuffle(idx)
    stride = int(n/nfolds)
    
    idx = [idx[i*stride:(i+1)*stride] for i in range(nfolds)]
    train_idx, test_idx = {},{}
    for fold in range(nfolds):
        test_idx[fold] = np.array(copy.deepcopy(idx[fold]))
        train_idx[fold] = []
        for i in range(nfolds):
            if i!=fold:
                train_idx[fold] += idx[i]
        train_idx[fold] = np.array(train_idx[fold])
    return train_idx, test_idx

def split_train_valid_test(n, nfolds, rnd_state=None):
    rnd_state = np.random.RandomState() if rnd_state is None else rnd_state
    idx = rnd_state.permutation(n)
    idx = idx.tolist()
    stride = int(n/nfolds)
    # 先把idx分成5份
    idx = [idx[i*stride:(i+1)*stride] for i in range(nfolds)]
    train_idx, valid_idx, test_idx = {},{},{}
    for fold in range(nfolds):
        test_idx[fold] = np.array(copy.deepcopy(idx[fold]))
        valid_idx[fold] = np.array(copy.deepcopy(idx[(fold+1)%nfolds]))
        train_idx[fold] = []
        for i in range(nfolds):
            if i!=fold and i!=(fold+1)%nfolds:
                train_idx[fold] += idx[i]
        train_idx[fold] = np.array(train_idx[fold])
    return train_idx, valid_idx, test_idx


def make_batch(self,train_ids, batch_size, seed):
    """
    return a list of batch ids for mask-based batch.
    Args:
        train_ids: list of train ids
        batch_size: ~
    Output:
        batch ids, e.g., [[1,2,3], [4,5,6], ...]
    """
    num_data = len(train_ids)
    rnd_state = seed
    permuted_idx = rnd_state.permutation(num_data)
    permuted_train_ids = train_ids[permuted_idx]
    batches = [permuted_train_ids[i*batch_size:(i+1)*batch_size] for i in range(int(num_data/batch_size))]
    if num_data%batch_size > 0:
        batches.append(permuted_train_ids[(num_data-num_data%batch_size):])

    return batches, num_data
