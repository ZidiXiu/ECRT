from __future__ import absolute_import, division, print_function

import math
import os

import numpy as np
import pandas
import typing

import seaborn as sns

import matplotlib.pyplot as plt
import torch

import sys

from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
from data.dataLoader import SimpleDataset

import pandas as pd
# from torch.utils.tensorboard import SummaryWriter
from torch.distributions import normal
import sklearn.metrics

# get mean and variances by label
def get_minority_distribution(minority_, minority_label):
    if type(minority_label) is not list:
        minority_label = [minority_label]
    base_dist_mean = []
    base_dist_var = []
    
    for label_idx in minority_label:        
        sub_idx = np.where(minority_['label'] ==label_idx)[0]
        base_dist_mean.append(minority_['s'][sub_idx,:].mean(axis = 0).tolist())
        base_dist_var.append(minority_['s'][sub_idx,:].std(axis = 0).tolist())
        
    return torch.tensor(base_dist_mean).to(minority_['s'].device), torch.tensor(base_dist_var).to(minority_['s'].device)


def clip_abnormal(s_, label_, label_list, base_dist_mean, base_dist_var, limit = 0.05):
    '''
    s_: source value to be clipped
    label_: s_'s corresponding label
    label_list: list of labels to be cut
    base_dist_mean, base_dist_var: tensors of mean and std of correspondant to each label
    '''
    # 
    n, p = s_.size()
    # get lower and upper bounds
    boundaries = clipping_value(label_list, base_dist_mean, base_dist_var, limit = limit)
    
    # initiate output dictionary
    # key is majority label
    # value is the clipped value
    clipped = {}

    for label_idx in label_list:
        sub_idx = torch.where(label_ == label_idx)[0]
        clipped[label_idx] = clip_abnormal_helper(s_[sub_idx,:], boundaries[label_idx])
    return clipped
        
def clip_abnormal_helper(s_label, boundary):
    l_bound, h_bound = boundary
    # all s_ need to share one common label here
    n, p = s_label.size()
    
    clipped_label = []
    # clipped by dimension
    for j in range(p):
        condition = (s_label[:,j]<h_bound[j]) & (s_label[:,j]>l_bound[j])
        clipped_label.append(s_label[:,j][condition].tolist())
    # return clipped list of s values per label
    return clipped_label

def clipping_value(majority_label, base_dist_mean, base_dist_var, limit = 0.05):
    # for each major label, get it's lower and upper bounds
    n_major = len(majority_label)
    clipped = {}
    for i in range(n_major):
        dist = normal.Normal(base_dist_mean[i,:], base_dist_var[i,:])
        l_bound = dist.icdf(torch.tensor(limit).to(base_dist_mean.device)).tolist()
        h_bound = dist.icdf(torch.tensor(1-limit).to(base_dist_mean.device)).tolist()
        clipped[majority_label[i]] = [l_bound, h_bound]
    return clipped
