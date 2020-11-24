import torch
import torch.nn as nn

# Type hinting
from torch import LongTensor, FloatTensor
from typing import List
from sklearn.metrics import confusion_matrix,f1_score
import numpy as np

def random_pick_wrong_target(target: LongTensor) -> LongTensor:
    """After shuffling each row, pick the one with the index just before the target.
    Parameters:
        target: the tensor (shape ``(n_sample,)``) of auxiliary variables
                used for generalized contrastive training.
    Return:
        the randomized fake auxiliary variables to be made as negative targets in generalized
        contrastive learning (shape ``(n_sample,)``).
    Note:
        Duplicate entries in ``target`` are allowed; ``torch.unique()`` is applied.
    """
    unique_labels = torch.unique(target, sorted=False)[None, :]
    n_labels = unique_labels.shape[1]
    masked = torch.masked_select(unique_labels,
                                 unique_labels != target[:, None]).view(
                                     -1, n_labels - 1)
    random_indices = torch.randint(high=n_labels - 1, size=(len(target), ))
    wrong_labels = masked[torch.arange(len(masked)), random_indices]
    return wrong_labels


def permute_target(target: LongTensor) -> LongTensor:
    """After shuffling each row, pick the one with the index just before the target.
    Parameters:
        target: the tensor (shape ``(n_sample,)``) of auxiliary variables
                used for generalized contrastive training.
    Return:
        the randomized fake auxiliary variables to be made as negative targets in generalized
        contrastive learning (shape ``(n_sample,)``).
    Note:
        Duplicate entries in ``target`` are allowed; ``torch.unique()`` is applied.
    """
#     unique_labels = torch.unique(target, sorted=False)[None, :]
#     n_labels = unique_labels.shape[1]
#     masked = torch.masked_select(unique_labels,
#                                  unique_labels != target[:, None]).view(
#                                      -1, n_labels - 1)
#     random_indices = torch.randint(high=n_labels - 1, size=(len(target), ))
#     wrong_labels = masked[torch.arange(len(masked)), random_indices]
    
    perm = np.random.permutation(len(target))
    return target[perm]


def random_pick_wrong_target_full(target: LongTensor, target_list: LongTensor) -> LongTensor:
    """After shuffling each row, pick the one with the index just before the target.
    Parameters:
        target: the tensor (shape ``(n_sample,)``) of auxiliary variables
                used for generalized contrastive training.
        target_list: List of all labels, in case the minority label has been removed
    Return:
        the randomized fake auxiliary variables to be made as negative targets in generalized
        contrastive learning (shape ``(n_sample,)``).
    Note:
        Duplicate entries in ``target`` are allowed; ``torch.unique()`` is applied.
    """
    unique_labels = target_list
    n_labels = unique_labels.shape[1]
    masked = torch.masked_select(unique_labels,
                                 unique_labels != target[:, None]).view(
                                     -1, n_labels - 1)
    random_indices = torch.randint(high=n_labels - 1, size=(len(target), ))
    wrong_labels = masked[torch.arange(len(masked)), random_indices]
    return wrong_labels

LOG_LOGISTIC_LOSS = nn.SoftMarginLoss()

        
def count_rates(test_e):
    ids=np.unique(test_e) 
    event_rate=np.array([len(test_e[test_e==i]) for i in ids])/len(test_e)
    return(event_rate)

def binary_logistic_loss(outputs: FloatTensor, positive: bool):
    """Utility function to wrap ``torch.SoftMarginLoss``."""
    device = outputs.device
    if positive:
        return LOG_LOGISTIC_LOSS(outputs, torch.ones(len(outputs)).to(device))
    else:
        return LOG_LOGISTIC_LOSS(outputs, -torch.ones(len(outputs)).to(device))

def idx2onehot(idx, n):
    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)

    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx, 1)
    return onehot

def get_cls_count(label):
    label_list = torch.unique(label)
    per_cls_count = []
    
    for label_ in label_list:
        per_cls_count.append(len(label[label==label_]))
    return per_cls_count


def accuracy_per_class(predict, label):
    cm = confusion_matrix(label, predict)
    return np.diag(cm)/np.sum(cm, axis = 1)
    #return f1_score(label, predict, average=None)

# predict classification label
def get_predicted_label(recon_y):
    # get the largest probability in the two categories
    pred_label = np.argmax(recon_y,axis=1)
    # return the label only
    return np.array(pred_label)

# def cross_entropy(pred, target, class_acc=False):
#     nc = pred.size()[1] 
#     sample_weight = torch.tensor(count_rates(target)).float()
#     loss = nn.CrossEntropyLoss(weight = sample_weight)
#     # pred shape = [N,C]
#     output = loss(pred, target)
#     if class_acc == True:
#         pred_label =  get_predicted_label(pred.detach().cpu().numpy())
#         class_acc_output = accuracy_per_class(pred_label.squeeze(), target.squeeze().detach().cpu().numpy())
#         return output, class_acc_output
#     else: 
#         return output
    
def cross_entropy(pred, label, sample_weight = None, class_acc=False):
    nc = pred.size()[1] 
    loss = nn.CrossEntropyLoss(weight = sample_weight)
    output = loss(pred, label.to(pred.device))
    
    
    
    if class_acc == True:
#         pred_label =  get_predicted_label(pred.detach().cpu().numpy())
#         class_acc_output = accuracy_per_class(pred_label.squeeze(), \
#                                               label.squeeze().detach().cpu().numpy())
        class_acc_output = np.mean((np.array(label.tolist())==np.array(torch.argmax(pred,axis=1).tolist()))*1)
        return output, class_acc_output
    else: 
        return output

    
def collect_minority(x, label, label_idx): 
    minority_x = []
    minority_label = []
    for label_ in label_idx:
        minority_x.append(x[label == label_, :])
        minority_label.append(np.repeat(label_, (label==label_).sum()))
    min_xs = torch.cat(minority_x)
    min_labels = np.concatenate(minority_label)
    return min_xs, min_labels

def min_ce(pred, label, label_list):
    min_preds, min_label = collect_minority(pred, label, label_list)
    min_labels = torch.tensor(min_label).to(pred.device)
    loss = nn.CrossEntropyLoss()
    output = loss(min_preds, min_labels)
    return output

# if GCL collapsed, reset all parameters
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()