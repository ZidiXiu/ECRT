import math
import os

import numpy as np

import torch

from torch.distributions import normal

from torch import nn, optim
from torch.nn import functional as F

from torchvision import transforms, utils
from utils.trainer_util import cross_entropy, binary_logistic_loss, count_rates, permute_target, get_cls_count
from networks.gcl_torch_multiple import GeneralizedContrastiveICAModel
from utils.regularizer import get_minority_distribution, clip_abnormal

def train_critic(gcl, encoder, opt_flow, x, label, reg_weight=1e-2, minority_label=None, majority_label = None, trainable = True, print_corr=True):
    '''
    constractive learning the invertible flow through the critic net

    '''
    encoder.eval()
    
    if minority_label is not None:
        x, label = remove_minority(x, label, minority_label)
        major_idx = [majority_label.index(label_s.item()) for label_s in label]
    else:
        major_idx = label.tolist()
    
    z = encoder(x)    
    
    if trainable:
        opt_flow.zero_grad()
        fake_label =  permute_target(label)
        pos_output = gcl.classify((z, label),return_hidden=False)
        neg_output = gcl.classify((z, fake_label),return_hidden=False) 
        critic_loss = binary_logistic_loss(pos_output, True) + binary_logistic_loss(neg_output, False)
        
        # calculate corresponding log probability

        critic_loss -= reg_weight*gcl.logpdf(z, major_idx).mean()

        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(gcl.parameters(), 1e-3)

        opt_flow.step()
        
        return critic_loss.item()

    else:
        gcl.eval()
        fake_label =  permute_target(label)
        pos_output = gcl.classify((z, label),
                                         return_hidden=False)
        neg_output = gcl.classify((z, fake_label),
                                         return_hidden=False) 
        critic_loss = binary_logistic_loss(pos_output, True) + binary_logistic_loss(neg_output, False)
        critic_loss -= reg_weight*gcl.logpdf(z, major_idx).mean()
        if print_corr:
            sorted_xs = sorted_x(x, label)
            corr_mean = check_results(gcl, encoder, sorted_xs, l_max=False)

            return critic_loss.item(), corr_mean
        else:
            return critic_loss.item()
        
        
def train_critic_noreg(gcl, encoder, opt_flow, x, label, reg_weight=0, minority_label=None, majority_label = None, trainable = True, print_corr=True):
    '''
    constractive learning the invertible flow through the critic net

    '''
    encoder.eval()
    
    if minority_label is not None:
        x, label = remove_minority(x, label, minority_label)
        major_idx = [majority_label.index(label_s.item()) for label_s in label]
    else:
        major_idx = label.tolist()
    
    z = encoder(x)    
    
    if trainable:
        opt_flow.zero_grad()
        fake_label =  permute_target(label)
        pos_output = gcl.classify((z, label),return_hidden=False)
        neg_output = gcl.classify((z, fake_label),return_hidden=False) 
        critic_loss = binary_logistic_loss(pos_output, True) + binary_logistic_loss(neg_output, False)
        
        # calculate corresponding log probability

#         critic_loss -= reg_weight*gcl.logpdf(z, major_idx).mean()

        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(gcl.parameters(), 1e-3)

        opt_flow.step()
        
        return critic_loss.item()

    else:
        gcl.eval()
        fake_label =  permute_target(label)
        pos_output = gcl.classify((z, label),
                                         return_hidden=False)
        neg_output = gcl.classify((z, fake_label),
                                         return_hidden=False) 
        critic_loss = binary_logistic_loss(pos_output, True) + binary_logistic_loss(neg_output, False)
#         critic_loss -= reg_weight*gcl.logpdf(z, major_idx).mean()
        if print_corr:
            sorted_xs = sorted_x(x, label)
            corr_mean = check_results(gcl, encoder, sorted_xs, l_max=False)

            return critic_loss.item(), corr_mean
        else:
            return critic_loss.item()
    
def all_s(x, encoder, gcl):
    encoder.eval()
    gcl.eval()
    return gcl.hidden(encoder(x))


def label_augmenting_multiple(s, label, label_list=None, n_aug=None, regularize = False, minority_bound = 20):
    '''
    label_idx : minority class to be inflated
    n_aug: number of augmentation samples
    '''
    if type(label_list)!= list:
        label_list = [label_list]
    
    if n_aug is None:
        n_list = get_cls_count(label)
        n_aug = int(np.mean(n_list))
    
    if regularize:
        min_base_dist_mean, min_base_dist_var = get_minority_distribution({'s':s, 'label':label}, label_list)
        clipped_minority = clip_abnormal(s, label, label_list,\
                                         min_base_dist_mean, min_base_dist_var, limit = 0.005)        
    aug_s = []
    aug_label = []
    n_label = len(label_list)
    
    for label_idx in range(n_label):
        if n_aug[label_idx]==0:
            continue
#         print(label_idx)
        if not regularize:
            s_aug = label_augmenting_s(s, label, label_list[label_idx], n_aug[label_idx])
        else:
            if len(clipped_minority[label_idx]) < minority_bound:
                dist = normal.Normal(min_base_dist_mean[label_idx,:], min_base_dist_mean[label_idx,:])
                s_aug_para = dist.sample([minority_bound]).cpu().detach().numpy()
                for p in range(s.size()[1]):
                    clipped_minority[label_idx][p].extend(s_aug_para[:,p].tolist())
                    print(clipped_minority[label_idx][p])
            s_aug = label_augmenting_s_reg(clipped_minority[label_idx], n_aug[label_idx])
        aug_s.append(s_aug)
        aug_label.append(torch.tensor(np.repeat(label_list[label_idx], n_aug[label_idx])).long())

    return torch.cat(aug_s), torch.cat(aug_label)

def label_augmenting_s_reg(s_target, n_aug, device='cpu'):
    '''
    n_aug: number of augmentation samples
    s_target: a list of lists, minority s to be inflated, all with same label, but length of each dimension might be different
    '''
    s_dim = len(s_target)
    
    s_aug = []
    
    # by permutation each dimension of s
    for j in range(s_dim):
        n_target = len(s_target[j])
        ind = list(np.random.choice(n_target, n_aug))
        s_aug.append([s_target[j][idx] for idx in ind])
    
    s_aug = np.array(s_aug).T
#     print(s_aug.size())
    return torch.tensor(s_aug).to(device).float()



def label_augmenting_s(s_target, label, label_idx, n_aug):
    '''
    label_idx : minority class to be inflated
    n_aug: number of augmentation samples
    '''
    s_dim = s_target.size()[1]
    sub_idx = torch.where(label==label_idx)[0]
#     print(sub_idx)
    n_target = len(sub_idx)
    s_aug = []
    
    # by permutation each dimension of s
    for j in range(s_dim):
        # only random sampling from current label!
        s_j = s_target[sub_idx,j]
#         print('orig s: ',s_j.min().item(), s_j.max().item())
        ind = np.random.choice(n_target, n_aug)
#         print('aug s: ',s_j[ind].min().item(), s_j[ind].max().item())
        s_aug.append(s_j[ind])
    
    s_aug = torch.stack(s_aug).T
#     print(s_aug.size())
    return s_aug



def label_augmenting_s_x(z, label, label_idx, n_aug, gcl):
    '''
    label_idx : minority class to be inflated
    n_aug: number of augmentation samples
    '''
    gcl.eval()
    z_dim = z.size()[1]
    sub_idx = torch.where(label==label_idx)[0]
    n_target = len(sub_idx)
    s_target = gcl.hidden(z[sub_idx,:])
    s_aug = []
    
    # by permutation each dimension of s
    for j in range(z_dim):
        ind = np.random.choice(n_target, n_aug)
        s_aug.append(s_target[ind,j])
    
    s_aug = torch.stack(s_aug).T  
    return s_aug

def label_augmenting_multiple_x(x, label, label_list=None,\
                              gcl=None, encoder=None, decoder=None):
    '''
    label_idx : minority class to be inflated
    n_aug: number of augmentation samples
    '''
    gcl.eval()
    encoder.eval()
    z =  encoder(x)
    s = gcl.hidden(z)

    n_list = get_cls_count(label)
    n_avg = int(np.mean(n_list))
    aug_s = []
    aug_s.append(s)

    aug_label = []
    aug_label.append(label.long())
    
    for label_idx in label_list:
        s_aug = label_augmenting_s(s, label, label_idx, n_avg, gcl)     
        aug_s.append(s_aug)
        aug_label.append(torch.tensor(np.repeat(label_idx, n_avg)).long())

    return torch.cat(aug_s), torch.cat(aug_label)

# def label_augmenting_multiple_archive(s, label, label_list=None, n_aug=None):
#     '''
#     label_idx : minority class to be inflated
#     n_aug: number of augmentation samples
#     '''
#     if type(label_list)!= list:
#         label_list = [label_list]
    
#     if n_aug is None:
#         n_list = get_cls_count(label)
#         n_aug = int(np.mean(n_list))
        
#     aug_s = []
#     aug_label = []
    
#     for i, label_idx in enumerate(label_list):
#         s_aug = label_augmenting_s(s, label, label_idx, n_aug[i])     
#         aug_s.append(s_aug)
#         aug_label.append(torch.tensor(np.repeat(label_idx, n_aug[i])).long())
#     return torch.cat(aug_s), torch.cat(aug_label)


def train_predictor_s(s, label, class_weight,\
                      decoder=None,opt_dec=None):

    opt_dec.zero_grad()   
 
    pred_label = decoder(s)
    CE_loss = cross_entropy(pred_label, label, class_weight)
          
    CE_loss.backward()
    opt_dec.step()
    
    return CE_loss

def train_predictor_s_aug(s, label, s_aug, label_aug, class_weight, class_weight_aug,\
                      decoder=None,opt_dec=None, device='cpu'):
    if s_aug is None:
        CE_loss = train_predictor_s(s, label, class_weight, decoder=decoder,opt_dec=opt_dec)
        return CE_loss

    opt_dec.zero_grad()   
    pred_label = decoder(s.to(device))
    pred_label_aug = decoder(s_aug.to(device))    
    CE_loss_raw = cross_entropy(pred=pred_label, label=label, sample_weight=class_weight, class_acc=False)
    CE_loss_aug = cross_entropy(pred=pred_label_aug, label=label_aug, sample_weight=class_weight_aug, class_acc=False)
    
    CE_loss = CE_loss_raw + CE_loss_aug
    CE_loss.backward()
    opt_dec.step()
    return CE_loss
    
def train_predictor_x(x, label, gcl=None,\
                      encoder=None, decoder=None,opt_dec=None, device='cpu'):
    gcl.eval()
    encoder.eval()
    z =  encoder(x.to(device))
    s = gcl.hidden(z.to(device))
    opt_dec.zero_grad()   

    pred_label = decoder(s.to(device))
    CE_loss = cross_entropy(pred_label, label)
          
    CE_loss.backward()
    opt_dec.step()
    return CE_loss.item()
    
def train_predictor_x_multiple(x, label, augmenting=False, n_aug=0, label_list=None,\
                      gcl=None, encoder=None, decoder=None,opt_dec=None):
    gcl.eval()
    encoder.eval()
    z =  encoder(x)
    s = gcl.hidden(z)
    opt_dec.zero_grad()   
 
    if augmenting:
        aug_s = []
        aug_s.append(s)

        aug_label = []
        aug_label.append(label)

        for label_idx in label_list:
            s_aug = label_augmenting_s(z, label, label_idx, n_aug, gcl)     
            aug_s.append(s_aug)
            aug_label.append(torch.tensor(np.repeat(label_idx, n_aug)))
        
        pred_label = decoder(torch.cat(aug_s))
        CE_loss = cross_entropy(pred_label, torch.cat(aug_label))
        
    else:
        pred_label = decoder(s)
        CE_loss = cross_entropy(pred_label, label)
          
    CE_loss.backward()
    opt_dec.step()
          

def eval_predictor_x(x, label, gcl=None, encoder=None, decoder=None):
    gcl.eval()
    encoder.eval()
    decoder.eval()
    
    z =  encoder(x) 
    s = gcl.hidden(z)
    pred_label = decoder(s)
    
    CE_loss, acc_ = cross_entropy(pred_label, label, class_acc=True)
          
    return pred_label, CE_loss, acc_

# return label count in label_list
def count_label(label_, label_list):
    label_count = label_.unique(return_counts=True)
#     print(label_count )
    output = []
    if type(label_list) is not list:
        label_list = [label_list]
    for label_idx in label_list:
        output.append(label_count[1][label_count[0]==label_idx])
        
    return output

# augment to maximum cateroty in the batch
def number_aug(label_, minority_label):
    n_aug_list = []
    raw_count = count_label(label_, minority_label)
    max_count = (label_.unique(return_counts=True)[1]).max()
    for raw_ in raw_count:
        n_aug = max_count-raw_
#         return(n_aug)
        if n_aug == 0:
            n_aug = n_aug+1
            
        n_aug_list.append(n_aug)
    return torch.cat(n_aug_list).detach().tolist()


# Remove minority class from batch
def remove_minority(x, label, label_list):
    if type(label_list) != list:
        label_list = [label_list]
    x_new, label_new = x, label
    for label_idx in label_list:
        x_new, label_new = x_new[label_new!= label_idx,:], label_new[label_new!= label_idx]
    return x_new, label_new

def select_minority(x, label, label_list):
    if type(label_list) != list:
        label_list = [label_list]
    x_new, label_new = [],[]
    for label_idx in label_list:
        x_new.append(x[label==label_idx])
        label_new.append(label[label==label_idx])
    return torch.cat(x_new).squeeze(), torch.cat(label_new)

def get_minority_s(data_loader, minority_label, encoder, gcl, device):
    # starting with a dataloader
    
    min_s = []
    min_label= []
    for batched_x, batched_label in data_loader:
        batched_x = batched_x.to(device).float().view(batched_x.shape[0], -1)
    #     batched_x, batched_label = batched_sample
    #     batched_x = batched_x.to(device).float().view(batched_x.shape[0], -1)
        matched_label = np.intersect1d(batched_label.detach().numpy(), np.array(minority_label))
        if not matched_label.size>0:
            continue
        else:    
            minority_x, minority_labels = select_minority(batched_x, batched_label, label_list=minority_label)
            if len(minority_labels) == 1:
                minority_x = minority_x.view(1,minority_x.shape[0]).to(device)
            min_s.append(all_s(minority_x.to(device), encoder, gcl))
            min_label.append(minority_labels)
    minority_ = {"s":torch.cat(min_s).detach(), 'label':torch.tensor(np.concatenate(min_label))}
    return minority_

def sorted_x(x_, label_):
    unique_label_= np.unique(np.array(label_.tolist()))
    x_sorted_list = []
    for label in unique_label_:
        x_sorted_list.append(x_[label_==label])
    return x_sorted_list


def check_results(gcl, encoder, sorted_xs, l_max=True):
    gcl.eval()
    encoder.eval()
    corr=[]
    for x in sorted_xs:
        z = encoder(torch.tensor(x).float())
        ls1 = gcl.hidden(z)[:,0]
        ls2 = gcl.hidden(z)[:,1]
        corr.append(np.corrcoef(ls1.detach().cpu().numpy(), ls2.detach().cpu().numpy())[0,1])
#     print('class correlation:', corr)
    if l_max:
        return np.nanmax(np.abs(corr))
    else:
        return(np.nanmean(np.abs(corr)))
    
def check_corr(sorted_xs, l_max=True):
    corr=[]
    for x in sorted_xs:
        ls1 = x[:,0]
        ls2 = x[:,1]
        corr.append(np.corrcoef(ls1, ls2)[0,1])
#     print('class correlation:', corr)
    if l_max:
        return np.nanmax(np.abs(corr))
    else:
        return(np.nanmean(np.abs(corr)))
    
# if GCL collapsed, reset all parameters
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()