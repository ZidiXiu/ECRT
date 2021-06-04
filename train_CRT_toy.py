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
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
from pathlib import Path

import argparse
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image

from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, utils
import pandas as pd
# from torch.utils.tensorboard import SummaryWriter
from torch.distributions import normal
import sklearn.metrics

# load models
from utils.plot import visualize_s_z_space, visualize_space
from data.create_dataset import myDataset_nofake_dic
from utils.trainer_util import binary_logistic_loss, idx2onehot, random_pick_wrong_target, count_rates, cross_entropy
from networks.gcl_multiple import GeneralizedContrastiveICAModel
from networks.EncoderNN import EncoderDecoderNN, DecoderNN

# load trainer functions
from data.dataLoader import SimpleDataset

from trainer.pre_trainer import train_predictor_pretrain
from trainer.GCL_multiple_trainer import train_critic, weight_reset

from trainer.GCL_multiple_trainer import remove_minority, select_minority, all_s
from trainer.GCL_multiple_trainer import eval_predictor_x, train_predictor_s, label_augmenting_multiple, get_minority_s
from trainer.GCL_multiple_trainer import count_label, number_aug

from torchvision import transforms, utils
from utils.trainer_util import cross_entropy, binary_logistic_loss, count_rates, permute_target, get_cls_count

# load data
import torchvision
import torchvision.transforms as transforms
from data.MNIST import IMBALANCEMNIST

from pathlib import Path

# except minority label, sample other label equally
from data.dataLoader import ImbalancedDatasetSampler, callback_get_label, Dataset_dic
from data.create_dataset import simulated_data_ablation

parser = argparse.ArgumentParser(description='PyTorch Toy Dataset Training')
parser.add_argument('--dataset', default='toy', help='dataset setting')
parser.add_argument('--minority_label', default=0, type=int, help='minority label')
parser.add_argument('--dir_path', default='.', type=str, help='result path')
parser.add_argument('--imb_factor', default=0.2, type=float, help='imbalance factor')

parser.add_argument('--lmbda', default=1e-3, type=float, help='augmentation strength')
parser.add_argument('--reg_weight', default=1e-3, type=float, help='likelihood regularization weight')

parser.add_argument('--imb_type', default="minority", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.2, type=float, help='imbalance factor')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size')

parser.add_argument('--enc_lr', '--encoder-learning-rate', default=1e-4,
                    type=float, help='encoder learning rate')
parser.add_argument('--gcl_lr', '--gcl-learning-rate', default=1e-5,
                    type=float, help='contrastive learning rate')
parser.add_argument('--dec_lr', '--dec-learning-rate', default=1e-4,
                    type=float, help='decoder learning rate')

args = parser.parse_args()

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

        
def main(minority_label):
    # set hyper-parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = args.dir_path + '/'+args.dataset
    plot_path = model_path + '/plot'

    model_name = 'CRT_'+str(len(args.minority_label))

    ncat = len(np.unique(label_))
    ncov = x_.shape[1]
    z_dim = 2
    if args.minority_label == 0:
        args.minority_label = [0]
        n_sample_list = [200] + [2000]*6
    else:
        args.minority_label = [0, 1, 2]
        n_sample_list = [200]*3 + [2000]*4

    n_sample_list = 
    x_, label_ = simulated_data_ablation(n_sample_list = n_sample_list)
        
    Path(model_path).mkdir(parents=True, exist_ok=True)
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    Path(data_path).mkdir(parents=True, exist_ok=True)

    ## permute the sources
    np.random.seed(12)
    n_samples = len(x_)
    
    permuted_idx = np.random.permutation(n_samples)
    train_idx = permuted_idx[:int(2*n_samples/3)]
    valid_idx = permuted_idx[int(2*n_samples/3):n_samples]

    train_ = SimpleDataset(x_[train_idx,:], label_[train_idx], transform=None)
    valid_ = SimpleDataset(x_[valid_idx,:], label_[valid_idx], transform=None)

    # valid_ = myDataset_nofake_dic(valid,transform=False)
    
    # initiate parameters
    # encoder-decoder setup
    predictionNN = EncoderDecoderNN(input_size=ncov, z_dim=z_dim, ncat=ncat,h_dim=[512,512])
    
    majority_label = list(set(range(ncat))-set(args.minority_label))
    args.majority_label = majority_label
    major_num = ncat - len(args.minority_label)

    gcl = GeneralizedContrastiveICAModel(dim = z_dim, n_label=ncat, n_steps=4, hidden_size=128, n_hidden=2, major_num=major_num)
    encoder = predictionNN.encoder

    decoder = DecoderNN(input_size=z_dim, ncat=ncat, h_dim=[32,32])

    # model path
    flow_path = model_path + '/'+model_name+'_gcl.pt'
    enc_path = model_path + '/'+model_name+'_enc.pt'
    dec_path = model_path + '/'+model_name+'_dec.pt'
    
    if training:
        
        train_loader = DataLoader(train_, batch_size= args.batch_size, num_workers=args.workers, shuffle= True)
        valid_loader = DataLoader(valid_, batch_size= args.batch_size, num_workers=args.workers, shuffle= True, drop_last = True)
        class_weight = torch.Tensor(np.ones(ncat))

        del train_, val_
        
        opt_enc = optim.Adam(predictionNN.parameters(), lr=args.enc_lr)
        opt_flow = optim.Adam(gcl.parameters(), lr=args.gcl_lr)
        opt_dec = optim.Adam(decoder.parameters(), lr=args.dec_lr)

        predictionNN.to(device)
        gcl.to(device)
        decoder.to(device)
        
        # identity map for toy dataset
        encoder = torch.nn.Identity()
        # pre-training the encoder
#         train_encoder(predictionNN, opt_enc, enc_path, train_loader, valid_loader, ncat, device)    
        # load pre-trained encoder
#         encoder.load_state_dict(torch.load(enc_path))

        # constractive learning of the invertible flow
        train_gcl(args.minority_label, args.majority_label, encoder, gcl, opt_flow, flow_path, train_loader, valid_loader, device, reg_weight= args.reg_weight, plot_path=plot_path, args=args)
        
        # load the best encoder
        encoder.load_state_dict(torch.load(enc_path))
        # load the best gcl
        gcl.load_state_dict(torch.load(flow_path))
        
        # load s from all minority classes
        minority_ = get_minority_s(train_loader, args.minority_label, encoder, gcl, device)
        
        # define minority class augmentation strength
        n_flag = np.zeros(ncov)
        n_flag[args.minority_label] = 1
        
        args.cls_weight = torch.tensor(np.array([1]*ncat)- (1-args.lmbda)*n_flag).float()
        args.cls_weight_aug = torch.tensor(np.array([1]*ncat)- (args.lmbda)*n_flag).float()
        args.cls_flag = torch.tensor(n_flag).float().to(device)
        
        balanced_loader = torch.utils.data.DataLoader(train_,batch_size=batch_size, \
                                                      sampler=ImbalancedDatasetSampler(train_, indices = list(range(train_.data.shape[0])),callback_get_label=callback_get_label))
        minority_ = get_minority_s(train_loader, args.minority_label, encoder.to(device), gcl.to(device), device)

        n_aug_per_class = 500
        s_aug, s_label = label_augmenting_multiple(minority_['s'], minority_['label'], label_list=args.minority_label, n_aug=[n_aug_per_class]*len(args.minority_label))

        aug_ds = SimpleDataset(s_aug.cpu(), s_label)
#         del s_aug, s_label
        aug_loader = DataLoader(aug_ds, batch_size= args.batch_size, num_workers=args.workers,shuffle= True)

#         del aug_ds, minority_

        # train the predictor
        train_decoder(encoder, gcl, decoder, opt_dec, dec_path, balanced_loader, aug_loader, valid_loader, device, args)

        
    #############################################    
    # Report performance on the validation dataset
    # load the best encoder
    encoder.load_state_dict(torch.load(enc_path))
    # load the best gcl
    gcl.load_state_dict(torch.load(flow_path))            
    # load the best decoder
    decoder.load_state_dict(torch.load(dec_path))

    encoder.eval()
    gcl.eval()
    decoder.eval()
    pred_risk = []
    true_label = []

    for batched_sample in valid_loader:
    #     batched_x = batched_sample['x']
    #     batched_label = batched_sample['label']
        batched_x, batched_label = batched_sample
        batched_x = batched_x.to(device).float().view(batched_x.shape[0], -1)
        batched_risk = decoder(gcl.hidden(encoder(batched_x)))
        pred_risk.append(batched_risk)
        true_label.append(batched_label)

    pred_risk = torch.cat(pred_risk)    
    true_label = torch.cat(true_label)    #         pred_label = get_predicted_label(pred_risk.detach())
    
    test_CE_loss, class_acc = cross_entropy(pred_risk, true_label, class_acc = True)
    
    # top1 error
    acc_top1 = np.mean((true_label.numpy()==np.argmax(pred_risk.detach(),axis=1).numpy())*1)
    # top 3 acc
    pred_risk = pred_risk.detach().numpy()
    val_top5 = [(pred_risk[i].argsort()[-3:][::-1]).tolist() for i in range(len(true_label))]
    true_y = true_label.numpy()

    acc_top5 = [true_y[i] in val_top5[i] for i in range(len(true_label))]
    acc_top5 = np.mean(acc_top5)        


    print('====> Test CE loss: {:.3f} \tper-class acc: {} \t'.format(test_CE_loss.item(), class_acc))
    print('====> Test top1 acc: {:.3f} \t top 3 acc: {:.3f} \t'.format(acc_top1, acc_top5))
    

def train_encoder(predictionNN, opt_enc, enc_path, train_loader, valid_loader, ncat, device, args):
    # def train_encoder():
    encoder = predictionNN.encoder
    best_valid_CE = np.inf

    best_epoch = 0
    nanFlag = 0
    epochs = 100

    # save training process
    #         model.train()
    for epoch in range(1, epochs + 1):
        if nanFlag == 1:
            break

        train_loss = 0
        valid_loss = 0

        improved_str = " "

        #
        for batch_idx, (batched_x, batched_label) in enumerate(train_loader):
            if nanFlag == 1:
                break
            predictionNN.train()
            if len(count_rates(batched_label))<ncat:
                break
            # pretrain with auto-encoder
    #         train_predictor_pretrain(batched_sample['x'], predictionNN, None)
            # pretrain with encoder-decoder
            train_predictor_pretrain(batched_x.view(batched_x.shape[0], -1), label=batched_label,\
                                     trainable = True, predictionNN=predictionNN, opt_enc=opt_enc)

        predictionNN.eval()

        for i, batched_sample in enumerate(valid_loader):
            batched_x, batched_label = batched_sample#.to(device).float().squeeze()
            batched_x = batched_x.to(device).float().view(batched_x.shape[0], -1)
            batched_z =  predictionNN.encoder(batched_x)
            pred = predictionNN(batched_x)

            valid_loss_, acc_ =  train_predictor_pretrain(batched_x, label=batched_label,\
                                     trainable = False, predictionNN=predictionNN, opt_enc=opt_enc)

            break

        save_model = 0
        if best_valid_CE > valid_loss_:
            best_epoch = epoch
            best_valid_CE = valid_loss_
            torch.save(encoder.state_dict(), enc_path)

            improved_str = "*"

        print('====> Valid CE loss: {:.4f} \tImproved: {}'.format(valid_loss_.item(), improved_str))

        if epoch - best_epoch >=10:
            print('Model stopped due to early stopping')
            break


def train_gcl(minority_label, majority_label, encoder, gcl, opt_flow, flow_path, train_loader, valid_loader, device, reg_weight = 1e-2, plot_path = '', model_name='', args = None):
    # Pretrain ICA
    best_valid_dist = np.inf
    best_valid_corr = 1.0
    best_epoch = 0
    nanFlag = 0
    epochs = 100

    # save training process
    #         model.train()
    for epoch in range(1, epochs + 1):

        train_loss = 0
        valid_loss = 0

        improved_str = " "

        for batch_idx, (batched_x, batched_label) in enumerate(train_loader):

            gcl.train()
            encoder.eval()
            batched_x = batched_x.to(device).view(batched_x.shape[0], -1)
            batched_label =  batched_label.to(device).squeeze()

            critic_loss = train_critic(gcl, encoder, opt_flow, batched_x, batched_label,\
                                       reg_weight=reg_weight, minority_label=minority_label,\
                                       majority_label = majority_label, trainable=True)        

            # checking if gcl has correct inversion
            try:
                with torch.no_grad():
                    batched_z = encoder(batched_x)
                    inv_batched_z = gcl.inv(gcl.hidden(batched_z))
                    
                assert ((inv_batched_z -  batched_z < 1e-1).detach().cpu().numpy()).all()

            except AssertionError:
                print('GCL collapsed, need to retrain the network')
                gcl.apply(weight_reset)

        gcl.eval()
        encoder.eval()
        
        valid_loss = 0
        valid_corr = 0
        for i, batched_sample in enumerate(valid_loader):

            batched_x, batched_label = batched_sample#.to(device).float().squeeze()
            batched_x = batched_x.to(device).float().view(batched_x.shape[0], -1)

            valid_loss_, corr_ = train_critic(gcl, encoder, opt_flow, batched_x, batched_label,\
                                             reg_weight=reg_weight, minority_label=minority_label,\
                                             majority_label = majority_label, trainable=False)

            valid_loss += valid_loss_
            corr += corr_


        save_model = 0
        if corr_ < best_valid_corr:
            best_epoch = epoch
            best_valid_corr = corr_
            torch.save(gcl.state_dict(), flow_path)
            encoder.eval()
            gcl.eval()
            batched_z = encoder(batched_x)
            batched_s = gcl.hidden(batched_z)
            visualize_s_z_space(batched_z, batched_s, torch.tensor(batched_label), torch.tensor(batched_label), ['z','s'], True, model_name, plot_path)

            improved_str = "*"

        print('====> Valid loss: {:.4f}, max corr: {:.3f}, \tImproved: {}'.format(valid_loss/i, corr_/i, improved_str))
        #print('====> Valid CE loss: {:.4f} \tper-class acc: {}'.format(valid_CE_loss.item(), class_acc))


        if epoch - best_epoch >=20:
            print('Model stopped due to early stopping')
            break
            
def train_decoder(encoder, gcl, decoder, opt_dec, dec_path, train_loader, aug_loader, valid_loader, device, args):
    # Augment data and train encoder/decoder
    best_valid_acc = 0
    best_valid_acc_minority = 0
    best_valid_CE = np.inf
    best_epoch = 0

    for epoch in range(1, epochs + 1):

        train_loss = 0
        valid_loss = 0

        print('epoch'+str(epoch))
        improved_str = " "
        for i, ((batched_x, batched_label), (s_aug,s_label)) in enumerate(zip(train_loader, aug_loader)):
            gcl.eval()
            encoder.eval()
            decoder.train()

            # encode x to z first

            batched_label =  batched_label.to(device).squeeze()
            # count minority class sample size

            batched_s = all_s(batched_x.to(device), encoder, gcl)

            train_loss = train_predictor_s_aug(s = batched_s.to(device), label = batched_label, \
                                               s_aug = s_aug.to(device), label_aug = s_label,\
                                               class_weight = args.cls_weight.to(device),\
                                               class_weight_aug = args.cls_weight_aug.to(device),\
                                               decoder = decoder, opt_dec = opt_dec, device=device)

        gcl.eval()
        encoder.eval()
        decoder.eval()
        valid_CE_loss_ = 0
        valid_acc_ = 0
        for i, (batched_x, batched_label) in enumerate(valid_loader):
            batched_x = batched_x.to(device).float().view(batched_x.shape[0], -1)

            pred_risk, valid_CE_loss, valid_acc = eval_predictor_x(batched_x, batched_label.to(device).squeeze(),\
                                                                   gcl=gcl, encoder=encoder, decoder=decoder)
            valid_CE_loss_ += valid_CE_loss
            valid_acc_ += valid_acc

            break
            
        valid_CE_loss_ = valid_CE_loss_/(i+1)
        valid_acc_ = valid_acc_/(i+1)

        save_model = 0
        if (best_valid_CE > valid_CE_loss):
            save_model += 1
        if (best_valid_acc < valid_acc_):
            save_model += 2

        if save_model >1:
            best_epoch = epoch
            best_valid_acc = valid_acc_
            best_valid_CE = valid_CE_loss
#             torch.save(gcl.state_dict(), flow_path)
            torch.save(decoder.state_dict(), dec_path)

            improved_str = "*"
        print('====> CRT Valid CE loss: {:.4f} \t mean acc: {} \tImproved: {}'.format(valid_CE_loss.item(), valid_acc_, improved_str))

        if epoch - best_epoch >=30:
            print('Model stopped due to early stopping')
            print('====> CRT Final Valid CE loss: {:.4f} \t mean acc: {} \tImproved: {}'.format(best_valid_CE.item(), best_valid_acc, improved_str))
            break
        
def CRT_result(encoder, gcl, decoder, flow_path, dec_path, valid_loader, device):
    # load the best encoder
#     encoder = torch.nn.Identity()
    # load the best gcl
    gcl.load_state_dict(torch.load(flow_path))
    # load the best decoder
    decoder.load_state_dict(torch.load(dec_path))

    encoder.eval()
    gcl.eval()
    decoder.eval()
    
    pred_risk = []
    true_label = []

    for batched_x, batched_label in valid_loader:
        batched_x = batched_x.to(device).float()
        batched_risk = decoder(gcl.hidden(encoder(batched_x)))
        pred_risk.append(batched_risk)
        true_label.append(batched_label)

    pred_risk = torch.cat(pred_risk)    
    true_label = torch.cat(true_label)
 
    loss_, acc_ = cross_entropy(pred_risk, true_label, class_acc=True)

    # top1 error
    acc_top1 = acc_
    # top 3 acc
    pred_risk = pred_risk.detach().cpu().numpy()
    val_top5 = [(pred_risk[i].argsort()[-5:][::-1]).tolist() for i in range(len(true_label))]
    true_y = true_label.numpy()

    acc_top5 = [true_y[i] in val_top5[i] for i in range(len(true_label))]
    acc_top5 = np.mean(acc_top5)        
    

    print('====> CRT CE loss: {:.3f} \t'.format(loss_.item()))
    print('====> CRT top1 acc: {:.3f} \t top 5 acc: {:.3f} \t'.format(acc_top1, acc_top5))
    
    return loss_.item(), acc_top1, acc_top5


if __name__ == '__main__':
    main()