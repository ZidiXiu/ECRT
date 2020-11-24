from torch import nn
from utils.trainer_util import cross_entropy


def train_predictor_pretrain(x, label=None, trainable = True, predictionNN=None, opt_enc=None):
    if trainable:
        predictionNN.train()
        opt_enc.zero_grad()
        pred = predictionNN(x)
        if label is None:
            '''auto-encoder set up'''
            label = x
            criterion = nn.MSELoss()
            loss_= criterion(pred, label)        
        else:
            '''encoder-decoder set up'''
            loss_= cross_entropy(pred, label)
            
        loss_.backward()
        nn.utils.clip_grad_norm_(predictionNN.parameters(), 1e-3)
        
        opt_enc.step()
    else:
        predictionNN.eval()
        pred = predictionNN(x)
        if label is None:
            '''auto-encoder set up'''
            label = x
            criterion = nn.MSELoss()
            loss_= criterion(pred, label)
            return loss_
        else:
            '''encoder-decoder set up'''
            loss_, acc_ = cross_entropy(pred, label, class_acc=True)
            return loss_, acc_
        
    return loss_
