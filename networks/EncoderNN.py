"""
Encoder and Decoder for MONICA

"""
import torch
import torch.nn as nn
import numpy as np

# Type hinting
from typing import Union, List, Optional, Any, Tuple
from torch import FloatTensor, LongTensor

class SimpleMLP(nn.Module):
    def __init__(self, input_size=2, output_size=2, h_dim=[32,32]):
        super(SimpleMLP, self).__init__()
        net = []
        hs = [input_size] + h_dim + [output_size]
        for h0, h1 in zip(hs, hs[1:]):
            net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        net.pop()  # pop the last ReLU for the output layer
        self.net = nn.Sequential(*net)
        
    def forward(self, x):
        z = self.net(x)
        return z
# torch.nn.init.xavier_uniform(linear1.weight)
    
class AutoEncoderNN(nn.Module):
    '''
    This network is to train encoder with an autoencoder
    '''
    def __init__(self, input_size=2, z_dim=2, h_dim=[32,32]):
        super(AutoEncoderNN, self).__init__()
        '''
        input_size: number of covariates in the dataset
        z_dim: hidden space dimension
        h_dim: network dimensions
        '''
        self.input_size = input_size
        self.z_dim = z_dim
        
        self.encoder = SimpleMLP(input_size, z_dim, h_dim)
        self.decoder = SimpleMLP(z_dim, input_size, h_dim)
        
    def forward(self, x):
        z = self.encoder(x)
        generate_x = self.decoder(z)
        return generate_x
    
class EncoderDecoderNN(nn.Module):
    '''
    This network is to train encoder with an decoder to the final category
    '''    
    def __init__(self, input_size=2, z_dim=2, ncat=4, h_dim=[32,32]):
        super(EncoderDecoderNN, self).__init__()
        '''
        input_size: number of covariates in the dataset
        z_dim: hidden space dimension
        ncat: number of unique outcome labels
        h_dim: network dimensions
        '''
        self.input_size = input_size
        self.z_dim = z_dim
        
        self.encoder = SimpleMLP(input_size, z_dim, h_dim)
        self.decoder = SimpleMLP(z_dim, ncat, h_dim)
        
    def forward(self, x):
        z = self.encoder(x)
        pred_risk = self.decoder(z)
        return pred_risk
    
class i_EncoderDecoderNN(nn.Module):
    '''
    This network is to train encoder with an decoder to the final category
    with identity encoder
    '''    
    def __init__(self, input_size=2, z_dim=2, ncat=4, h_dim=[32,32]):
        super(EncoderDecoderNN, self).__init__()
        '''
        input_size: number of covariates in the dataset
        z_dim: hidden space dimension
        ncat: number of unique outcome labels
        h_dim: network dimensions
        '''
        self.input_size = input_size
        self.z_dim = z_dim
        
        self.encoder = torch.nn.Identity()
        self.decoder = SimpleMLP(z_dim, ncat, h_dim)
        
    def forward(self, x):
        z = self.encoder(x)
        pred_risk = self.decoder(z)
        return pred_risk
    
class DecoderNN(nn.Module):
    '''
    This network is to define decoder from s to the final category
    '''    
    def __init__(self, input_size=2, ncat=4, h_dim=[32,32]):
        super(DecoderNN, self).__init__()
        '''
        input_size: number of covariates in the dataset
        z_dim: hidden space dimension
        ncat: number of unique outcome labels
        h_dim: network dimensions
        '''
        
        self.decoder = SimpleMLP(input_size, ncat, h_dim)
        
    def forward(self, z):
        pred_risk = self.decoder(z)
        return pred_risk