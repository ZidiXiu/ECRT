"""
Generalized Contrastive Learning ICA with infoNCE loss
"""
import torch
import torch.nn as nn
import numpy as np

# Type hinting
from typing import Union, List, Optional, Any, Tuple
from torch import FloatTensor, LongTensor
from networks.MAF_flow_multiple import MAF
from torch.nn import functional as F


class ComponentwiseTransform(nn.Module):
    """A neural network module to represent trainable dimension-wise transformation."""
    def __init__(self, modules: Union[List[nn.Module], nn.ModuleList]):
        """
        Parameters:
            modules: list of neural networks each of which is a univariate function
                     (or additionally it can take an auxiliary input variable).
        """
        super().__init__()
        if isinstance(modules, list):
            self.module_list = nn.ModuleList(modules)
        else:
            self.module_list = modules

    def __call__(self, x: FloatTensor, aux: Optional[Any] = None):
        """Perform forward computation.
        Parameters:
            x: input tensor.
            aux: auxiliary variable.
        """
        if aux is not None:
            return torch.cat(tuple(self.module_list[d](x[:, (d, )], aux)
                                   for d in range(x.shape[1])),
                             dim=1)
        else:
            return torch.cat(tuple(self.module_list[d](x[:, (d, )])
                                   for d in range(x.shape[1])),
                             dim=1)



class ComponentWiseTransformWithAuxSelection(ComponentwiseTransform):
    """A special type of ``ComponentWiseTransform``
    that takes discrete auxiliary variables and
    uses it as labels to select the output value out of an output vector.
    """
    def __init__(self,
                 x_dim: int,
                 out_dim: int,
                 hidden_dim: int = 512,
                 n_layer: int = 1):
        """
        Parameters:
            x_dim: input variable dimensionality.
            out_dim: the embedding dimension of the label
            hidden_dim: the number of the hidden units for each hidden layer (fixed across all hidden layers).
            n_layer: the number of layers to stack.
        """
        super().__init__([
            nn.Sequential(
                nn.Linear(1, hidden_dim), nn.ReLU(),
                *([nn.Linear(hidden_dim, hidden_dim),
                   nn.ReLU()] * n_layer), nn.Linear(hidden_dim, out_dim))
            for _ in range(x_dim)
        ])

    def __call__(self, x: FloatTensor):
        """Perform the forward computation.
        Parameters:
            x: input vector.
            aux: auxiliary variables.
        """
        # by dimension output
        # [batch_size, n_embedding, x_dim]
        # sum dimensions
        outputs = torch.cat(tuple(self.module_list[d](x[:, (d, )]).unsqueeze(2)
                                  for d in range(x.shape[1])), dim=2).sum(2)
        
        # [batch_size, n_embedding]
        return outputs

class _LayerWithAux(nn.Module):
    """A utility wrapper class for a layer that passes the auxiliary variables."""
    def __init__(self, net: nn.Module):
        """
        Parameters:
            net: the neural network.
        """
        super().__init__()
        self.net = net

    def __call__(self, x: FloatTensor, aux: Optional[Any] = None):
        """Perform forward computation.
        Parameters:
            x: input tensor.
            aux: auxiliary variable.
        """
        return self.net(x), aux

class GeneralizedContrastiveICAModel(nn.Module):
    """Example implementation of the ICA (wrapper) model that can be trained via GCL.
    It takes a feature extractor to estimate the mixing function
    and adds a classification functionality to enable the training procedure of GCL.
    """
    def __init__(
            self,
#             network: nn.Module,
            dim: int,
            n_label: int,
            n_emb: int,
            n_steps: int,
            hidden_size: int,
            n_hidden: int,
            major_num: int,
            tau: Optional[float] = 1.,
            componentwise_transform: Optional[ComponentwiseTransform] = None,
            linear: nn.Module = None,
            input_order: str = 'sequential'):
        """
        Parameters:
            network: An invertible neural network (of PyTorch).
            linear: A linear layer to be placed at the beginning of the classification layer.
            dim: The dimension of the input (i.e., the dimension of the signal source).
        """
        super().__init__()
        self.network = MAF(n_steps, dim, hidden_size, n_hidden, major_num, None, 'relu', \
            input_order, batch_norm=True)

        if componentwise_transform is not None:
            self.componentwise_transform = componentwise_transform
        else:
            # by dimension networks
            self.componentwise_transform = ComponentWiseTransformWithAuxSelection(
                dim, n_emb)

        if linear is not None:
            self.linear = _LayerWithAux(linear)
            self.classification_net = nn.Sequential(
                self.linear, self.componentwise_transform)
        else:
            self.linear = None
            self.classification_net = self.componentwise_transform
        
        # define y linear embedding
#         self.y_emb_level = torch.nn.Parameter(F.softmax(torch.rand(n_label, n_emb),dim=-1))
        if n_emb == n_label:
            self.y_emb_level = torch.nn.Parameter(torch.eye(n_emb))
        else:
            idx = torch.randint(n_emb,(n_label,1)).squeeze()
            y_emb_level = torch.zeros(n_label, n_emb)
            y_emb_level[np.arange(n_label).tolist(),idx] = 1
            self.y_emb_level = torch.nn.Parameter(y_emb_level)
            
        # learnable temperature
        self.log_tau = torch.nn.Parameter(torch.Tensor([np.log(tau)]))
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
    def hidden(self, x: FloatTensor) -> FloatTensor:
        """Extract the hidden vector from the input data.
        Parameters:
            x: input tensor.
        """
        result, _ = self.network(x)
        return result

    def logpdf(self, x, major_label):
        """Extract the hidden vector from the input data.
        Parameters:
            x: input tensor.
            major_label: corresponding label in majority_label list
        """
        logpdf = self.network.log_prob(x, major_label)
        return logpdf


            
    def bilinearFDVNCE(self, data, tau = None,return_hidden = True):

        '''replacing Contrastive Loss in GCL

        features: sourse s=f(z)
        labels: auxillary variables (labels)

        temperature: hyper-parameter to tune
        '''
        if tau is None:
            tau = torch.exp(self.log_tau)
        tau = torch.sqrt(tau)    
        z, aux = data
        device = z.device
        hidden = self.hidden(z)
        # by dimension encoding
        # [batch_size, n_emb]
        hz = (self.norm(self.classification_net(hidden)))/tau
        # [batch_size, n_emb]
        hy = (self.norm(self.cat_interpolation(aux)))/tau
        
        # [batch_size, batch_size]
        similarity_matrix = hz @ hy.t()
        batch_dim = hz.size(0)
        
        del hz, hy, z, aux
        pos_mask = torch.eye(batch_dim,dtype=torch.bool)
        
        g = similarity_matrix[pos_mask].view(batch_dim,-1)
        g0 = similarity_matrix[~pos_mask].view(batch_dim,-1)
            
        del pos_mask
        logits = g0 - g
            
        slogits = torch.logsumexp(logits,1).view(-1,1)
            
        labels = torch.tensor(range(batch_dim),dtype=torch.int64).to(device)
        dummy_ce = self.criterion(similarity_matrix,labels) - torch.log(torch.Tensor([batch_dim]).to(device))
        
        del similarity_matrix
        dummy_ce = dummy_ce.view(-1,1)
            
        output = dummy_ce.detach()+torch.exp(slogits-slogits.detach())-1
        
        del dummy_ce
        output = torch.clamp(output,-5,15)
        
        if return_hidden:
            return output, hidden
        else:
            return output
        
    def MI(self, x, y, K=10):
        '''
        mutual information
        '''
        mi = 0
        for k in range(K):
            # randomly permutation the batch
            y0 = y[torch.randperm(y.size()[0])]
            mi += self.forward(x,y,y0)
            
        return -mi/K   
    
    def inv(self, x: FloatTensor) -> FloatTensor:
        """Perform inverse computation.
        Parameters:
            x: input data (FloatTensor).
        Returns:
            numpy array containing the input vectors.
        """
        result, _ = self.network.inverse(x)
        return result

    def forward(self, x: FloatTensor) -> FloatTensor:
        """Alias of ``hidden()``.
        Parameters:
            x: input data.
        """
        return self.hidden(x)

    def __call__(self, x: FloatTensor) -> FloatTensor:
        """Perform forward computation.
        Parameters:
            x: input data.
        """
        return self.forward(x)
    
    def norm(self,z):
        return torch.nn.functional.normalize(z,dim=-1)
    
    def cat_interpolation(self, y):
        return self.y_emb_level[y.long()]