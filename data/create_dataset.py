import numpy as np
import torch
from torch.utils.data import Dataset

from networks.MAF_flow_likelihood_debugging import MAF

def MAF_model():
    n_blocks = 4
    hidden_size = 32
    n_hidden = 2
    activation_fn = 'relu'
    input_order = 'sequential'
    conditional = False

    # input_size = train_loader.dataset.input_size
    input_dims = 2
    input_size = input_dims
    cond_label_size = None
    # cond_label_size = 1

    trans_flow = MAF(n_blocks, input_size, hidden_size, n_hidden, cond_label_size, activation_fn, \
                input_order, batch_norm=True)
    return trans_flow
    
def MAF_trans(s_, trans_flow_path):
    model = MAF_model()
    model.load_state_dict(torch.load(trans_flow_path))
    model.eval()
    
    x_, reg_loss = model.inverse(torch.Tensor(s_).float())
    
    return x_.detach().cpu().clone().numpy()

def simulated_data_ablation(mu_list= [[0,-1], [2,1], [5,2], [1,3], [-2,1], [-3.5,4],[-4,-1]],
                   sig_list=[[0.5,0.5], [3,1], [0.3,2.], [3,1], [1, 0.2], [1,1], [2, 0.3]],
                   n_sample_list=[2000,2000,2000,2000, 2000, 2000, 2000], seed=123):
    np.random.seed(seed)
    m = len(mu_list)
    s_list = []
    label_list = []
    for i in range(m):
        n_sample = n_sample_list[i]
#         print(i, n_sample)
        s_list.append(np.random.multivariate_normal(
            mu_list[i],np.diag(sig_list[i]),(n_sample)))
        label_list.append(np.repeat(i, n_sample))
#     label_ = np.repeat(np.array(range(m)),n_sample)
    label_ = np.concatenate(label_list)
    s_ = np.concatenate(s_list)
    if name == 'henon':
    def henon_map(X):
        a1 = 1-1.4*X[:,0]**2+X[:,1]
        a2 = .3*X[:,0]
        a = np.vstack((a1, a2))
        return a.T

    return henon_map(s_), label_

def simulated_data(mu_list= [[0,-1], [2,1], [4,2], [1,3]],
                   sig_list=[[0.5,0.5], [3,1], [0.3,2.], [1,1]],
                   n_sample_list=[5000,5000,5000,5000], name='rotate_scale'):
    m = len(mu_list)
    s_list = []
    label_list = []
    for i in range(m):
        n_sample = n_sample_list[i]
        
        s_list.append(np.random.multivariate_normal(
            mu_list[i],np.diag(sig_list[i]),(n_sample)))
        label_list.append(np.repeat(i, n_sample))
#     label_ = np.repeat(np.array(range(m)),n_sample)
    label_ = np.concatenate(label_list)
    s_ = np.concatenate(s_list)
    
    if name =='rotate_scale':
        def rotate(X, theta):
            c = np.cos(theta)
            s = np.sin(theta)
            A = np.array([[c,s],[-s,c]])
            return np.dot(X, A)

        def scale(X, s):
            for i in range(len(s)):
                X[:,i] = s[i]*X[:,i]        
            return X        
        
        z_ = rotate(s_, 30./180*np.pi)
        z_ = scale(z_, [-1,1.5])
        z_ = rotate(z_, 45./180*np.pi)
        z_ = scale(z_, [1.5,.75])
        z_ = rotate(z_, 45./180*np.pi)
        return s_, label_
    
    if name == 'henon':
        def henon_map(X):
            a1 = 1-1.4*X[:,0]**2+X[:,1]
            a2 = .3*X[:,0]
            a = np.vstack((a1, a2))
            return a.T
        return s_, label_

# normalizing data
def normalizing_data(z_):
    for j in range(z_.shape[1]):
        z_[:,j] = (z_[:,j]-np.mean(z_[:,j]))/np.std(z_[:,j])
    return z_

# inherit torch.utils.data.Dataset for custom dataset
class myDataset_dic(Dataset):
    """normalizing continuous input"""

    def __init__(self, data_dictionary, transform=None, norm_mean = 0.0, norm_std = 1.0, continuous_variables=[]):
        """
        Args:
            csv_file (string): Path to the csv file of datasets.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dictionary = data_dictionary
        self.transform = transform
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __len__(self):
        return len(self.data_dictionary['s'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        id_e = self.data_dictionary['label'][idx]
        id_f = self.data_dictionary['fake_label'][idx]
        id_s = self.data_dictionary['s'][idx,:]
        id_s = np.array(id_s)
        id_x = self.data_dictionary['x'][idx,:]
        id_x = np.array(id_x)        
        # sample = {'e': np.array([id_e]), 'x': id_x}

        if self.transform:
            pass
            # sample = self.transform(sample)
#             id_[continuous_variables] = (id_x.copy()[continuous_variables] - norm_mean) / norm_std

        return {"label":id_e, "fake_label":id_f, "s": id_s, 'x':id_x}


# inherit torch.utils.data.Dataset for custom dataset
class myDataset_nofake_dic(Dataset):
    """normalizing continuous input"""

    def __init__(self, data_dictionary, transform=None, norm_mean = 0.0, norm_std = 1.0, continuous_variables=[]):
        """
        Args:
            csv_file (string): Path to the csv file of datasets.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dictionary = data_dictionary
        self.transform = transform
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __len__(self):
        return len(self.data_dictionary['s'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        id_e = self.data_dictionary['label'][idx]
        id_s = self.data_dictionary['s'][idx,:]
        id_s = np.array(id_s)
        id_x = self.data_dictionary['x'][idx,:]
        id_x = np.array(id_x)        
        # sample = {'e': np.array([id_e]), 'x': id_x}

        if self.transform:
            pass
            # sample = self.transform(sample)
#             id_[continuous_variables] = (id_x.copy()[continuous_variables] - norm_mean) / norm_std

        return {"label":id_e, "s": id_s, 'x':id_x}