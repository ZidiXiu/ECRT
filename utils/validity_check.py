import torch
import numpy as np



def load_model(model, model_path):
    model.eval()
    orig_data = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            orig_data.append(param.data.clone())
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    loaded_data = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            loaded_data.append(param.data.clone())
    
    while not torch.allclose(orig_data[0], loaded_data[0]):
        model.eval()
        orig_data = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                orig_data.append(param.data.clone())

        model.load_state_dict(torch.load(model_path))
        model.eval()
        loaded_data = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                loaded_data.append(param.data.clone())
