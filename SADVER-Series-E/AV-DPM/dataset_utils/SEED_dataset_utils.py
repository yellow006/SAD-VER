import numpy as np
import torch
import os
from einops import rearrange

def set_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_SEED_data(args, config):
    sub_idx = args.subject_idx
    category_idx =args.category_idx
    path = config.data.data_path

    data = np.load(path + 'S%dL%d/data.npy'%(sub_idx, category_idx))

    datamax, datamin = data.max(), data.min()
    data = (data - datamin) / (datamax - datamin)

    # save min-max value for inverse normalize
    minmaxpath = ('./inverse_normalization/S%dL%d'%(sub_idx, category_idx))
    os.makedirs(minmaxpath, exist_ok=True)
    np.save(minmaxpath + '/max.npy', datamax)
    np.save(minmaxpath + '/min.npy', datamin)

    data = torch.tensor(data, dtype=torch.float)
    label = torch.zeros(data.shape[0], dtype=torch.int64) + category_idx
    print('SEED data from subject: %d, label: %d is loaded. current data shape: '%(sub_idx, category_idx), data.shape)
    
    return data, label


def inverse_normalization(args, X):
    sub_idx = args.subject_idx
    category_idx =args.category_idx

    minmaxpath = ('./inverse_normalization/S%d/label%d'%(sub_idx, category_idx))
    X_min = torch.tensor(np.load(minmaxpath + '/min.npy'), dtype=torch.float)
    X_max = torch.tensor(np.load(minmaxpath + '/max.npy'), dtype=torch.float)
    X_out = X * (X_max - X_min) + X_min

    return X_out

    
# load_OCED_data()