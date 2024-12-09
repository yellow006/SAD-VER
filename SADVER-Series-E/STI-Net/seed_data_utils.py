import numpy as np
import torch
from einops import rearrange

def set_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_SEED_data(sub, load_gen_data=False):
    path = 'paste your own data path here'
    if load_gen_data:
        path += 'data_mapped_gen/'
    else:
        path += 'data_mapped_new/'

    data = []
    label = []
    for i in range(3):
        data_ = np.load(path + 'S%dL%d/data.npy'%(sub, i))
        label_ = np.zeros(data_.shape[0]) + i

        if load_gen_data:
            data_ *= np.load('./SEEDmask.npy')
        
        data.append(data_)
        label.append(label_)
    
    data = np.concatenate(data, axis=0)
    label = np.concatenate(label, axis=0)

    shuffled_indices = np.random.permutation(data.shape[0])
    data = data[shuffled_indices]
    label = label[shuffled_indices]

    np.expand_dims(data, axis=1)

    return data, label

# load_seed_data(1, 1)