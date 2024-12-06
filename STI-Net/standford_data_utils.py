import numpy as np
import torch
from einops import rearrange

def set_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_OCED_data(sub_idx: int, 
                   category_num: int = 6, 
                   load_gen_data: bool = False, 
                   network_type: str = "STINet"):
    """
    params:
        sub_idx (int): current subject for training & validating
        category_num (int): number of label type. default = 6.
        load_gen_data (bool): load original data & generated data.
        use_124 (bool): use data with 124 electrode format & 13 x 13 format. 

        Before using this function, you should specify the location 
        where the dataset is stored by yourself, and organize it according to 
        whether it is for generating EEG, and whether it uses 124 channels or 13x13 mapping.

        Different networks have different requirements for data shape. 
        Check the list below if error occurs.
        (samples, 32, 13, 13) when using STI-Net.
        (samples, 1, 124, 32) when using EEGNet, Conformer, FBCNet.
        (samples, 124, 32) when using EEGLSTM, DGCNN.
    """
    use_124 = False
    expand_dims = False
    if network_type != "STINet": use_124 = True
    if network_type == "EEGNet" or network_type == "Conformer" or network_type == "FBCNet": expand_dims = True

    if load_gen_data == False:
        if use_124:
            path='./EEG_data/original_data/124/'
        else:
            path='./EEG_data/original_data/13_13/'
        data_arr = np.load(path + 'S%d/data.npy'%(sub_idx))
        label_arr = np.load(path + 'S%d/label_6.npy'%(sub_idx), allow_pickle=True).flatten()

    else:
        if use_124:
            path = './EEG_data/gen_data/124/'
        else:
            path = './EEG_data/gen_data/13_13/'
        data_arr = []
        label_arr = []
        for label in range(1,7):
            tmp = np.load(path + 'S%d/label_%d.npy'%(sub_idx, label))
            data_arr.append(tmp)
            lb = np.zeros(tmp.shape[0]) + label
            label_arr.append(lb)
        data_arr = np.concatenate(data_arr, axis=0)
        label_arr = np.concatenate(label_arr, axis=0)

    # rearrange data & labels by index
    data_all = []
    label_all = []

    for label_idx in range(1, category_num+1):
        data_in_single_label = []
        label_single = []

        for idx in range(data_arr.shape[0]):
            if label_arr[idx] == label_idx:
                data_in_single_label.append((data_arr[idx,:,:]))
                label_single.append((label_arr[idx]))

        data_in_single_label = np.array(data_in_single_label)
        label_single = np.array(label_single)

        data_all.append(data_in_single_label)
        label_all.append(label_single)

    data = np.concatenate((data_all), axis=0)
    label = np.concatenate((label_all), axis=0)
    
    label -= 1
    
    if expand_dims:
        data = np.expand_dims(data, axis=1)

    return data, label