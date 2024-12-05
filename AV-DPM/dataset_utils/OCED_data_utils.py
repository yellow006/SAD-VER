import numpy as np
import torch
import os
from einops import rearrange

def set_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_OCED_data(args, config, category_num=6):
    assert category_num in (6, 72), 'category_num error in OCED datasets'

    sub_idx = args.subject_idx
    category_idx =args.category_idx
    path = config.data.data_path

    # load data & labels
    data_arr = np.load(path + 'S%d/data.npy'%(sub_idx))
    if category_num == 6:
        label_arr = np.load(path + 'S%d/label_6.npy'%(sub_idx), allow_pickle=True).flatten()
    else:
        label_arr = np.load(path + 'S%d/label_72.npy'%(sub_idx), allow_pickle=True).flatten()

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

    # for i in range(len(data_all)):
    #     assert data_all[i].shape[0] == label_all[i].shape[0], 'label & data match error'
    #     assert data_all[i].shape[0] in (72,73,74,864,865,866), 'shape error'
    
    data = np.concatenate((data_all), axis=0)
    label = np.concatenate((label_all), axis=0)

    # acquire EEG data for current label
    cat_idx = np.where(label==category_idx)
    data = data[cat_idx]
    label = label[cat_idx]
    # assert data.shape[0] == label.shape[0]

    # Min-Max Normalize
    datamax, datamin = data.max(), data.min()
    data = (data - datamin) / (datamax - datamin)

    # save min-max value for inverse normalize
    minmaxpath = ('./inverse_normalization/S%d/label%d'%(sub_idx, category_idx))
    os.makedirs(minmaxpath, exist_ok=True)
    np.save(minmaxpath + '/max.npy', datamax)
    np.save(minmaxpath + '/min.npy', datamin)

    data = torch.tensor(data, dtype=torch.float)
    label = torch.tensor(label, dtype=torch.int64)

    print('OCED data from subject: %d, label: %d is loaded. current data shape: '%(sub_idx, category_idx), data.shape)

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