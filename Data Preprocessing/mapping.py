import scipy.io
import numpy as np
import math
import pandas as pd
import os

from einops import rearrange
from tqdm import tqdm
from scipy import signal

def get_eeg_data_unpreprocessed(o_path, file_idx):
    eeg_data_raw = scipy.io.loadmat('o_path' + '/S%d.mat'%(file_idx))['X_3D']
    eeg_data_raw = rearrange(eeg_data_raw, 'b t s -> s t b')
    print('\toriginal data shape after rearrange:  ', eeg_data_raw.shape)
    return eeg_data_raw


def mapping_unpreprocessed(data):
    map_location = pd.read_excel('./XY_location.xlsx')
    ml_data = map_location.values
    location_X = ml_data[:,1]
    location_Y = ml_data[:,2]

    mapping_matrix_all_samples = []
    for samples in range(data.shape[0]):

        mapping_matrix_all_secs = []
        for secs in range(data.shape[1]):

            mapping_matrix_temp = np.zeros(169).reshape(13,13)
            # mask = np.zeros(169).reshape(13,13)
            for channel in range(data.shape[2]):
                tempdata = data[samples,:,:][secs,:][channel]
                # flag = 1
                x = location_X[channel]
                y = location_Y[channel]
                mapping_matrix_temp[x, y] = tempdata
                # mask[x, y] = flag

            mapping_matrix_all_secs.append(mapping_matrix_temp)
            # np.save('mask.npy',mask)
            # print('stop!')

        mapping_matrix_all_secs = np.array(mapping_matrix_all_secs)
        mapping_matrix_all_samples.append(mapping_matrix_all_secs)

    mapping_matrix_all_samples = np.array(mapping_matrix_all_samples)
    print(mapping_matrix_all_samples.shape)
    return mapping_matrix_all_samples

def get_labels(o_path, file_idx):
    label_6_category = scipy.io.loadmat('o_ptah' + '/S%d.mat'%(file_idx))['categoryLabels']
    assert (np.max(label_6_category) == 6 and np.min(label_6_category) == 1)

    label_72_category = scipy.io.loadmat('o_path' + '/S%d.mat'%(file_idx))['exemplarLabels']
    assert (np.max(label_72_category) == 72 and np.min(label_72_category) == 1)

    return label_6_category, label_72_category


if __name__ == "__main__":
    originalpath = 'paste your path of original OCED dataset here'
    savepath = 'paste your path of saved dataset here'

    for file_idx in tqdm(range(1,11), leave=True, desc='preprocessing progress: ', ncols=80, colour='YELLOW'):
        print('\tsubject %d preprocessing started.'%(file_idx))
        feature_arr = get_eeg_data_unpreprocessed(originalpath, file_idx)
        data_preprocessed = mapping_unpreprocessed(feature_arr)
        label_6, label_72 = get_labels(originalpath, file_idx)

        if os.path.exists(savepath + 'S%d'%(file_idx)) == False:
            os.makedirs(savepath + 'S%d'%(file_idx))
        np.save(savepath + 'S%d/data.npy'%(file_idx), data_preprocessed)
        np.save(savepath + 'S%d/label_6.npy'%(file_idx), label_6)
        np.save(savepath + 'S%d/label_72.npy'%(file_idx), label_72)
        print('subject %d labels & data save completed.'%(file_idx))