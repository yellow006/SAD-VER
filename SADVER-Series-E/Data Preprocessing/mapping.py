"""
transfer SEED de_LDS features(62, times, 5) into a correct shape(sample_num, 5, 9, 9)
"""

import numpy as np
import scipy.io
import pandas as pd
import os

dataset_dir = './ExtractedFeatures/' # or paste your own data path here.

label = scipy.io.loadmat(dataset_dir + 'label.mat')
label = label['label']
label = label.flatten()

maplocation = pd.read_excel('./channel-order mapping.xlsx')   
ml_data = maplocation.values
mapping_location_row = ml_data[:,1]
mapping_location_column = ml_data[:,2]

def mapping(data_de):
    # projection
    mapping_matrix_all_freqs = []
    for freq in range(data_de.shape[2]):
        mapping_matrix_all_secs = []
        
        for secs in range(data_de.shape[1]):
            mapping_matrix_temp = np.zeros(81).reshape(9,9)
            
            for channel in range(data_de.shape[0]):
                mapping_matrix_temp[mapping_location_row[channel], mapping_location_column[channel]] = data_de[:,:,freq][:,secs][channel]
            
            full_channel_matrix = np.expand_dims(mapping_matrix_temp, axis=0)
            mapping_matrix_all_secs.append(full_channel_matrix)
        
        mapping_matrix_all_secs_temp1 = np.concatenate(mapping_matrix_all_secs)
        mapping_matrix_all_secs_temp2 = np.expand_dims(mapping_matrix_all_secs_temp1, axis=0)
        mapping_matrix_all_freqs.append(mapping_matrix_all_secs_temp2)
    
    mapping_matrix_all_channels_temp = np.concatenate(mapping_matrix_all_freqs)
    mapping_matrix_all_channels = np.transpose(mapping_matrix_all_channels_temp, (1,0,2,3))

    print('after transform, de_LDS shape: ',mapping_matrix_all_channels.shape)

    return mapping_matrix_all_channels

def feature_and_label_extraction(data):
    all_features = []
    all_labels = []

    for index in range(15):
        print('feature de_LDS%d'%(index+1))
        data_de = data['de_LDS%d'%(index+1)]
        print(data_de.shape)

        mapping_matrix = mapping(data_de)
        label_for_mapping_matrix = np.zeros(mapping_matrix.shape[0]) + label[index] + 1
        # print('label: ',label_for_mapping_matrix)
        print('label shape: ',label_for_mapping_matrix.shape)
        all_features.append(mapping_matrix)
        all_labels.append(label_for_mapping_matrix)
    
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_features, all_labels

def map():
    for subject in range(15):
        all_features = []
        all_labels = []

        for session in range(3):
            if (subject < 9):
                data = scipy.io.loadmat(dataset_dir + 'B0%d0%d.mat' % (subject + 1, session + 1)) 
                all_f, all_l = feature_and_label_extraction(data)
            else:
                data = scipy.io.loadmat(dataset_dir + 'B%d0%d.mat' % (subject + 1, session + 1)) 
                all_f, all_l = feature_and_label_extraction(data)
            all_features.append(all_f)
            all_labels.append(all_l)
        
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        for i in range(3):
            idx = np.where(all_labels == i)
            features = all_features[idx]
            print('current label: ', i)
            print('features shape: ', features.shape)

            d = './data_mapped_new/S%dL%d'%(subject+1, i)
            os.makedirs(d, exist_ok=True)
            np.save(d + '/data.npy', features)

def mask():
    data = np.load('./data_mapped_new/S1L0/data.npy')
    int_data = data[0][0].astype(int)
    mask = np.zeros(81).reshape(9,9)

    for i in range(9):
        for j in range(9):
            if int_data[i][j] != 0:
                mask[i][j] = 1
    print(int_data)
    print(mask)
    np.save('./SEEDmask.npy', mask)

if __name__ == '__main__':
    map()
    # mask()
