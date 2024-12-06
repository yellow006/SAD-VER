import pandas as pd
import numpy as np

"""
Transfer your data shape from 13x13 2D grid to original 124 electrodes
"""

for sub in range(1,11):
    for idx in range(1,7):
        data_1 = np.load('paste your 13x13 data path here, seperate by subjects & labels/S%d/label_%d.npy'%(sub, idx))

        mask = np.load("./mask.npy")
        data_1 *= mask
        np.save('paste your 13x13 data path here, seperate by subjects & labels/S%d/label_%d.npy'%(sub, idx), data_1)
        print(data_1.shape)
        print(mask.shape)
        location = pd.read_excel('./XY_location.xlsx', header=None)
        map_loc = location.values
        # print(map_loc[1][2])

        map_loc_list = []
        for row in range(1, 125):
            X = map_loc[row][1]
            Y = map_loc[row][2]
            XY = np.array((X, Y), dtype=np.int16)
            map_loc_list.append(XY)

        map_loc_list = np.array(map_loc_list)
        # print(map_loc_list)
        data_124 = []
        for _, loc in enumerate(map_loc_list):
            temp = data_1[:,:,loc[0], loc[1]]
            data_124.append(temp)

        data_124 = np.array(data_124).transpose(1,2,0)
        print(data_124.shape)
        np.save('paste your 124 electrodes data path here/S%d/label_%d.npy'%(sub, idx), data_124)