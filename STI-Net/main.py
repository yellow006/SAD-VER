import numpy as np
import torch
import datetime

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.model_selection import KFold

from train_test_utils import Trainer
from standford_data_utils import set_seeds, load_OCED_data
# from SEED_data_utils import load_SEED_data

def main(sub: int, network_type: str, use_aug: bool):
    # time_start = datetime.datetime.now()
    print('Launching 10-fold cross validation.')
    set_seeds()
    print('Current Subject: %d'%(sub))
    # hyper parameters
    ks = 10
    batch_size = 128
    epoches = 50

    data, label = load_OCED_data(sub_idx=sub, category_num=6, load_gen_data=False, network_type=network_type)
    category_num = int(label.max() + 1)

    print(data.shape)

    # temporary variables
    best_validate_acc = 0
    worst_validate_acc = 0
    acc_total = 0
    acc_array = []

    kfold = KFold(n_splits=ks, shuffle=True)
    ktimes = 1

    for train_idx, test_idx in kfold.split(data):
        print('the times of K: ', ktimes)

        # split data & labels by k-fold cross validation.
        train_arr = data[train_idx]
        train_label = label[train_idx]

        # use data augmentation
        if use_aug:
            data_gen, label_gen = load_OCED_data(sub_idx=sub, category_num=6, load_gen_data=True, network_type=network_type)
            train_arr = np.concatenate((train_arr, data_gen), axis=0)
            train_label = np.concatenate((train_label, label_gen), axis=0)

        print('train arr shape: ', train_arr.shape)

        test_arr = data[test_idx]
        test_label = label[test_idx]

        # standardize for traindata & testdata
        train_mean = train_arr.mean(axis=0)
        train_std = train_arr.std(axis=0)
        train_arr_std = (train_arr - train_mean) / train_std
        train_arr_std = np.nan_to_num(train_arr_std)

        test_arr_std = (test_arr - train_mean) / train_std
        test_arr_std = np.nan_to_num(test_arr_std)

        train_tensor = torch.tensor(train_arr_std, dtype=torch.float)
        train_label = torch.tensor(train_label, dtype=torch.int64)
        test_tensor = torch.tensor(test_arr_std, dtype=torch.float)
        test_label = torch.tensor(test_label, dtype=torch.int64)

        train_dataset = TensorDataset(train_tensor, train_label)
        test_dataset = TensorDataset(test_tensor, test_label)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # Build Trainer
        trainer = Trainer(train_loader=train_loader,  
                          test_loader=test_loader,
                          category_num=category_num,
                          initial_lr=2e-4,  # 1e-3
                          use_dynamic_lr=False,
                          lr_decay_rate=0.01,
                          network_type=network_type)
        
        # start training 
        test_acc = trainer.train_val(epoches)

        # show test results
        if ((test_acc <= worst_validate_acc) or (ktimes == 1)):
            worst_validate_acc = test_acc
        if (test_acc > best_validate_acc):
            best_validate_acc = test_acc
        
        acc_array.append(test_acc.cpu().numpy())
        acc_total += test_acc.cpu().numpy()
        print('current validate average acc = {:.3f}%, best acc = {:.3f}% , worst acc = {:.3f}%'
              .format(acc_total / ktimes * 100, best_validate_acc * 100, worst_validate_acc * 100))
        
        # one fold train & test finished.
        ktimes += 1

    # write results to txt.
    acc_array = np.array(acc_array)
    with open('./val_result/result_S%d.txt'%(sub),'w') as log_write:
        for i in range(len(acc_array)):
            log_write.write('k = %d, validate acc = %5f\n'%(i+1, acc_array[i]))
        log_write.write('validate average acc = {:.3f}%, best acc = {:.3f}% , worst acc = {:.3f}%\n'
              .format(acc_array.mean()*100, acc_array.max()*100, acc_array.min()*100))
        log_write.write('std: %7f'%(acc_array.std()))


if __name__ == '__main__':
    for sub in range(1,11):
        main(sub, "STINet", False)
