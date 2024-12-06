import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.optim.lr_scheduler import LambdaLR

from models.Mymodel import STINet
from models.GNN import DGCNN
from models.CNN import EEGNet, FBCNet
from models.RNN import LSTM
from models.Conformer import Conformer

from early_stopping import EarlyStopping

def exists(item):
    return item is not None

class Trainer:
    def __init__(self, train_loader, test_loader, category_num, initial_lr, network_type, use_dynamic_lr=False, lr_decay_rate=0.01):
        # if cuda error raises, set self.device to 'cpu' to check errors.
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'
        
        self.train_loader = train_loader
        self.test_loader = test_loader


        if network_type == "EEGNet":
            self.model = EEGNet(chunk_size=32, num_electrodes=124, num_classes=category_num)
        elif network_type == "LSTM":
            self.model = LSTM(num_electrodes=124, num_classes=category_num, hid_channels=32)
        elif network_type == "DGCNN":
            self.model = DGCNN(in_channels=32, num_electrodes=124, num_classes=category_num)
        elif network_type == "Conformer":
            self.model = Conformer(n_classes=category_num)
        elif network_type == "FBCNet":
            self.model = FBCNet(num_classes=category_num, num_electrodes=124, chunk_size=32, in_channels=1)
        else:
            self.model = STINet(emb_dim=32, category_num=category_num)

        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=initial_lr, betas=(0.9,0.999))
        self.loss_func = nn.CrossEntropyLoss()

        if use_dynamic_lr:
            self.scheduler = LambdaLR(self.optimizer,lr_lambda = (lambda epoch: 1/(1+lr_decay_rate*epoch)))
        else:
            self.scheduler = None
        

    def train_val(self, epochnum):
        earlystop = EarlyStopping(path='./saved_models/model.pth', patience=15, verbose=True)
        valid_acc_list = []

        for epoch in range(1, epochnum+1):

            losses_in_one_epoch, acc_in_one_epoch = 0, 0
            n_batches, n_samples = len(self.train_loader), len(self.train_loader.dataset)

            losses_in_one_epoch_val, acc_in_one_epoch_val = 0, 0
            n_batches_val, n_samples_val = len(self.test_loader), len(self.test_loader.dataset)

            self.model.train()
            for i, batch in enumerate(self.train_loader):
                inputs, labels = map(lambda x: x.to(self.device), batch)

                aug_inputs = inputs
                aug_labels = labels

                model_output = self.model(aug_inputs)

                losses = self.loss_func(model_output, aug_labels)
                losses_in_one_epoch += losses

                acc = ((model_output.argmax(dim=-1) == aug_labels).sum()) / len(aug_labels)
                acc_in_one_epoch += acc

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()
                if exists(self.scheduler):
                    self.scheduler.step()
            
            train_loss = losses_in_one_epoch / n_batches
            train_acc = acc_in_one_epoch / n_batches
            
            if exists(self.scheduler):
                print('Epoch: {} \t Train Loss:{:.7f}\tAcc:{:.3f}%\tlearning rate:{:.7f}'
                      .format(epoch, train_loss, train_acc * 100. ,self.scheduler.get_last_lr()[0]))
            else:
                print('Epoch: {} \t Train Loss:{:.7f}\tAcc:{:.3f}%'.format(epoch, train_loss, train_acc * 100.))

            # validate
            self.model.eval()
            with torch.no_grad():

                for i, batch_val in enumerate(self.test_loader):
                    inputs_val, labels_val = map(lambda x: x.to(self.device), batch_val)
                    model_output_val = self.model(inputs_val)

                    losses_val = self.loss_func(model_output_val, labels_val)
                    losses_in_one_epoch_val += losses_val

                    acc_val = (model_output_val.argmax(dim=-1) == labels_val).sum()
                    acc_in_one_epoch_val += acc_val
                
                test_loss = losses_in_one_epoch_val / n_batches_val
                test_acc = acc_in_one_epoch_val / n_samples_val

                valid_acc_list.append(test_acc)

                print(('\tValidation -\tLoss:{:.7f}\tAcc:{:.3f}%').format(test_loss, test_acc * 100.))
                # earlystop(test_loss, self.model)
                earlystop(test_acc, self.model)
                if earlystop.early_stop:
                    best_acc = torch.tensor(valid_acc_list).max()
                    print('best validate acc in this fold: {:.3f}%. early stopping activated. skipping to next fold.'.format(best_acc * 100.))
                    break
                if epoch == epochnum:
                    best_acc = torch.tensor(valid_acc_list).max()
                    print('best validate acc in this fold: {:.3f}%. reached max epoch num. skipping to next fold.'.format(best_acc * 100.))

        return best_acc
