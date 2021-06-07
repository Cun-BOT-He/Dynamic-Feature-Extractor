# -*- coding: utf-8 -*-

import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import models
import config
import Datasets

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

class MLP_1Layer(nn.Module):
    def __init__(self, input_size, hl_num):
        """
        hl_num: int/list, the number of neurons in each hidden layer 
        """
        super(MLP_1Layer, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hl_num)
        self.output_layer = nn.Linear(hl_num, 1)
    def forward(self, input_data):
        out = F.rrelu(self.hidden_layer(input_data))
        out = F.rrelu(self.output_layer(out))
        return out

class MLP_2Layers(nn.Module):
    def __init__(self, input_size, h1_num, h2_num):
        """
        hl_num: int/list, the number of neurons in each hidden layer 
        """
        super(MLP_2Layers, self).__init__()
        self.h1_layer = nn.Linear(input_size, h1_num)
        self.h2_layer = nn.Linear(h1_num, h2_num)
        self.output_layer = nn.Linear(h2_num, 1)
    def forward(self, input_data):
        out = F.rrelu(self.h1_layer(input_data))
        out = F.rrelu(self.h2_layer(out))
        out = F.rrelu(self.output_layer(out))
        return out

def computeLoss(model, input_batch, target_batch, criterion, opt = None):
    loss = criterion(model(input_batch).squeeze(), target_batch)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
        return loss.item()
    return loss.item()
    
class SoftsenseData(Dataset):
        def __init__(self, X, y):
            """
            X: Tensor, (N, D), 用于预测的自变量
            y: Tensor, (N, 1), 用于预测的目标变量
            """                             
            self.X = X
            self.y = y
        def __len__(self):
            return self.X.size(0)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

def main():
    # Reproducibility and device setting
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    cfg = config.RNDFE_reg_cfg()
    DFE = models.RNDFE(cfg).to(dev)
    DFE.load_state_dict(torch.load(cfg.save_path))
    DFE.eval()
    
    BF_train = Datasets.BFDataset(cfg.train_dataset)
    BF_valid = Datasets.BFDataset(cfg.validate_dataset)
    BF_test = Datasets.BFDataset(cfg.test_dataset)
    
    train_loader = DataLoader(BF_train, batch_size = cfg.test_batch_size, shuffle = True)
    valid_loader = DataLoader(BF_valid, batch_size = cfg.test_batch_size, shuffle = True)
    test_loader = DataLoader(BF_test, batch_size = cfg.test_batch_size, shuffle = False)

    # reset the dynamic feature
    hgq_train = torch.tensor([], device = dev)
    dyna_feature_train = torch.tensor([], device = dev)
    hgq_train_label = torch.tensor([], dtype = torch.float, device = dev)
    
    hgq_valid= torch.tensor([], device = dev)
    dyna_feature_valid = torch.tensor([], device = dev)
    hgq_valid_label = torch.tensor([], dtype = torch.float, device = dev)
    
    hgq_test = torch.tensor([], device = dev)
    dyna_feature_test = torch.tensor([], device = dev)
    hgq_test_label = torch.tensor([], dtype = torch.float, device = dev)
    
    start = time.time()
    with torch.no_grad():
        for trainData, trainLabel in train_loader:
            dynamic_feature, _ = DFE(trainData.float().to(dev), trainData.float().to(dev))
            x_t = trainData.transpose(0,1)[-1].float().to(dev)
            hgq_train = torch.cat((hgq_train, x_t), 0)
            dyna_feature_train = torch.cat((dyna_feature_train, dynamic_feature), 1)
            hgq_train_label = torch.cat((hgq_train_label, trainLabel.float().to(dev)), 0)
        for validData, validLabel in valid_loader:
            dynamic_feature, _ = DFE(validData.float().to(dev), validData.float().to(dev))
            x_t = validData.transpose(0,1)[-1].float().to(dev)
            hgq_valid = torch.cat((hgq_valid, x_t), 0)
            dyna_feature_valid = torch.cat((dyna_feature_valid, dynamic_feature), 1)
            hgq_valid_label = torch.cat((hgq_valid_label, validLabel.float().to(dev)), 0)
        for testData, testLabel in test_loader:
            dynamic_feature, _ = DFE(testData.float().to(dev), testData.float().to(dev))
            x_t = testData.transpose(0,1)[-1].float().to(dev)
            hgq_test = torch.cat((hgq_test, x_t), 0)
            dyna_feature_test = torch.cat((dyna_feature_test, dynamic_feature), 1)
            hgq_test_label = torch.cat((hgq_test_label, testLabel.float().to(dev)), 0)
    end = time.time()
    s = end - start
    print("done")
    print('Time using to extract dynamic feature : %d s' % (s))
    
    dyna_dim = cfg.encoder_hidden_size
    # feature combine
    hgq_train_hc = torch.cat((hgq_train, dyna_feature_train.view(-1, dyna_dim)), 1)
    hgq_valid_hc = torch.cat((hgq_valid, dyna_feature_valid.view(-1, dyna_dim)), 1)
    hgq_test_hc = torch.cat((hgq_test, dyna_feature_test.view(-1, dyna_dim)), 1)
    
    softsense_train_data = SoftsenseData(X = hgq_train_hc, y = hgq_train_label)
    softsense_valid_data = SoftsenseData(X = hgq_valid_hc, y = hgq_valid_label)
    ss_train = DataLoader(softsense_train_data, batch_size = cfg.test_batch_size, shuffle = True)
    ss_valid = DataLoader(softsense_valid_data, batch_size = cfg.test_batch_size, shuffle = True)
    
    input_size = hgq_train_hc.shape[1]
    hiddenL1_size = 32
    hiddenL2_size = 16
    learning_rate = 1e-3
    l2_yita = 1e-4
    hgq = MLP_2Layers(input_size, hiddenL1_size, hiddenL2_size).to(dev)
    # flq = MLP_1Layer(input_size, num_classes, hiddenL1_size).to(dev)
    
    criterion = nn.MSELoss()
    hgq_optimizer = optim.Adam(hgq.parameters(), lr = learning_rate, weight_decay = l2_yita)
    hgq_scheduler = optim.lr_scheduler.ReduceLROnPlateau(hgq_optimizer)
    
    start = time.time()
    plot_losses = [[],[]]
    print_loss_total = 0  # Reset every print_every
    print_valid_loss_total = 0
    
    plot_loss_total = 0  # Reset every plot_every
    plot_valid_loss_total = 0
    for iter in range(1, cfg.ds_max_epoch + 1):
        hgq.train()
        b_count = 0
        train_loss = 0
        for trainData, trainLabel in ss_train:
            loss = computeLoss(hgq, trainData.float().to(dev), trainLabel.float().to(dev),
                              criterion, hgq_optimizer)
            train_loss += loss
            b_count += 1
        print_loss_total += train_loss / b_count
        plot_loss_total += train_loss / b_count
        
        with torch.no_grad():
            valid_loss_total = 0
            hgq.eval()
            for validData, validLabel in ss_valid:
                loss = computeLoss(hgq, validData.float().to(dev), validLabel.float().to(dev), criterion)
                valid_loss_total += loss * len(validData)
        hgq_scheduler.step(valid_loss_total / len(BF_valid))
        print_valid_loss_total += valid_loss_total / len(BF_valid)
        plot_valid_loss_total += valid_loss_total / len(BF_valid)
        if iter % cfg.ds_print_every == 0:
            print_loss_avg = print_loss_total / cfg.ds_print_every
            print_valid_loss_avg = print_valid_loss_total / cfg.ds_print_every
            print_loss_total = 0
            print_valid_loss_total = 0
            print('%s (%d %d%%) train loss: %.6f   valid loss: %.6f' % (timeSince(start, iter / cfg.ds_max_epoch),
                                         iter, iter / cfg.ds_max_epoch * 100, print_loss_avg, print_valid_loss_avg))
    
        if iter % cfg.ds_plot_every == 0:
            plot_loss_avg = plot_loss_total / cfg.ds_plot_every
            plot_valid_loss_avg = plot_valid_loss_total / cfg.ds_plot_every
            plot_losses[0].append(plot_loss_avg)
            plot_losses[1].append(plot_valid_loss_avg)
            plot_loss_total = 0
            plot_valid_loss_total = 0
    
    predicted_test = hgq(hgq_test_hc)
    BF_rmse = np.sqrt(mean_squared_error(hgq_test_label.detach().cpu().numpy(), predicted_test.detach().cpu().numpy()))
    BF_r2 = r2_score(hgq_test_label.detach().cpu().numpy(), predicted_test.detach().cpu().numpy())
    print('Blast Furnance Test RMSE: %.6f' % (BF_rmse))
    print('Blast Furnance Test R2: %.6f' % (BF_r2))
    
if __name__ == "__main__":
    main()