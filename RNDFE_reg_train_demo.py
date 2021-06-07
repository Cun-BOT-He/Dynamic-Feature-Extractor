# -*- coding: utf-8 -*-

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
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

def main():
    cfg = config.RNDFE_reg_cfg()
    
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    BF_train = Datasets.BFDataset(cfg.train_dataset)
    BF_valid = Datasets.BFDataset(cfg.validate_dataset)
    BF_test = Datasets.BFDataset(cfg.test_dataset)
    
    train_loader = DataLoader(BF_train, batch_size = cfg.train_batch_size, shuffle = True)
    valid_loader = DataLoader(BF_valid, batch_size = cfg.train_batch_size, shuffle = True)
    test_loader = DataLoader(BF_test, batch_size = cfg.train_batch_size, shuffle = False)
    
    DFE = models.RNDFE(cfg).to(dev)
    DFE.train()
    DFE_optimizer = optim.SGD(DFE.parameters(), lr = cfg.lr)
    DFE_scheduler = optim.lr_scheduler.ReduceLROnPlateau(DFE_optimizer)
    
    # train process
    start = time.time()
    plot_losses = [[],[]]
    print_loss_total = 0  # Reset every print_every
    print_valid_loss_total = 0
    plot_loss_total = 0  # Reset every plot_every
    plot_valid_loss_total = 0
    for iter in range(1, cfg.train_max_epoch + 1):
        train_loss_total = 0
        for trainData, trainLabel in train_loader:
            loss = DFE(trainData.float().to(dev), trainData.float().to(dev), trainLabel.float().to(dev))
            loss.backward()
            DFE_optimizer.step()
            train_loss_total += loss * trainData.shape[0]
        print_loss_total += train_loss_total / len(BF_train)
        plot_loss_total += train_loss_total / len(BF_train)
        
        with torch.no_grad():
            valid_loss_total = 0
            for validData, validLabel in valid_loader:
                loss = DFE(validData.float().to(dev), validData.float().to(dev), validLabel.float().to(dev))
                valid_loss_total += loss * validData.shape[0]
        DFE_scheduler.step(valid_loss_total / len(BF_valid))
        print_valid_loss_total += valid_loss_total / len(BF_valid)
        plot_valid_loss_total += valid_loss_total / len(BF_valid)
        if iter % cfg.train_print_every == 0:
            print_loss_avg = print_loss_total / cfg.train_print_every
            print_valid_loss_avg = print_valid_loss_total / cfg.train_print_every
            print_loss_total = 0
            print_valid_loss_total = 0
            print('%s (%d %d%%) train loss: %.4f   valid loss: %.4f' % (timeSince(start, iter / cfg.train_max_epoch),
                                         iter, iter / cfg.train_max_epoch * 100, print_loss_avg, print_valid_loss_avg))
    
        if iter % cfg.train_plot_every == 0:
            plot_loss_avg = plot_loss_total / cfg.train_plot_every
            plot_valid_loss_avg = plot_valid_loss_total / cfg.train_plot_every
            plot_losses[0].append(plot_loss_avg)
            plot_losses[1].append(plot_valid_loss_avg)
            plot_loss_total = 0
            plot_valid_loss_total = 0
            
    PATH = "DFE_reg_demo_ATTN.pt"
    torch.save(DFE.state_dict(), PATH)
    
if __name__ == "__main__":
    main()