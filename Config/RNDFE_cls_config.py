# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import models

class RNDFE_cls_cfg(models.RNDFEConfig):
    def __init__(self):
        super().__init__()
        self.seq_len = 20
        # ------------------------data config------------------------ #
        # fault_list and fault_dict are parameters for classification tasks
        # train_seg, valid_seg, test_seg are parameters for regression tasks
        self.train_dataset = dict(
            name = "_xl.csv",
            root = "E:\seq2seq Model\data\TEzeall_15fault",
            fault_list = [0,2,5,8,9,14],
            train_fault_list = [0,2,5,8,9,14],
            fault_dict = dict(
            {0:0,2:1,5:2,8:3,9:4,14:5}
            ),
            variable_list = [0,1,2,3,4,5,8,9,10,12,13,15,17,18,20,21],
            seq_len = self.seq_len
            )
        self.validate_dataset = dict(
            name = "_jy.csv",
            root = "E:\seq2seq Model\Data\TEzeall_15fault",
            fault_list = [0,2,5,8,9,14],
            train_fault_list = [0,2,5,8,9,14],
            fault_dict = dict(
            {0:0,2:1,5:2,8:3,9:4,14:5}
            ),
            variable_list = [0,1,2,3,4,5,8,9,10,12,13,15,17,18,20,21],
            seq_len = self.seq_len
        )
        self.test_dataset = dict(
            name = "_cs.csv",
            root = "E:\seq2seq Model\Data\TEzeall_15fault",
            fault_list = [0,2,5,8,9,14],
            train_fault_list = [0,2,5,8,9,14],
            fault_dict = dict(
            {0:0,2:1,5:2,8:3,9:4,14:5}
            ),
            variable_list = [0,1,2,3,4,5,8,9,10,12,13,15,17,18,20,21],
            seq_len = self.seq_len,
        )
        # ------------------------Train config----------------------- #
        self.dfe_task = "classification"
        self.lr = 4e-4
        self.train_batch_size = 20		