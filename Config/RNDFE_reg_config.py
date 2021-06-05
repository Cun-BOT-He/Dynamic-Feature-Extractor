# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import models

class RNDFE_reg_cfg(models.RNDFEConfig):
    def __init__(self):
        super().__init__()
        self.seq_len = 20
        # ------------------------data config------------------------ #
        # train_seg, valid_seg, test_seg are parameters for regression tasks
        self.train_dataset = dict(
            name = "NSC_N3_Si_Content_data_20210227.csv",
            root = "E:\BF Softsense\Data\\",
            seq_len = self.seq_len,
            train_seg = 14000,
            start_seg = 0,
            end_seg = 14000
        )
        self.validate_dataset = dict(
            name = "NSC_N3_Si_Content_data_20210227.csv",
            root = "E:\BF Softsense\Data\\",
            seq_len = self.seq_len,
            train_seg = 14000,
            start_seg = 14000,
            end_seg = 16000
        )
        self.test_dataset = dict(
            name = "NSC_N3_Si_Content_data_20210227.csv",
            root = "E:\BF Softsense\Data\\",
            seq_len = self.seq_len,
            train_seg = 14000,
            start_seg = 16000,
            end_seg = 20000
        )
        # -------------------Attention smooth config----------------- #
        self.need_attn_smooth = False
        # ------------------------Train config----------------------- #
        self.dfe_task = "regression"
        self.lr = 4e-4
        self.train_batch_size = 20