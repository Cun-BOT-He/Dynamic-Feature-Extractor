# -*- coding: utf-8 -*-
# Copyright 2019 - present, Zhang Xinyu

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attnsmooth(nn.Module):
    def __init__(self, input_size : int, hidden1 : int, hidden2 : int, hidden3 : int, device = 'cpu'):
        """
        Args:
        input_size : 输入向量维度
        hidden1 : fc1隐变量维度
        hidden2 ：fc2层隐变量维度
        hidden3 ： fc3层隐变量维度
        device ： input sequence所处设备，默认为cpu
        """
        super(Attnsmooth, self).__init__()
        self.inputsize = input_size
        self.hidden1 = nn.Linear(input_size, hidden1)
        self.hidden2 = nn.Linear(hidden1, hidden2)
        self.hidden3 = nn.Linear(hidden2, hidden3)
        self.uw = nn.Parameter(torch.empty([hidden3], dtype = torch.float, device=device))
        
    def forward(self, input_seq):
        """
        Args:
        input_seq: T * BATCH * DH, Encoder输出的所有隐变量
        """
        if len(input_seq.shape) == 2:
            input_seq = torch.unsqueeze(input_seq, 1)
        attenuation_values = []
        dev = input_seq.device
        for input_hidden in input_seq:
            ui = F.relu(self.hidden1(input_hidden))
            ui = F.relu(self.hidden2(ui))
            ui = F.relu(self.hidden3(ui))
            attenuation_row = torch.matmul(ui, self.uw)
            attenuation_values.append(attenuation_row)
        attenuation = torch.stack(attenuation_values, 0)
        attenuation = F.softmax(attenuation, 0)
        output = torch.sum(torch.mul(input_seq, attenuation.unsqueeze(2)), 0)

        return output, attenuation