# -*- coding: utf-8 -*-
# Copyright 2019 - present, Zhang Xinyu

import torch
import torch.nn as nn
import torch.nn.functional as F

class fullyconnected(nn.Module):
    def __init__(self, input_size : int, hidden1 : int, output_size : int):
        super(fullyconnected, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden1)
        self.hidden2 = nn.Linear(hidden1, output_size)
    def forward(self, input_data):
        out = F.relu(self.hidden1(input_data))
        out = F.relu(self.hidden2(out))
        return out