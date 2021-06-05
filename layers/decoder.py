# -*- coding: utf-8 -*-
# Copyright 2019 - present, Zhang Xinyu

import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderRNN(nn.Module):
    def __init__(self, input_size : int, hidden_size : int, seq_len : int):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.seq_len = seq_len
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input_seq, hidden=None):
        """
        Args:
        input_seq : Tensor, shape of (T,N,D), 
        T is the length of sequence, N is batch size, D is dimension of data.
        hidden : Tensor, shape of (1,N,DH), initial Hidden state.
        """
        if len(input_seq.shape) == 2:
            input_seq = torch.unsqueeze(input_seq, 1)
        if hidden == None:
            hidden = self.initHidden(input_seq.shape[1], input_seq.device)
        output, _ = self.gru(input_seq.view(self.seq_len, -1, self.input_size), hidden)
        output = output.view(self.seq_len, -1, self.hidden_size)
        return output
    def initHidden(self, batch_size, device):
        return torch.zeros((1, batch_size, self.hidden_size), device=device)