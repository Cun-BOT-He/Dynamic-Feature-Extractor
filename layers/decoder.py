# -*- coding: utf-8 -*-
# Copyright 2019 - present, Zhang Xinyu

import torch
import torch.nn as nn
import torch.nn.funtional as F

class DecoderRNN(nn.Module):
    def __init__(self, Datainput_size : int, Encoderoutput_size : int, seq_len : int):
        super(DecoderRNN, self).__init__()
        self.hidden_size = Datainput_size
        self.input_size = Encoderoutput_size
        self.seq_len = seq_len
        self.gru = nn.GRU(Encoderoutput_size, self.hidden_size)

    def forward(self, input, hidden=None):
        """
        Args:
        input_seq : Tensor, shape of (T,N,D), 
        T is the length of sequence, N is batch size, D is dimension of data.
        hidden : Tensor, shape of (1,N,DH), initial Hidden state.
        """
        if len(input_seq.shape) == 2:
            input_seq = torch.unsqueeze(input_seq, 1)
        if hidden == None:
            hidden = initHidden(input.shape[1], input.device)
        output, _ = self.gru(input.view(self.seq_len, -1, self.input_size), hidden)
        output = output.view(self.seq_len, -1, self.input_size)
        return output
    def initHidden(self, batch_size, device):
        return torch.zeros((1, batch_size, self.hidden_size), device=device)