# -*- coding: utf-8 -*-
# Copyright 2019 - present, Zhang Xinyu

import torch
import torch.nn as nn
import torch.nn.functional as F
import layers

class RNDFE(nn.Module):
    """
    Implement Recurrent Networks based Dynamic Feature Extractor( https://ieeexplore.ieee.org/document/8859224 )
    """
    def __init__(self, cfg):
        super(RNDFE, self).__init__()
        self.cfg = cfg
        # --------------------build Encoder----------------- #
        self.encoder = layers.EncoderRNN(cfg.encoder_input_size, cfg.encoder_hidden_size, cfg.seq_len)
        # --------------------build Decoder----------------- #
        self.decoder = layers.DecoderRNN(cfg.decoder_input_size, cfg.decoder_hidden_size, cfg.seq_len)
        # ----------------build Attention Smooth------------ #
        self.attn_smooth = layers.Attnsmooth(cfg.attn_input_size, 
                                             cfg.attn_hidden_L1, 
                                             cfg.attn_hidden_L2, 
                                             cfg.attn_hidden_L3, 
                                             cfg.attn_device)
        # ----------------build Fully-connected------------- #
        self.fc = layers.fullyconnected(cfg.fc_input_size, cfg.fc_hidden_L1, cfg.fc_output_size)
        
    def forward(self, input_seq, target_seq, target=None):
        input_seq = input_seq.transpose(0,1)
        target_seq = target_seq.transpose(0,1)
        encoder_output, encoder_hidden = self.encoder(input_seq)
        decoder_input = encoder_hidden.repeat(self.cfg.seq_len,1,1)
        decoder_output = self.decoder(decoder_input)
        if self.training:
            if self.cfg.need_attn_smooth:
                attn_output, _ = self.attn_smooth(encoder_output)
                fc_output = self.fc(attn_output)
                loss_rec = F.mse_loss(decoder_output, target_seq)
                if self.cfg.dfe_task == "regression":
                    loss_fc = F.mse_loss(fc_output.squeeze(), target.squeeze())
                elif self.cfg.dfe_task == "classification":
                    loss_fc = F.cross_entropy(fc_output.squeeze(), target.squeeze())
                else:
                    raise RuntimeError("The task of DFE must be regression or classification.")
                loss = loss_rec + loss_fc
                return loss
            else:
                loss = F.mse_loss(decoder_output, target_seq)
                return loss
        else:
            if self.cfg.need_attn_smooth:
                dynamic_feature = encoder_hidden
                _, attenuation = self.attn_smooth(encoder_output)
                smooth_value = 1 / (torch.exp(torch.max(attenuation) - torch.min(attenuation)))
                dyna_mean = torch.mean(dynamic_feature)
                dynamic_feature = smooth_value * dynamic_feature + (1 - smooth_value) * dyna_mean
                return dynamic_feature, attenuation
            else:
                return encoder_hidden

class RNDFEConfig:
    def __init__(self):
        self.seq_len = 20
        # ------------------------data config------------------------ #
        # fault_list and fault_dict are parameters for classification tasks
        # train_seg, valid_seg, test_seg are parameters for regression tasks
        self.train_dataset = dict(
            name = "NSC_N3_Si_Content_data_20210227.csv",
            root = "E:\BF Softsense\Data\\",
            seq_len = self.seq_len
        )
        self.validate_dataset = dict(
            name = "NSC_N3_Si_Content_data_20210227.csv",
            root = "E:\BF Softsense\Data\\",
            seq_len = self.seq_len
        )
        self.test_dataset = dict(
            name = "NSC_N3_Si_Content_data_20210227.csv",
            root = "E:\BF Softsense\Data\\",
            seq_len = self.seq_len
        )
        # -----------------------encoder config---------------------- #
        self.encoder_input_size = 110
        self.encoder_hidden_size = 40
        # -----------------------decoder config---------------------- #
        self.decoder_input_size = 40
        self.decoder_hidden_size = 110
        # ------------------attention smooth config------------------ #
        self.need_attn_smooth = True
        self.attn_input_size = 40 
        self.attn_hidden_L1 = 64
        self.attn_hidden_L2 = 32
        self.attn_hidden_L3 = 16
        self.attn_device = 'cuda'
        # -------------------fully connected config------------------ #
        self.fc_input_size = 40
        self.fc_hidden_L1 = 20
        self.fc_output_size = 1
		# ------------------------Train config----------------------- #
        self.dfe_task = "regression"
        self.lr = 1e-2
        self.train_batch_size = 20
		# ------------------------Test config----------------------- #
        self.test_batch_size = 20