import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from tqdm import tqdm

from utils.utils import *


class SCINet_decompose(nn.Module):
    def __init__(self, output_len, input_len, input_dim = 9, hid_size = 1, num_stacks = 1,
                num_levels = 3, concat_len = 0, groups = 1, kernel = 5, dropout = 0.5,
                 single_step_output_One = 0, input_len_seg = 0, positionalE = False, modified = True, RIN=False):
        super(SCINet_decompose, self).__init__()
        
        self.input_dim = input_dim
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_size = hid_size
        self.num_levels = num_levels
        self.groups = groups
        self.modified = modified
        self.kernel_size = kernel
        self.dropout = dropout
        self.single_step_output_One = single_step_output_One
        self.concat_len = concat_len
        self.pe = positionalE
        self.RIN=RIN
        self.decomp = series_decomp(25)
        self.trend = nn.Linear(input_len,input_len)
        self.trend_dec = nn.Linear(input_len,output_len)
        self.blocks1 = EncoderTree(
            in_planes=self.input_dim,
            num_levels = self.num_levels,
            kernel_size = self.kernel_size,
            dropout = self.dropout,
            groups = self.groups,
            hidden_size = self.hidden_size,
            INN =  modified)

        if num_stacks == 2: # we only implement two stacks at most.
            self.blocks2 = EncoderTree(
                in_planes=self.input_dim,
            num_levels = self.num_levels,
            kernel_size = self.kernel_size,
            dropout = self.dropout,
            groups = self.groups,
            hidden_size = self.hidden_size,
            INN =  modified)

        self.stacks = num_stacks

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        self.projection1 = nn.Conv1d(self.input_len, self.output_len, kernel_size=1, stride=1, bias=False)
        if self.single_step_output_One: # only output the N_th timestep.
            if self.stacks == 2:
                if self.concat_len:
                    self.projection2 = nn.Conv1d(self.concat_len + self.output_len, 1,
                                                kernel_size = 1, bias = False)
                else:
                    self.projection2 = nn.Conv1d(self.input_len + self.output_len, 1,
                                                kernel_size = 1, bias = False)
        else: # output the N timesteps.
            if self.stacks == 2:
                if self.concat_len:
                    self.projection2 = nn.Conv1d(self.concat_len + self.output_len, self.output_len,
                                                kernel_size = 1, bias = False)
                else:
                    self.projection2 = nn.Conv1d(self.input_len + self.output_len, self.output_len,
                                                kernel_size = 1, bias = False)

        # For positional encoding
        self.pe_hidden_size = input_dim
        if self.pe_hidden_size % 2 == 1:
            self.pe_hidden_size += 1
    
        num_timescales = self.pe_hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0

        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                max(num_timescales - 1, 1))
        temp = torch.arange(num_timescales, dtype=torch.float32)
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales)

        ### RIN Parameters ###
        if self.RIN:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, input_dim))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, input_dim))
            self.affine_weight2 = nn.Parameter(torch.ones(1, 1, input_dim))
            self.affine_bias2 = nn.Parameter(torch.zeros(1, 1, input_dim))
    
    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32, device=x.device)  # tensor([0., 1., 2., 3., 4.], device='cuda:0')
        temp1 = position.unsqueeze(1)  # 5 1
        temp2 = self.inv_timescales.unsqueeze(0)  # 1 256
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)  # 5 256
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)  #[T, C]
        signal = F.pad(signal, (0, 0, 0, self.pe_hidden_size % 2))
        signal = signal.view(1, max_length, self.pe_hidden_size)
    
        return signal

    def forward(self, x):
        assert self.input_len % (np.power(2, self.num_levels)) == 0 # evenly divided the input length into two parts. (e.g., 32 -> 16 -> 8 -> 4 for 3 levels)
        x, trend = self.decomp(x)

        if self.RIN:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
            # seq_means = x[:,-1,:].unsqueeze(1).repeat(1,self.input_len,1).detach()
            # pred_means = x[:,-1,:].unsqueeze(1).repeat(1,self.output_len,1).detach()
            # x = x - seq_means
            x = x * self.affine_weight + self.affine_bias

            # print('/// RIN ACTIVATED ///\r',end='')
            means2 = trend.mean(1, keepdim=True).detach()
            trend = trend - means2
            stdev2 = torch.sqrt(torch.var(trend, dim=1, keepdim=True, unbiased=False) + 1e-5)
            trend /= stdev2
            # seq_means2 = trend[:,-1,:].unsqueeze(1).repeat(1,self.input_len,1).detach()
            # pred_means2 = trend[:,-1,:].unsqueeze(1).repeat(1,self.output_len,1).detach()
            # trend = trend - seq_means2 
            trend = trend * self.affine_weight2 + self.affine_bias2
        

        if self.pe:
            pe = self.get_position_encoding(x)
            if pe.shape[2] > x.shape[2]:
                x = x + pe[:, :, :-1]
            else:
                x = x + self.get_position_encoding(x)

        ### activated when RIN flag is set ###
        

        # the first stack
        res1 = x
        x = self.blocks1(x)
        x = self.projection1(x)

        trend = trend.permute(0,2,1)
        trend = self.trend(trend)  
        trend = self.trend_dec(trend).permute(0,2,1)

        if self.stacks == 1:
            ### reverse RIN ###
            if self.RIN:
                x = x - self.affine_bias
                x = x / (self.affine_weight + 1e-10)
                # x = x + pred_means
                x = x * stdev
                x = x + means

                trend = trend - self.affine_bias2
                trend = trend / (self.affine_weight2 + 1e-10)
                # trend = trend + pred_means2
                trend = trend * stdev2
                trend = trend + means2

            return x + trend

        elif self.stacks == 2:
            MidOutPut = x
            if self.concat_len:
                x = torch.cat((res1[:, -self.concat_len:,:], x), dim=1)
            else:
                x = torch.cat((res1, x), dim=1)

            # the second stack
            x = self.blocks2(x)
            x = self.projection2(x)
            
            ### Reverse RIN ###
            if self.RIN:
                MidOutPut = MidOutPut - self.affine_bias
                MidOutPut = MidOutPut / (self.affine_weight + 1e-10)
                MidOutPut = MidOutPut * stdev
                MidOutPut = MidOutPut + means

                x = x - self.affine_bias
                x = x / (self.affine_weight + 1e-10)
                x = x * stdev
                x = x + means

                trend = trend - self.affine_bias2
                trend = trend / (self.affine_weight2 + 1e-10)
                # trend = trend + pred_means2
                trend = trend * stdev2
                trend = trend + means2

            return x + trend, MidOutPut
