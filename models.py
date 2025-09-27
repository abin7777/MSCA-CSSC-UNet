import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import scipy.io
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # print("position", position.shape)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # print("div_term", div_term.shape)
        # print("position * div_term", (position * div_term).shape)
        try:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        except:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)[:,:-1]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:,:x.size(1), :]
        return x

class Lite_MSCA(nn.Module):
    def __init__(self, d_model, max_len, n_head=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pe = PositionalEncoding(d_model, max_len=max_len)
        self.MHA = nn.MultiheadAttention(d_model, n_head, batch_first=True)

    def forward(self, x1, x2):
        x1 = self.pe(x1)
        x2 = self.pe(x2)
        attn_output, attn_output_weights = self.MHA(x1, x2, x2)
        return attn_output

class Full_MSCA(nn.Module):
    def __init__(self, n_block, d_model, max_len, n_head=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_block = n_block
        self.models = nn.ModuleList([Lite_MSCA(d_model, max_len, n_head) for i in range(n_block)])

    def forward(self, inputs=[]):
        outputs = []
        x2 = inputs[-1]
        for i in range(self.n_block):
            output = self.models[i](inputs[i], x2)
            outputs.append(output)
        outputs = torch.cat(outputs, 1)
        return outputs

class SimpleUpsampleNetwork(nn.Module):
    def __init__(self, in_c, out_c, scale_factor=17.6):
        super().__init__()
        
        self.channel_reduce = nn.Conv1d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        self.upsample = nn.Upsample(
            scale_factor=scale_factor,
            mode='linear',
            align_corners=False
        )
        
        self.refine = nn.Conv1d(
            in_channels=out_c,
            out_channels=out_c,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        
        x = self.channel_reduce(x)  
        x = self.upsample(x)        
        x = self.refine(x)          
        return x

class MSA_UNet_skip_level_connation(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, deep_supervision=False, seq_len=88, *args, **kwargs):
        super().__init__(*args, **kwargs)

        nb_filter = [128, 256, 512, 1024, 2048]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool1d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.up_ = nn.Upsample(scale_factor=2.2, mode='linear', align_corners=True)

        # Downsample path (Encoder)
        self.conv0_0 = ConvBlock(in_channels, nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4])

        # Upsample path (Decoder)
        self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.x0_2_up = SimpleUpsampleNetwork(in_c=nb_filter[2],out_c=nb_filter[0],scale_factor=4)
        self.conv0_2 = ConvBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.x1_2_up = SimpleUpsampleNetwork(in_c=nb_filter[3],out_c=nb_filter[1],scale_factor=4)
        self.conv1_2 = ConvBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])
        self.x2_2_up = SimpleUpsampleNetwork(in_c=nb_filter[4],out_c=nb_filter[2],scale_factor=4.4)
        self.conv2_2 = ConvBlock(nb_filter[2] * 3 + nb_filter[3], nb_filter[2])

        self.x0_3_up_0 = SimpleUpsampleNetwork(in_c=nb_filter[3],out_c=nb_filter[0],scale_factor=8)
        self.x0_3_up_1 = SimpleUpsampleNetwork(in_c=nb_filter[2],out_c=nb_filter[0],scale_factor=4)
        self.conv0_3 = ConvBlock(nb_filter[0] * 5 + nb_filter[1], nb_filter[0])
        self.x1_3_up_0 = SimpleUpsampleNetwork(in_c=nb_filter[4],out_c=nb_filter[1],scale_factor=8.8)
        self.x1_3_up_1 = SimpleUpsampleNetwork(in_c=nb_filter[3],out_c=nb_filter[1],scale_factor=4)
        self.conv1_3 = ConvBlock(nb_filter[1] * 5 + nb_filter[2], nb_filter[1])

        self.x0_4_up_0 = SimpleUpsampleNetwork(in_c=nb_filter[4],out_c=nb_filter[0])
        self.x0_4_up_1 = SimpleUpsampleNetwork(in_c=nb_filter[3],out_c=nb_filter[0],scale_factor=8)
        self.x0_4_up_2 = SimpleUpsampleNetwork(in_c=nb_filter[2],out_c=nb_filter[0],scale_factor=4)
        self.conv0_4 = ConvBlock(nb_filter[0] * 7 + nb_filter[1], nb_filter[0])

        max_len = nb_filter[-1]*2
        self.attblock0_1 = Lite_MSCA(seq_len, max_len)
        self.attblock1_1 = Lite_MSCA(math.ceil(seq_len/2), max_len)
        self.attblock0_2 = Full_MSCA(2, seq_len, max_len)
        self.attblock2_1 = Lite_MSCA(math.ceil(seq_len/4), max_len, n_head=2)
        self.attblock1_2 = Full_MSCA(2, math.ceil(seq_len/2), max_len)
        self.attblock0_3 = Full_MSCA(3, seq_len, max_len)
        self.attblock3_1 = Lite_MSCA(math.ceil(seq_len/8), max_len, n_head=1)
        self.attblock2_2 = Full_MSCA(2, math.ceil(seq_len/4), max_len, n_head=2)
        self.attblock1_3 = Full_MSCA(3, math.ceil(seq_len/2), max_len)
        self.attblock0_4 = Full_MSCA(4, seq_len, max_len)

        if self.deep_supervision:
            self.final1 = nn.Conv1d(nb_filter[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv1d(nb_filter[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv1d(nb_filter[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv1d(nb_filter[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv1d(nb_filter[0], out_channels, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input) # [b, 1, 88] -> [b, 128, 88]
        x1_0 = self.conv1_0(self.pool(x0_0)) # [b, 128, 88] -> [b, 128, 44] -> [b, 256, 44]
        x0_1 = self.conv0_1(torch.cat([self.attblock0_1(x0_0, self.up(x1_0)), self.up(x1_0)], 1)) # [b, 128+256, 88] -> [b, 128, 88]

        x2_0 = self.conv2_0(self.pool(x1_0)) # [b, 256, 44] -> [b, 256, 22] -> [b, 512, 22]
        x1_1 = self.conv1_1(torch.cat([self.attblock1_1(x1_0, self.up(x2_0)), self.up(x2_0)], 1)) # [b, 256+512, 44] -> [b, 256, 44]
        
        x0_2 = self.conv0_2(torch.cat([self.attblock0_2([x0_0, x0_1, self.up(x1_1)]), self.up(x1_1), self.x0_2_up(x2_0)], 1)) # [b, 128+128+256, 88] -> [b, 128, 88]

        x3_0 = self.conv3_0(self.pool(x2_0)) # [b, 512, 22] -> [b, 512, 11] -> [b, 1024, 11]
        x2_1 = self.conv2_1(torch.cat([self.attblock2_1(x2_0, self.up(x3_0)), self.up(x3_0)], 1)) # [b, 512+1024, 22] -> [b, 512, 22]
        x1_2 = self.conv1_2(torch.cat([self.attblock1_2([x1_0, x1_1, self.up(x2_1)]), self.up(x2_1), self.x1_2_up(x3_0)], 1)) # [b, 256*2+512, 44] -> [b, 256, 44]
        x0_3 = self.conv0_3(torch.cat([self.attblock0_3([x0_0, x0_1, x0_2, self.up(x1_2)]), self.up(x1_2), self.x0_3_up_0(x3_0), self.x0_3_up_1(x2_1)], 1)) # [b, 128*3+256, 88] -> [b, 128, 88]

        x4_0 = self.conv4_0(self.pool(x3_0)) # [b, 1024, 11] -> [b, 1024, 5] -> [b, 2048, 5]
        x3_1 = self.conv3_1(torch.cat([self.attblock3_1(x3_0, self.up_(x4_0)), self.up_(x4_0)], 1)) # [b, 1024+2048, 11] -> [b, 1024, 11]
        x2_2 = self.conv2_2(torch.cat([self.attblock2_2([x2_0, x2_1, self.up(x3_1)]), self.up(x3_1), self.x2_2_up(x4_0)], 1)) # [b, 512*2+1024, 22] -> [b, 512, 22]
        x1_3 = self.conv1_3(torch.cat([self.attblock1_3([x1_0, x1_1, x1_2, self.up(x2_2)]), self.up(x2_2), self.x1_3_up_0(x4_0), self.x1_3_up_1(x3_1)], 1)) # [b, 256*3+512, 44] -> [b, 256, 44]
        x0_4 = self.conv0_4(torch.cat([self.attblock0_4([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)]), self.up(x1_3), self.x0_4_up_0(x4_0), self.x0_4_up_1(x3_1), self.x0_4_up_2(x2_2)], 1)) # [b, 128*4+256, 88] -> [b, 128, 88]

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1.squeeze()[:,-1], output2.squeeze()[:,-1],
                     output3.squeeze()[:,-1], output4.squeeze()[:,-1]]
        else:
            output = self.final(x0_4)
            return output.squeeze()[:,-1]







