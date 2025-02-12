import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils
import copy

class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0.0, use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x

class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0.0, use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class Context_Excitation(nn.Module):
    def __init__(self, config,args):
        super(Context_Excitation, self).__init__()
        self.mlp1 = MLP(config.feature_size+6,int(config.feature_size/2),6,args.dropout)
        self.mlp2 = MLP(6+config.feature_size,int(config.feature_size/2),config.feature_size,args.dropout)
        self.sigmoid = nn.Sigmoid()
        self.args = args
        
    def forward(self, i, p):

        # p: batch_size * feature_size * hidden_size
        # i: batch_size * 6 * item_size

        p = torch.mean(p,dim=2) # batch_size * feature_size
        i = torch.mean(i,dim=2) # batch_size * 6

        union = torch.cat((i,p),dim=1)

        attn_i = self.mlp1(union)
        attn_p = self.mlp2(union)

        attn_i = F.softmax(attn_i,dim=1).unsqueeze(dim=2)
        attn_p = F.softmax(attn_p,dim=1).unsqueeze(dim=2)

        return attn_i,attn_p