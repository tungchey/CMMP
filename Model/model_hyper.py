import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch.nn import Parameter
import numpy as np, itertools, random, copy, math
import math
import scipy.sparse as sp

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

class Hyper(nn.Module):
    def __init__(self, n_dim, nhidden, nclass, dropout, variant, use_residue, n_speakers=2, modals=['a', 'v', 'l'], use_speaker=True, use_modal=False):
        super(Hyper, self).__init__()
        self.use_residue = use_residue
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.modals = modals
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.use_position = False
        #------------------------------------    
        self.fc1 = nn.Linear(n_dim, nhidden)


    def forward(self, a, v, l, dia_len, qmask, epoch):

        features = self.hyper(a, v, l, dia_len, self.modals)
        x1 = self.fc1(features)
        out = x1
        if self.use_residue:
            out1 = torch.cat([features, out], dim=-1)

        nn_out = x1
        out2 = torch.cat([out, nn_out], dim=1)
        if self.use_residue:
            out2 = torch.cat([features, out2], dim=-1)
        out1 = self.reverse_features(dia_len, out2)
        # ---------------------------------------
        return out1
    def hyper(self, a, v, l, dia_len, modals):
        self_loop = False
        num_modality = len(modals)
        n_count = 0
        for i in dia_len:
            if n_count == 0:
                ll = l[0:0+i]
                aa = a[0:0+i]
                vv = v[0:0+i]
                features = torch.cat([ll, aa, vv], dim=0)
                temp = 0+i
            else:
                ll = l[temp:temp+i]
                aa = a[temp:temp+i]
                vv = v[temp:temp+i]
                features_temp = torch.cat([ll, aa, vv],dim=0)
                features = torch.cat([features, features_temp],dim=0)
                temp = temp+i

            n_count = n_count + i*num_modality
        return features

    def reverse_features(self, dia_len, features):
        l=[]
        a=[]
        v=[]
        for i in dia_len:
            ll = features[0:1*i]
            aa = features[1*i:2*i]
            vv = features[2*i:3*i]
            features = features[3*i:]
            l.append(ll)
            a.append(aa)
            v.append(vv)
        tmpl = torch.cat(l,dim=0)
        tmpa = torch.cat(a,dim=0)
        tmpv = torch.cat(v,dim=0)
        features = torch.cat([tmpl, tmpa, tmpv], dim=-1)
        return features

