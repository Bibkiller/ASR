import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# semantic decoupling
class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, label, class_num):
        label = label.unsqueeze(1)
        gt = torch.zeros(label.shape[0],class_num).scatter_(1,label,1)
        out = torch.log(1+torch.exp(torch.mm(-input,gt.transpose(1,0).cuda())))
        res = torch.diag(out).sum()/out.shape[0]
        return res
    
    
