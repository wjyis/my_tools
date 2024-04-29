import torch
import torch.nn as nn

class Swish(nn.Module):
    def __init__(self):
        super(Swish,self).__init__()
        #这里可以选择beta是否是可学习的
        self.beta = nn.Parameter(torch.ones(1))
        # self.beta = 1
    
    def forward(self,x):
        return x * torch.sigmoid(self.beta*x)

