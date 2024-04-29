import torch
import torch.nn as nn



#这是3D通道注意力
class CAlayer3D(nn.Module):
    def __init__(self, channel, reduction, act = 'swish'):
        super(CAlayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        
        self.conv1 = nn.Conv3d(channel, channel // reduction, 1, padding=0, bias=True)
        self.conv2 = nn.Conv3d(channel // reduction, channel, 1, padding=0, bias=True)
        if act == 'swish':
            self.act = Swish()
        elif act == 'relu':
            self.act = nn.Relu()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        y = self.avg_pool(x)
        y = self.act(self.conv1(y))
        y = self.sigmoid(self.conv2(y))
        return x * y
    

#这是2D通道注意力
class CAlayer2D(nn.Module):
    def __init__(self, channel, reduction, act = 'swish'):
        super(CAlayer2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.conv1 = nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True)
        if act == 'swish':
            self.act = Swish()
        elif act == 'relu':
            self.act = nn.Relu()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        y = self.avg_pool(x)
        y = self.act(self.conv1(y))
        y = self.sigmoid(self.conv2(y))
        return x * y


class Swish(nn.Module):
    def __init__(self):
        super(Swish,self).__init__()
        self.beta = nn.Parameter(torch.ones(1))
        # self.beta = 1
    
    def forward(self,x):
        return x * torch.sigmoid(self.beta*x)

if __name__ == 'main':
    a = torch.zeros((16,8,48,48))
    clayer2d = CAlayer2D(8,4)
    CAlayer2D(a)
