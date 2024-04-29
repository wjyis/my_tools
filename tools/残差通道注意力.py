import torch
import torch.nn as nn


#这里是3D残差通道注意力
#注意激活函数可以自行更换,默认使用的是swish
class RCAB3D(nn.Module):
    def __init__(self, in_channel, kernel_size, reduction, bias=True, bn=False, res_scale = 1):
        super(RCAB3D,self).__init__()
        self.res_scale = res_scale
        self.conv = nn.Conv3d(in_channel, in_channel, kernel_size, padding=kernel_size//2, bias=bias)
        self.if_bn = bn
        self.bn = nn.BatchNorm3d(in_channel)
        self.swish1 = Swish()
        self.swish2 = Swish()
        self.calayer = CAlayer3D(in_channel, reduction)
    
    def forward(self,x):
        res = self.conv(x)
        if self.if_bn:
            res = self.bn(res)
        res = self.swish1(res)
        res = self.conv(res)
        if self.if_bn:
            res = self.bn(res)
        res = self.swish2(res)
        res = self.calayer(res) * self.res_scale
        res += x
        return res
        

#这里的reduciton表示缩减比例,在运算过程中,通道数 = 实际通道数/reduciton
class CAlayer3D(nn.Module):
    def __init__(self, channel, reduction):
        super(CAlayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        
        self.conv1 = nn.Conv3d(channel, channel // reduction, 1, padding=0, bias=True)
        self.conv2 = nn.Conv3d(channel // reduction, channel, 1, padding=0, bias=True)
        self.swish = Swish()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        y = self.avg_pool(x)
        y = self.swish(self.conv1(y))
        y = self.sigmoid(self.conv2(y))
        return x * y


class RCAB2D(nn.Module):
    def __init__(self, in_channel, kernel_size, reduction, bias=True, bn=False, res_scale = 1):
        super(RCAB2D,self).__init__()
        self.res_scale = res_scale
        self.conv = nn.Conv2d(in_channel, in_channel, kernel_size, padding=kernel_size//2, bias=bias)
        self.if_bn = bn
        self.bn = nn.BatchNorm2d(in_channel)
        self.swish1 = Swish()
        self.swish2 = Swish()
        self.calayer = CAlayer2D(in_channel, reduction)
    
    def forward(self,x):
        res = self.conv(x)
        if self.if_bn:
            res = self.bn(res)
        res = self.swish1(res)
        res = self.conv(res)
        if self.if_bn:
            res = self.bn(res)
        res = self.swish2(res)
        res = self.calayer(res) * self.res_scale
        res += x
        return res





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


#使用示例
if __name__ == '__main__':
    a = torch.zeros((16,8,48,48))
    b = torch.zeros((16,8,9,48,48))
    clayer2d = RCAB2D(8,3,4)
    clayer3d = RCAB3D(8,3,4)
    clayer2d(a)
    clayer3d(b)
