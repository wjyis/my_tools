import torch
import torch.nn as nn
import torch.nn.functional as F


#Swish激活函数,可以理解为Relu的上位替代品
class Swish(nn.Module):
    def __init__(self):
        super(Swish,self).__init__()
        #这里可以选择beta是否是可学习的
        self.beta = nn.Parameter(torch.ones(1))
        #当beta=1时,称为SiLU激活函数
        # self.beta = 1
    
    def forward(self,x):
        return x * torch.sigmoid(self.beta*x)

#Mish激活函数,据说用在视觉任务比较好    
class Mish(nn.Module):
    def __init__(self):
        super(Mish,self).__init__()
    
    def forward(self,x):
        return x*(torch.tanh(F.softplus(x)))



#softplus的平替版,据说计算速度很快
class Squareplus(nn.Module):
    def __init__(self, b=0.2):
        super(Squareplus, self).__init__()
        self.b = b

    def forward(self, x):
        x = 0.5 * (x + torch.sqrt(x+self.b))
        return x




#SwiGLU激活函数,引入了门控机制
#过于复杂,暂时不加入



class ACT(nn.Module):
    def __init__(self,act,) -> None:
        super(ACT,self,).__init__()
        if act == 'relu':
            self.act = nn.relu()
        if act == 'leakRelu':
            self.act = nn.LeakyReLU(0.01)
        if act == 'swish':
            self.act = Swish()
        if act == 'Mish':
            self.act = Mish()
        if act == 'Gelu':
            self.act = F.gelu()
        if act == 'Tanh':
            self.act = F.tanh()
        if act == 'softplus':
            self.act = nn.Softplus()
        if act == 'squareplus':
            self.act = Squareplus()
        #如果有其他的,就在这里添加
    
    def forward(self,x):
        return self.act(x)
        
        
        
        
        