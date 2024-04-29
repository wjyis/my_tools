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

#Gelu激活函数,据说适用于NLP领域
#内置于nn.functional中
# F.gelu(x)

#SwiGLU激活函数,引入了门控机制
#过于复杂,暂时不加入



