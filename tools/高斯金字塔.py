import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianPyramid(nn.Module):
    def __init__(self, levels, kernel_size=5, sigma = 1.0):
        super(GaussianPyramid,self).__init__()
        self.levels = levels
        self.kernel_size = kernel_size
        self.gaussian_filter = self.gaussian_kernel(kernel_size, sigma).to(torch.float32)

    def gaussian_kernel(self,kernel_size,sigma = 1.0):
        #生成一个高斯核
        kernel_range = np.arange(-kernel_size//2 + 1. , kernel_size//2 + 1. )
        x, y = np.meshgrid(kernel_range, kernel_range)
        kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel /= np.sum(kernel)
        return torch.Tensor(kernel).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        self.gaussian_filter = self.gaussian_filter.to(x.device)
        pyramid = [[] for _ in range(self.levels)]
        b,c,v,h,w = x.shape
        for view_index in range(v):
            current_image = x[:,:,view_index,:,:]
            for level in range(self.levels):
                if level == 0:
                    
                    pyramid[level].append(current_image)
                else:
                    filter_image = F.conv2d(current_image, self.gaussian_filter, padding = self.kernel_size//2, groups=c)
                    current_image = F.interpolate(filter_image, scale_factor=0.5, mode='area')
                    pyramid[level].append(current_image)
        for level in range(self.levels):
            pyramid[level] = torch.stack(pyramid[level], dim = 2)
        return pyramid