import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#需要注意的是,这个拉普拉斯金字塔返回的每一层都具有相同的形状
#这是由于我所要完成的任务来决定的
#使用时根据具体任务再做更改
class LaplacePyramid(nn.Module):
    def __init__(self,levels, kernel_size = 5, sigma = 1.0):
        super(LaplacePyramid,self).__init__()
        self.levels = levels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian_filter = GaussianPyramid(levels, kernel_size, sigma)

    def forward(self, x):
        laplacian_pyramid = []
        gaussian_pyramid = self.gaussian_filter(x)
        laplacian_pyramid.append(gaussian_pyramid[0])
        for idx in range(self.levels - 1):
            current_gaussian = gaussian_pyramid[idx]
            current_img = gaussian_pyramid[idx + 1]
            laplace_imgs = []
            for i in range(current_gaussian.shape[2]):
                upsampled_gaussian = F.interpolate(current_img[:, :, i], scale_factor=2, mode='bilinear', align_corners=False)
                laplace_img = current_gaussian[:, :, i] - upsampled_gaussian
                #这里与一般的拉普拉斯金字塔逻辑不同
                #使用时请格外注意
                if idx == 1:
                    laplace_img = F.interpolate(laplace_img, scale_factor=2, mode='bilinear', align_corners=False)
                laplace_imgs.append(laplace_img)
            laplacian_pyramid.append(torch.stack(laplace_imgs, dim=2))
        output = torch.cat(laplacian_pyramid, dim = 1)
        return output



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
