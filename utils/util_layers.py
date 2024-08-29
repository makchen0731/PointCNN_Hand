import torch

import torch.nn as nn
from typing import Callable, Union, Tuple
from torch import cuda

import numpy as np

class Depthwise_conv(nn.Module):
    
    def __init__(self, in_channels, kernel_size ,depth_multiplier = 1, with_bn = True,activation = True):
        
        super(Depthwise_conv,self).__init__()
        self.conv =  nn.Conv2d(in_channels,in_channels* depth_multiplier,kernel_size = kernel_size,groups = in_channels)
        
        self.with_bn = with_bn
        self.activation = activation
        
        
        # self.tanh = nn.Tanh()
        # self.ReLU= nn.ReLU()
        self.ELU= nn.ELU()
        
        self.bn = nn.BatchNorm2d(in_channels* depth_multiplier, momentum = 0.99) if with_bn else None
        
    def forward(self, x):
        
        # print(self.conv)
        x = self.conv(x)
        if self.activation:
            x = self.ELU(x)
            # x = self.ReLU(x)
            # x = self.tanh(x)
        
        if self.with_bn:
            x = self.bn(x)
        
        return x

def EndChannels(f, make_contiguous = False):
    """ Class decorator to apply 2D convolution along end channels. """

    class WrappedLayer(nn.Module):

        def __init__(self):
            super(WrappedLayer, self).__init__()
            self.f = f

        def forward(self, x):
            x = x.permute(0,3,1,2)
            x = self.f(x)
            x = x.permute(0,2,3,1)
            return x

    return WrappedLayer()

class Dense(nn.Module):
    """
    Single layer perceptron with optional activation, batch normalization, and dropout.
    """

    def __init__(self, in_features , out_features ,
                 drop_rate = 0, with_bn = True,
                   
                    # activation  = nn.Tanh()
                    # activation = nn.ReLU()
                    activation = nn.ELU()
                ) :
        """
        :param in_features: Length of input featuers (last dimension).
        :param out_features: Length of output features (last dimension).
        :param drop_rate: Drop rate to be applied after activation.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(Dense, self).__init__()

        self.linear = nn.Linear(in_features, out_features)

        
        ###---BN Test---###
        # if with_bn:
        #     self.bn = nn.BatchNorm1d(out_features, momentum = 0.9) if with_bn else None
        # else:
        #     self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else None

        # print(drop_rate)

        self.activation = activation
        # self.bn = LayerNorm(out_features) if with_bn else None
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else None
        self.bn = nn.BatchNorm1d(out_features) if with_bn else None
    
    
    def forward(self, x) :
        """
        :param x: Any input tensor that can be input into nn.Linear.
        :return: Tensor with linear layer and optional activation, batchnorm,
        and dropout applied.
        """


        
        x = self.linear(x)
        
        if self.activation:
            x = self.activation(x)
            

        if self.bn:
            B = np.shape(x)[0]
            P = np.shape(x)[1]
            C = np.shape(x)[2]
        
            x = x.view(-1,C)
            x = self.bn(x).view(B,P,C)
        
        
        if self.drop:
            x = self.drop(x)
        
        
        return x

class Conv(nn.Module):
    """
    2D convolutional layer with optional activation and batch normalization.
    """

    def __init__(self, in_channels , out_channels ,
                 kernel_size, with_bn = True,
                    # activation  = nn.Tanh()
                    # activation =  nn.ReLU()
                    activation = nn.ELU()
                ) :
        """
        :param in_channels: Length of input featuers (first dimension).
        :param out_channels: Length of output features (first dimension).
        :param kernel_size: Size of convolutional kernel.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias = not with_bn)
        self.activation = activation
        self.bn = nn.BatchNorm2d(out_channels, momentum = 0.99) if with_bn else None

    def forward(self, x ):
        """
        :param x: Any input tensor that can be input into nn.Conv2d.
        :return: Tensor with convolutional layer and optional activation and batchnorm applied.
        """
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x)
        return x

class SepConv(nn.Module):
    """ Depthwise separable convolution with optional activation and batch normalization"""

    def __init__(self, in_channels , out_channels , kernel_size ,
                 depth_multiplier = 1, with_bn = True,
                    # activation  = nn.Tanh()
                    # activation =  nn.ReLU()
                    activation = nn.ELU()
                 ) :
        """
        :param in_channels: Length of input featuers (first dimension).
        :param out_channels: Length of output features (first dimension).
        :param kernel_size: Size of convolutional kernel.
        :depth_multiplier: Depth multiplier for middle part of separable convolution.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(SepConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * depth_multiplier, kernel_size, groups = in_channels),
            nn.Conv2d(in_channels * depth_multiplier, out_channels, 1, bias = not with_bn)
        )

        self.activation = activation
        self.bn = nn.BatchNorm2d(out_channels, momentum = 0.99) if with_bn else None

    def forward(self, x ):
        """
        :param x: Any input tensor that can be input into nn.Conv2d.
        :return: Tensor with depthwise separable convolutional layer and
        optional activation and batchnorm applied.
        """
        
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x)
        return x

# class LayerNorm(nn.Module):
#     """
#     Batch Normalization over ONLY the mini-batch layer (suitable for nn.Linear layers).
#     """

#     def __init__(self, N : int, dim : int, *args, **kwargs) -> None:
#         """
#         :param N: Batch size.
#         :param D: Dimensions.
#         """
#         super(LayerNorm, self).__init__()
#         if dim == 1:
#             self.bn = nn.BatchNorm1d(N, *args, **kwargs)
#         elif dim == 2:
#             self.bn = nn.BatchNorm2d(N, *args, **kwargs)
#         elif dim == 3:
#             self.bn = nn.BatchNorm3d(N, *args, **kwargs)
#         else:
#             raise ValueError("Dimensionality %i not supported" % dim)

#         self.forward = lambda x: self.bn(x.unsqueeze(0)).squeeze(0)
