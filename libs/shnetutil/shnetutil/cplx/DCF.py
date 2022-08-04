import datetime
import os, sys
import random
import argparse
import numpy as np

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
from torch import nn

import torch.nn.functional as F
import pdb

import time

from torch.nn.parameter import Parameter
import math

from ..bases.fourier_bessel import *

class Conv_DCF(nn.Module):
	r"""Pytorch implementation for 2D DCF Convolution operation.
	Link to ICML paper:
	https://arxiv.org/pdf/1802.04145.pdf


	Args:
		in_channels (int): Number of channels in the input image
		out_channels (int): Number of channels produced by the convolution
		kernel_size (int): Size of the convolving kernel
		stride (int, optional): Stride of the convolution. Default: 1
		padding (int, optional): Zero-padding added to both sides of
			the input. Default: 0
		num_bases (int, optional): Number of basis elements for decomposition.
		bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
		mode (optional): Either `mode0` for two-conv or `mode1` for reconstruction + conv.

	Shape:
		- Input: :math:`(N, C_{in}, H_{in}, W_{in})`
		- Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

		  .. math::
			  H_{out} = \left\lfloor\frac{H_{in}  + 2 * \text{padding}[0] - \text{dilation}[0]
						* (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

			  W_{out} = \left\lfloor\frac{W_{in}  + 2 * \text{padding}[1] - \text{dilation}[1]
						* (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

	Attributes:
		weight (Tensor): the learnable weights of the module of shape
						 (out_channels, in_channels, kernel_size, kernel_size)
		bias (Tensor):   the learnable bias of the module of shape (out_channels)

	Examples::
		
		>>> from DCF import *
		>>> m = Conv_DCF(16, 33, 3, stride=2)
		>>> input = torch.randn(20, 16, 50)
		>>> output = m(input)

	"""
	__constants__ = ['kernel_size', 'stride', 'padding', 'num_bases',
					 'bases_grad', 'mode']
	__name__ = "Conv_DCF"
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
		num_bases= -1, bias=True,  bases_grad=False, dilation=1, initializer='FB', 
		mode='mode1', bases_drop = 0.1):
		super(Conv_DCF, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		# self.edge = (kernel_size-1)/2
		self.stride = stride
		self.padding = padding
		self.kernel_list = {}
		self.num_bases = num_bases
		assert mode in ['mode0', 'mode1', 'mode0_1'], 'Only mode0 and mode1 are available at this moment.'
		self.mode = mode
		self.bases_grad = bases_grad
		self.dilation = dilation
		# self.bases_drop = bases_drop

		assert initializer in ['FB', 'random'], 'Initializer should be either FB or random, other methods are not implemented yet'

		# build bases
		if initializer == 'FB':
			if kernel_size % 2 == 0:
				raise Exception('Kernel size for FB initialization only supports odd number for now.')
			base_np, _ = initialize_spatial_bases_FB(kernel_size=self.kernel_size, 
								Ntheta=8, num_bases=self.num_bases)
			# for kernel_size=3, with shape [6, 1, 3, 3], as [chn_out=num_bases, 1, k_h, k_w]
			base_np = np.array(np.expand_dims(base_np.transpose(2,0,1), 1), np.float32)

		else:
			if num_bases <= 0:
				raise Exception('Number of basis elements must be positive when initialized randomly.')
			base_np = np.random.randn(num_bases, 1, kernel_size, kernel_size)

		if bases_grad:
			self.bases = Parameter(torch.tensor(base_np), requires_grad=bases_grad)
			# self.bases.data.normal_(0, 1.0)
			#  self.bases.data.uniform_(-1, 1)
		else:
			self.register_buffer('bases', torch.tensor(base_np, requires_grad=False).float())

		# set parameters as coefficients of bases, with shape [chn_out, num_bases*chn_in, 1, 1]
		self.weight = Parameter(torch.Tensor(
				out_channels, in_channels*num_bases, 1, 1))
		if bias:
			self.bias = Parameter(torch.Tensor(out_channels))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()
		
		# print(self.weight.shape)
		# print(self.bases.shape)
		
		
		## bases drop out to impose regularization
		# self.bases_droput_layer = nn.Dropout2d(p=self.bases_drop)

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		# self.weight.data.uniform_(-stdv, stdv)
		self.weight.data.normal_(0, stdv) #Normal works better, working on more robust initializations
		if self.bias is not None:
			# self.bias.data.uniform_(-stdv, stdv)
			self.bias.data.zero_()

	def forward_mode0(self, input):
		FE_SIZE = input.size()
		feature_list = []
		# reshape input as [batch_size*chn_in, 1, H, W]
		input = input.view(FE_SIZE[0]*FE_SIZE[1], 1, FE_SIZE[2], FE_SIZE[3])

		# conv input with bases:
		# [batch_size*chn_in, 1, H, w] * [chn_out=num_bases, 1, k_h, k_w]
		# = [batch_size*chn_in, num_bases, H', W']
		feature = F.conv2d(input, self.bases,
			None, self.stride, self.padding, dilation=self.dilation)

		# with shape [batch_size, chn_in*num_bases, H', W']
		feature = feature.view(
			FE_SIZE[0], FE_SIZE[1]*self.num_bases, 
			int((FE_SIZE[2]-self.kernel_size+2*self.padding)/self.stride+1), 
			int((FE_SIZE[3]-self.kernel_size+2*self.padding)/self.stride+1))

		# conv the features convolved with bases with coefficients
		# [batch_size*chn_in, num_bases, H', W'] * [chn_out, num_bases*chn_in, 1, 1]
		# = [batch_size, chn_out, H', w']
		feature_out = F.conv2d(feature, self.weight, self.bias, 1, 0)  # stride 1, padding 0

		return feature_out
	
	def forward_mode0_1(self, input):
		"""
		  another implementation of two-step convolution
		"""       
		# with input shape [batch_size, chn_in, H, W]  
		FE_SIZE = input.size()

		# conv input with bases, using group property,
		# [batch_size, chn_in, H, w] * [num_bases, 1, k_h, k_w]
		# = [batch_size, chn_in*num_bases, H', W']
		bases_w = self.bases.repeat(FE_SIZE[1], 1, 1, 1)
		feature = F.conv2d(input, bases_w, None, stride=self.stride, padding=self.padding, 
								dilation=self.dilation, groups=FE_SIZE[1])

		# conv the features convolved with bases with coefficients
		# [batch_size*chn_in, num_bases, H', W'] * [chn_out, num_bases*chn_in, 1, 1]
		# = [batch_size, chn_out, H', w']
		feature_out = F.conv2d(feature, self.weight, self.bias, 1, 0)  # stride 1, padding 0

		return feature_out

	def forward_mode1(self, input):
		# directly compute the reconstructed convolution kernel
		# [chn_out*chn_in, num_bases] × [num_bases, 1, k_h, k_w]
		# = [chn_out, chn_in, k_h, k_w]
		bases = self.bases.permute(1,0,2,3)
		# bases = self.bases_droput_layer(bases).squeeze()

		weight= self.weight.view(self.out_channels*self.in_channels, self.num_bases)
		bases = bases.view(self.num_bases, self.kernel_size*self.kernel_size)

		rec_kernel = torch.mm(weight, bases).view(self.out_channels,
									self.in_channels, self.kernel_size, self.kernel_size)

		# conv
		feature = F.conv2d(input, rec_kernel,
			self.bias, self.stride, self.padding, dilation=self.dilation)
		
		return feature

	def forward(self, input):
		if self.mode == 'mode1':
			return self.forward_mode1(input)
		elif self.mode == 'mode0':
			return self.forward_mode0(input)
		else:
			return self.forward_mode0_1(input)

	def extra_repr(self):
		return 'kernel_size={kernel_size}, stride={stride}, padding={padding}, num_bases={num_bases}' \
			', bases_grad={bases_grad}, mode={mode}'.format(**self.__dict__)