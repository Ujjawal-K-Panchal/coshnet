# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from cplxmodule import nn as cplxnn
from cplxmodule import cplx

def MagRelu(z):
	"""
	Apply ReLU to Magnitude and leave phase alone.
	Assuming Magnitude Channels are stacked before phase channels.
	"""
	num_channels = z.shape[1] //2
	new_mag_maps = F.relu(z[:, 0:num_channels, :, :])
	new_phase_maps = z[:,num_channels:(2*num_channels), :, :]
	newz = torch.cat((new_mag_maps, new_phase_maps), dim = 1) * z 
	#z[:,0:num_channels, :, :] = F.relu(z[:, 0:num_channels, :, :])
	return z

def cardioid(z):
	"""
	Cardioid activation function. newZ = (1 + cos(phase(z))) * z
	"""
	#1. get phase angles.
	num_channels = z.shape[1] // 2 #// to make sure it is int. although it always should be. (num complex channels = num real channels).
	theta = torch.atan2(z[:, num_channels:(2*num_channels), :, :], z[:, 0:num_channels, :, :])
	
	#2. apply cardioid on Z.
	newz = torch.cat(((1 + torch.cos(theta)), (1 + torch.cos(theta))), dim = 1) * z
	return newz
		
def cplx_softmax(z, dim = 0):
    """
    Softmax function extended to complex domain as follows:
    
    [Scardapane et al. 2018]’s formulation:
	softmax n (h) = exp(R{h_n }^2 + I{h_n }^2) / \sum_{t = 1}^C exp(R{h_t}^2 + I{h_t}^2)
	
	where n, c ∈ [1, C ].
    """
    r, i = z.real, z.imag
    term = torch.exp(r**2 + i** 2)
    sumterm = torch.sum(term, dim = dim).unsqueeze(dim)
    return Cplx(torch.div(term, sumterm))

if __name__ == "__main__":
	...