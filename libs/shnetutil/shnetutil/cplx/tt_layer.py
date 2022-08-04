"""
Title: tensor layers.

Description: These are Split CVnns layers for TT. 

Created on Thur Jul 29 14:48:29 2021

@author: Ujjawal.K.Panchal, Manny Ko.
"""
import torch
import torch.nn as nn
import numpy as np
import math
from typing import Union

from shnetutil.cplx import cplxlayer 	#TODO: make TTLayer inherit from this base-class



#supported methods in Linear2InOut()
tt_factor_methods={"gcd", "range"}

def InOut2Linear(in_factors: tuple, out_factors: tuple):
	return (in_factors[0]*in_factors[1], out_factors[0]*out_factors[1])

def get_factors(n1, n2, low, high):    
    return [i for i in range(low+1, high) if n1 % i == 0 and n2 % i == 0]

def Linear2InOut(in_features: int, out_features: int, method: str = "range", factor_range: tuple = (20, 30)):
	if method == "gcd": #use the greatest common divisor as the number of features.
		val = np.gcd.reduce([in_features, out_features])
	elif method == "range": #use a factor in the given range.
		val = min(get_factors(in_features, out_features, factor_range[0], factor_range[1]))
	else:
		raise Exception(f"method '{method}' not supported. Supported methods are: 'range', 'gcd'.")
	quotient_in, quotient_out = in_features // val, out_features // val

	print(f"({quotient_in}, {val}) ({quotient_out}, {val})")
	return ((quotient_in, val), (quotient_out, val))

def TTDesc2Linear(ttdesc: cplxlayer.TDesc):
	return InOut2Linear(ttdesc.in_factors, ttdesc.out_factors)
# 
# https://github.com/vicontek/lrbtnn
#
class TTLayer(nn.Module, cplxlayer.CplxLayer):	# Adopted from lrbtnn/tt_model.py 
	def __init__(self, 
		in_factors: Union[list, np.ndarray],
		out_factors: Union[list, np.ndarray],
		ranks: Union[list, np.ndarray],
		ein_string: str,  		#control string for torch.einsum() https://pytorch.org/docs/stable/generated/torch.einsum.html
		bias: bool=True
	):
		super().__init__()

		self.in_factors = np.asarray(in_factors)
		self.out_factors = np.asarray(out_factors)
		self.ranks = np.asarray(ranks)
		self.ein_string = ein_string
		#self.bias = None 
		assert len(in_factors) == len(out_factors) == len(ranks) + 1, 'Input factorization should match output factorization and should be equal to len(ranks) - 1'
#         assert len(ranks) == 4, 'Now we consider particular factorization for given dataset'

		self.cores = nn.ParameterList([nn.Parameter(torch.randn(in_factors[0], 1, ranks[0], out_factors[0], ) * 0.8)])
		for i in range(1, len(in_factors) - 1):
			self.cores.append(nn.Parameter(torch.randn(in_factors[i], ranks[i-1], ranks[i], out_factors[i],) * 0.1))
		self.cores.append(nn.Parameter(torch.randn(in_factors[-1], ranks[-1], 1, out_factors[-1], ) * 0.8))

		self.init_bias(bias)

	@classmethod
	def factory(self, recipe=cplxlayer.TDesc):
		assert(type(recipe) is cplxlayer.TDesc)
		assert(recipe.kind == "tt")
		
		return TTLayer(
			in_factors=recipe.in_factors, 
			out_factors=recipe.out_factors,
			ranks=recipe.ranks,
			ein_string=recipe.ein_string,
		)	

	def init_bias(self, bias: bool=True):
		if bias:	#TODO: move to initWeights
			fan_out = np.prod(self.out_factors)
			fan_in = np.prod(self.in_factors)
			self.bias = nn.Parameter(torch.Tensor(fan_out))
			bound = 1 / math.sqrt(fan_in)		#from cplxmodule.nn.module.linear - line 53
			nn.init.uniform_(self.bias, a=-bound, b=bound)
			#nn.init.normal_(self.bias, mean=0, std=1./math.sqrt(fan_out))
			#nn.init.zeros_(self.bias)		
		else:
			self.register_parameter('bias', None)

	def getConfigStr(self):
		""" define a nice condense pretty printing for ourself. Called from the model's getModelConfigStr.
		"""
		model_config_str = f"TTLayer(in{self.in_factors} out{self.out_factors} rank{self.ranks})"
		return  model_config_str

	def forward(self, x):
		reshaped_input = x.reshape(-1, *self.in_factors)
#         print('reshaped_input', reshaped_input.shape)
		# in the einsum below, n stands for index of sample in the batch,
		# abcde - indices corresponding to h1, h2, hw, w1, w2 modes
		# o, i, j, k, l, p - indices corresponding to the 4 tensor train ranks
		# v, w, x, y, z - indices corresponding to o1, o2, o3, o4, o5
		result = torch.einsum(      #https://pytorch.org/docs/stable/generated/torch.einsum.html
			self.ein_string,
			reshaped_input, *self.cores
		)
		result = result.reshape(-1, np.prod(self.out_factors))

		if self.bias is not None:
			# fused op is marginally faster
			result = torch.add(self.bias, result)
		return result
	
#     def parameters(self):
#         return self.cores