# -*- coding: utf-8 -*-
"""
Title: Cplx Layers with base class as base.py.
	
Created on Mon Feb 5 12:29:29 2021

@author: Ujjawal.K.Panchal
"""
#normal python imports.
import math
from typing import Iterable, Callable
from collections import namedtuple

#torch.
import torch
import torch.nn as nn

#cplxmodule import.
from cplxmodule import cplx
import cplxmodule.nn as cplxnn

from pyutils.enumutils import *

# t3nsor import.
import t3nsor
import tensorly as tl



from shnetutil.utils.torchutils import modelName
#relative imports.
from . import utils, activations, DCF
from . import tt_layer, cplxlayer

#Tensor Descriptor.
TDesc  = cplxlayer.TDesc	#make this available since most client do not import tt_layer
TTDesc = cplxlayer.TTDesc	#make this available since most client do not import tt_layer


class Complex2RealMethod(str, Enumbase):
	kMagSq 	= 'magsq'
	kMag 	= 'mag'
	kModulus= 'modulus'	#the modulus is another name for cplx.mag or is it? mck

class CplxComplex2Real(cplxlayer.CplxLayer, nn.Module):
	""" Complex to real by concating real|imag + Linear (2->1)  """
	def __init__(self, 
		in_features: int, out_features: int, 	#20 -> 10
		dtype = None
	):
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.realfc = torch.nn.Linear(in_features, out_features, dtype=dtype)
#		print(f"CplxComplex2Real")

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = torch.cat([x.real, x.imag], dim = 1)
		out = self.realfc(x)
		return out

class CplxComplex2mag(cplxlayer.CplxLayer, nn.Module):
	""" Complex to real as magnitude^2 dropping phase  """
	dispatch = {
		Complex2RealMethod.kMagSq: lambda x: utils.cplx_complex2mag(x).square(),
		Complex2RealMethod.kMag:   lambda x: utils.cplx_complex2mag(x),
		Complex2RealMethod.kModulus: lambda x: utils.cplx_complex2mag(x),
	}

	def __init__(self, 
		in_features: int, out_features: int, 
		dtype = None,
		square: Complex2RealMethod = Complex2RealMethod.kMagSq
	):
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.method = CplxComplex2mag.dispatch[square]
		self.square = square
#		print(f"CplxComplex2mag {square}")

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		#refer: https://en.wikipedia.org/wiki/Probability_amplitude
		#x_mag := |\Phi|^2$ or |\Phi|
		out = self.method(x)
		return out

	def __repr__(self):
		return f"CplxComplex2mag({self.square})"	

	def __str__(self):
		return f"CplxComplex2mag({self.square})"	


class BatchReshape(cplxlayer.CplxLayer, nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = x.reshape(x.shape[0], -1)
		return x

	def __repr__(self):
		return f"BatchReshape()"	

	def __str__(self):
		return f"BatchReshape()"

class CplxTensorLinear(cplxlayer.CplxLayer, nn.Module):
	""" base class for a tensor (tt|tcl) CplxLayer  """
	def __init__(self,
		in_factors:tuple,
		out_factors:tuple,
	):
		super().__init__()
		self.in_factors = in_factors
		self.out_factors = out_factors

class CplxSplitTTLayer(CplxTensorLinear, nn.Module):
	"""
	Tensor Train Layer with Fixed Ranks configuration. As shown in https://arxiv.org/abs/1905.10478
	Implementation of split parts taken from: https://github.com/vicontek/lrbtnn.

	Few questions to answer: 
		Q.1. What are infactors and outfactor formats?
		Q.2. Wht should be the predetermined rank values?
	---
	Args:
		1. in_factors := in factors is the number of factors in input.
		2. out_factors := output factors is the number of factors in output.
		3. ranks := predetermined ranks of the Core Tensors of the Tensor train.
		4. ein_string := einstein summation notation of the calculation of outer products of the core tensors; it determines the type of layer being implemented (convoution or linear).
			- convolution:= "nabcde, aoiv, bijw, cjkx, dkly, elpz".
			- linear:= "nab, aoix, bipy". (default)
		5. device := torch device on which the given layer should exist.

	"""
	@staticmethod
	def InOut2Linear(in_factors: tuple, out_factors: tuple):
		return tt_layer.InOut2Linear(in_factors, out_factors)

	@staticmethod
	def Linear2InOut(in_features: int, out_features: int, tt_factor_method: str = "range", tt_factor_range: tuple = (20, 30)):
		return tt_layer.Linear2InOut(in_features, out_features, tt_factor_method, tt_factor_range)

	@staticmethod
	def TTDesc2Linear(ttdesc: TDesc):
		return tt_layer.TTDesc2Linear(ttdesc)

	def __init__(self,
		kind:str = "tt",
		in_factors:tuple = (50, 25),
		out_factors:tuple = (20, 25),
		ranks:tuple = (8,),
		ein_string:str = 'nab, aoix, bipy',
		init_method = "tt-svd-1",
		split = True,
	):
		print(f"CplxSplitTTLayer {in_factors=}")
		assert(type(in_factors) == tuple)
		assert(type(out_factors) == tuple)
		assert(type(ranks) == tuple)
		assert(type(ein_string) == str)
		assert(type(init_method) == str)

		super().__init__(in_factors, out_factors)
		self.ranks = ranks
		self.ein_string = ein_string

		self.r_splitlayer = tt_layer.TTLayer(in_factors, out_factors, ranks, ein_string)
		self.i_splitlayer = tt_layer.TTLayer(in_factors, out_factors, ranks, ein_string)
		self.forward = self.split_tt_layer if split else self.real_tt_layer 	#what is .real_tt_layer? real passes only through the real part i.e. the normal layer. was used in the test where I had to run 1 split through the source repo.
		self.init_method = init_method
		self.init_methods = {			
			"tt-svd-1": self.initTTSVDWeights_,
			"tt-svd-2": self.initTTSVDWeightsHacked_,
			"cores-init": self.initCoreWeights_,
			"no-init": self.noop
		}
		return

	@classmethod
	def factory(cls, 
		desc = TDesc,
		split = True,
		init_method = "tt-svd-1",
	):
		return CplxSplitTTLayer(
			in_factors=desc.in_factors, 
			out_factors=desc.out_factors,
			ranks=desc.ranks,
			ein_string=desc.ein_string,
			split=split,
			init_method=init_method,
		)

	def initWeights(
		self,
		cplxInitWeights_,
		kind = "glorot",
		init_method = None,
	):
		"""
		tt core weight initializer.
		---
		Args:
			1. initWeights_: cplxmodule weight initializer.
			2. kind: init kind one of : "xavier", "glorot", "kaiming" or "he".
			3. init_method: method of weight init. Either one of:-
															a. "tt-svd-1": instantiate + initialize the layer, do tt-svd decomposition, then copy cores to TT-Layers (using to_tt_matrix way).
															b. "tt-svd-2": option "tt-svd" but in the hacked way which seems to provide higher performance.
															c. "cores-init": initialize each core using the given initWeights_ function.
															d. "noop": no initialization (random weights).
															e. None: use self.init_method.
			4. svd_source_layer (default: None): In case when using tt-svd method, the layer which is to be approximated.
		"""
		init_method = self.init_method if init_method is None else init_method
		assert (init_method in self.init_methods.keys()), f"{init_method} not in supported method {self.init_methods.keys()}."
		svd_source_layer = cplxlayer.CplxLinear(*tt_layer.InOut2Linear(self.in_factors, self.out_factors))
		self.init_methods[init_method](cplxInitWeights_, kind, svd_source_layer)
		return

	def noop(*args, **kwargs):
		"""
		do nothing.
		"""
		return

	def initCoreWeights_(
		self,
		cplxInitWeights_,
		kind,
		svd_source_layer,
	):
		"""
		Initialize Core Weights using the given weight strategy.
		---
		Args:
			1. cplxInitWeights_ : Initialization function that takes in a Cplx Tensor and assigns then weights.
			2. kind: one of "xavier", "glorot" or "kaiming", "he", whichever one is to be followed.
			3. svd_source_layer: Used to Copy over bias from.
		"""
		#1. Init weights according to initialization scheme.
		for rcore, icore in zip(self.r_splitlayer.cores, self.i_splitlayer.cores):
			cplxInitWeights_(cplx.Cplx(rcore, icore), kind = kind)
		#2. Copy over bias if required.
		self.copy_layer_bias_(svd_source_layer.bias)
		return

	def initTTSVDWeights_(
		self,
		cplxInitWeights_,
		kind,
		svd_source_layer,
		shape_permutation: Iterable = (1,0,3,2),
	):
		"""
		Initialize layer with `cplxInitWeights_` & approximate using TT-SVD. Copy core Weights to our CplxSplitTTSVD Core layer.
		---
		Args:
			1. cplxInitWeights_ : Initialization function that takes in a Cplx Tensor and assigns then weights.
			2. kind: one of "xavier", "glorot" or "kaiming", "he", whichever one is to be followed.
			3. svd_source_layer: The source layer for TT-SVD.
			4. shape_permutation: A set of indexes to permute the output (approximated TT-SVD'd cores) so that it corresponds to the same shape as the cores of the layer.
		"""
		#1. svd source layer initialization.
		svd_source_layer.initWeights(cplxInitWeights_, kind = kind)

		#2. svd approximation.
		tt_real = t3nsor.decompositions.to_tt_matrix(svd_source_layer.weight.real, shape = [self.in_factors, self.out_factors],  max_tt_rank = self.ranks)
		tt_imag = t3nsor.decompositions.to_tt_matrix(svd_source_layer.weight.imag, shape = [self.in_factors, self.out_factors], max_tt_rank = self.ranks)

		#4. copy over tt params to our cores.
		cplxList = []
		for rcore, icore in zip(tt_real.tt_cores, tt_imag.tt_cores):
			cplxList.append(cplx.Cplx(rcore, icore))
		self.copy_tt_cores_(cplxList, shape_permutation)

		#5. copy bias if required.
		if svd_source_layer.bias is not None:
			self.copy_layer_bias_(svd_source_layer.bias)
		return

	def copy_tt_cores_(self, src: Iterable, permutation: Iterable):
		"""
		copy tt cores from src tt cores (a list of cplx.Cplx cores) to self.
		---
		Args:
			1. src: An Iterable of cplx.Cplx cores.
			2. permutation: list of integers representing axes to use when permuting.
		"""
		#real.
		assert (len(src) == len(self.r_splitlayer.cores)), "Error<!>: Length of source not the same as length of destination."	
		for i in range(len(self.r_splitlayer.cores)):
			#print(f"dst: {reduce(mul, dst[i].shape, 1)}, src: {reduce(mul, src[i].shape, 1)}")
			self.r_splitlayer.cores[i] = torch.nn.Parameter(src[i].real.permute(*permutation))
			self.i_splitlayer.cores[i] = torch.nn.Parameter(src[i].imag.permute(*permutation))
		return

	def initTTSVDWeightsHacked_(
		self,
		cplxInitWeights_,
		kind,
		svd_source_layer,
	):
		"""
		Initialize layer with `cplxInitWeights_` & approximate using TT-SVD. Copy core Weights to our CplxSplitTTSVD Core layer.
		---
		Args:
			1. cplxInitWeights_ : Initialization function that takes in a Cplx Tensor and assigns then weights.
			2. kind: one of "xavier", "glorot" or "kaiming", "he", whichever one is to be followed.
			3. svd_source_layer: The source layer for TT-SVD.
		"""
		#1. svd source layer initialization.
		svd_source_layer.initWeights(cplxInitWeights_, kind = kind)
		layer_shape = svd_source_layer.weight.shape

		#2. preparations for shape morphing.
		operation_list = [2,] + [1 for i in range(len(layer_shape) - 2)] + [0.5,] #Hack.
		morph_shape = [int(shape_int * op_int) for shape_int, op_int in zip(layer_shape,operation_list)] #Hack.

		#3. svd approximation.
		tt_real = t3nsor.decompositions.to_tt_tensor(svd_source_layer.weight.real.reshape(*morph_shape), max_tt_rank = self.ranks)
		tt_imag = t3nsor.decompositions.to_tt_tensor(svd_source_layer.weight.imag.reshape(*morph_shape), max_tt_rank = self.ranks)

		#4. copy over tt params to our cores.
		cplxList = []
		for rcore, icore in zip(tt_real.tt_cores, tt_imag.tt_cores):
			cplxList.append(cplx.Cplx(rcore, icore))
		self.copy_tt_coresHacked_(cplxList)

		#5. copy over bias if required.
		if svd_source_layer.bias is not None:
			self.copy_layer_bias_(svd_source_layer.bias)
		return

	def copy_tt_coresHacked_(self, src):
		"""
		copy tt cores from src tt cores (a list of cplx.Cplx cores) to self.
		---
		Args:
			1. src: An Iterable of cplx.Cplx cores.  
		"""
		#real.
		assert (len(src) == len(self.r_splitlayer.cores)), "Error<!>: Length of source not the same as length of destination."	
		for i in range(len(self.r_splitlayer.cores)):
			#print(f"dst: {reduce(mul, dst[i].shape, 1)}, src: {reduce(mul, src[i].shape, 1)}")
			self.r_splitlayer.cores[i] = torch.nn.Parameter(src[i].real.reshape(*self.r_splitlayer.cores[i].shape))
			self.i_splitlayer.cores[i] = torch.nn.Parameter(src[i].imag.reshape(*self.i_splitlayer.cores[i].shape))
		return

	def copy_layer_bias_(self, src_bias: cplx.Cplx):
		"""
		Copy over bias from the given `cplx.Cplx` tensor.
		---
		Args:
			1. src_bias: `cplx.Cplx` bias which is to be copied over.
		"""
		self.r_splitlayer.bias, self.i_splitlayer.bias = src_bias.real, src_bias.imag
		return

	def getConfigStr(self):
		""" define a nice condense pretty printing for ourself. Called from the model's getModelConfigStr.
		"""
		r = self.r_splitlayer.getConfigStr()
		i = self.i_splitlayer.getConfigStr()
		return f"CplxSplitTTLayer({r}, {i})"

	def split_tt_layer(self, x):
		return cplx.Cplx(self.r_splitlayer(x.real), self.i_splitlayer(x.imag))

	def real_tt_layer(self, x):
		return self.r_splitlayer(x)

	def forward(self, x):
		return split_tt_layer(x)

	@property
	def weight(self):
		self.real_params = torch.cat([torch.tensor(core).reshape(-1,1) for core in self.r_splitlayer.cores])
		self.complex_params = torch.cat([torch.tensor(core).reshape(-1, 1) for core in self.i_splitlayer.cores])
		return cplx.Cplx(self.real_params, self.complex_params)

def CplxTensorLayerFactory(tdesc:TDesc):
	assert(isinstance(tdesc, cplxlayer.TDesc))
	factories = {
		"tt":  CplxSplitTTLayer.factory,
	}
	return factories[tdesc.kind](tdesc)

class CplxSplitDCF(cplxlayer.CplxLayer, nn.Module):
	"""
	Convolutions decomposed with pre-fixed Fourier Bessel bases.
	---
	Args:
		All input arguments are the same as that of the Regular DCF. Please refer args of: `shnetutil.cplx.DCF.conv_DCF`. Some arguments are repeated below with greater elaboration.
		1. mode (repeated elaboration): str:
		   - mode0 := project input to kernel space and then perform conv.
		   - mod0_1:= conv input with bases wrt. group properties then convolve input with filters space.
		   - mode1 := project filter to input space and then perform convolution.
	"""
	def __init__(
		self, in_channels, out_channels,
		kernel_size, stride=1, padding=0, 
		num_bases=3, #Num bases has been changed to 3 because the default `-1` (as in DCF source) was not working. Giving Error: RuntimeError: Trying to create tensor with negative dimension -20: [30, -20, 1, 1]
		bias=True,  bases_grad=False,
		dilation=1,
		initializer='FB', #FB stands for Fourier Bessel bases. We can change it to Shearlet Bases once we approximate them. 
		mode='mode1',
		bases_drop = 0.1
	):
		super().__init__()

		self.split_DCF_real = DCF.Conv_DCF(
				in_channels, out_channels, kernel_size,
				stride, padding, num_bases,
				bias,  bases_grad,
				dilation, initializer,
				mode, bases_drop
			)
		self.split_DCF_imag = DCF.Conv_DCF(
				in_channels, out_channels, kernel_size,
				stride, padding, num_bases,
				bias,  bases_grad,
				dilation, initializer,
				mode, bases_drop
			)
	
	def initWeights(
		self,
		cplxInitWeights_,
		kind = "glorot",
	):
		pass
		
	def getConfigStr(self):
		""" define a nice condense pretty printing for ourself. Called from the model's getModelConfigStr.
		"""
		modelname = modelName(self)
		in_channels  = self.split_DCF_real.in_channels
		out_channels = self.split_DCF_real.out_channels
		kernel_size	 = self.split_DCF_real.kernel_size
		num_bases = self.split_DCF_real.num_bases
		return f"{modelname}({in_channels}, {out_channels}, kernel_size={kernel_size}, num_bases={num_bases})"

	@property
	def weight(self):
		return cplx.Cplx(self.split_DCF_real.weight, self.split_DCF_imag.weight)

	@weight.setter #Not used by cplxmodule for weight init but we can choose to.
	def weight(self, values):
		...

	def split_DCF(self, x):
		return cplx.Cplx(self.split_DCF_real(x.real), self.split_DCF_imag(x.imag))
		
	def forward(self, x):
		return self.split_DCF(x)

if __name__ == "__main__":
	...