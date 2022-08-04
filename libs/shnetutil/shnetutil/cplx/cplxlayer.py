"""
Title: tensor layers.

Description: These are some of our layer templates and some basic cplx layers. Many are wrappers around cplxmodule. 

Created on Thur Jul 29 14:48:29 2021

@author: Ujjawal.K.Panchal, Manny Ko.
"""
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Callable, List, Tuple, Optional, Union

import torch.nn as nn
import numpy as np

from cplxmodule import cplx
import cplxmodule.nn as cplxnn

#import from our package(s)
from shnetutil.utils.torchutils import modelName #TODO: check if this is allowed, change with relative imports.


class TDesc():
	""" Descriptor for a Tensor-linear layer.
 	"""
	def __init__(self,
		kind:str,				#tt|tcl
		in_factors:Union[list, np.ndarray],
		out_factors:Union[list, np.ndarray],
	):
		assert(kind in ("tt", "tcl"))
		self.kind = kind
		self.in_factors = in_factors
		self.out_factors = out_factors
		return

	def __iter__(self):
		return iter((self.kind, self.in_factors, self.out_factors))

class TTDesc(TDesc):
	""" Descriptor for a Tensor-Train linear layer.
 	"""
	def __init__(self,
		kind:str,				#tt|tcl
		in_factors:Union[list, np.ndarray],
		out_factors:Union[list, np.ndarray],
		ranks:Optional[tuple] = None,
		ein_string:Optional[str] = None,
		tt_init:Optional[str] = None,
	):
		assert(kind in ("tt", "tcl"))
		super().__init__(kind, in_factors, out_factors)
		self.ranks = ranks
		self.ein_string = ein_string
		self.tt_init = tt_init
		return

	def __iter__(self):
		return iter((self.kind, self.in_factors, self.out_factors, self.ranks, self.ein_string, self.tt_init))

if False:
	def defaultInitW(model: CplxLayer):
	    for weight in model.parameters():
	        nn.init.normal_(weight.data, mean, std)

	    if model.bias is not None:
	        nn.init.zeros_(model.bias)

class CplxLayer(ABC):
	"""
	Abstract base-class for a 'cplx' based layer.
	---
	Args:

	"""
	def __init__(self,
		initW: Callable = cplxnn.init.cplx_trabelsi_standard_,		#cplx_trabelsi_standard_|cplx_trabelsi_independent_
		initkind:str = "xavier",
	):
		super().__init__()
		self._initWeights = initW
		self.initkind = initkind
		#self.initWeights()		#TODO: should we call it here?
		return

	def initWeights(self,
		**kwargs			#to ignore extra params passed during other inits.
	):
		"""
		weight initializer. Give derived class a chance to override this.
		---
		Args:
			None
		"""
		return self._initWeights(self.weight)

	def getConfigStr(self):
		""" define a nice condense pretty printing for ourself. Called from the model's getModelConfigStr.
		"""
		config_str = modelName(self)
		return config_str

	@abstractmethod	
	def forward(self, x):
		pass

	@property
	def weight(self):
		self._weight

	#@property
	#def bias(self):	#we cannot define this here since it conflicts with nn.Module which expects a .bias too
		#return self._bias

class CplxLinear(CplxLayer, nn.Module):
	""" A wrapper around cplxnn's CplxLinear to conform to CplxLayer ABC """
	def __init__(self,
		in_features: tuple, out_features: tuple,
		initW: Callable = cplxnn.init.cplx_trabelsi_standard_,		#cplx_trabelsi_standard_|cplx_trabelsi_independent_
		initkind: str = "xavier",
		bias: bool = True
	):
		super().__init__(initW=initW, initkind=initkind)
		self.in_features = in_features
		self.out_features = out_features
		self.linear = cplxnn.modules.CplxLinear(self.in_features, self.out_features, bias=bias)
 
	@property
	def weight(self):
		return self.linear.weight
	
	@property
	def bias(self):
		return self.linear.bias

	@weight.setter #Not used by cplxmodule for weight init but we can choose to.
	def weight(self, values):
		self.linear.weight = values

	def initWeights(self, initW = None, kind = "xavier", **kwargs): #to ignore extra params passed during other inits.
		initW = self._initWeights if initW is None else initW
		initW(self.linear.weight, kind = kind)
		return

	def getConfigStr(self):
		""" define a nice condense pretty printing for ourself. Called from the model's getModelConfigStr.
		"""
		config_str = modelName(self)
		return f"{config_str}{tuple(self.weight.shape)}"

	def forward(self, input):
		return self.linear.forward(input)


class CplxConv2d(CplxLayer, nn.Module):
	def __init__(self,
		in_channels,
		out_channels,
		kernel_size,
		stride=1,
		padding=0,
		dilation=1,
		groups=1,
		bias=True,
		padding_mode="zeros"
	):
		super().__init__()
		self.conv2d = cplxnn.modules.CplxConv2d(
			in_channels, out_channels, kernel_size, stride, padding,
			dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode
		)

	@property
	def weight(self):
		return self.conv2d.weight
	
	@property
	def bias(self):
		return self.conv2d.bias

	@weight.setter #Not used by cplxmodule for weight init but we can choose to.
	def weight(self, values):
		self.conv2d.weight = values

	def initWeights(self, initW = None, kind = "xavier", **kwargs): #to ignore extra params passed during other inits.
		initW = self._initWeights if initW is None else initW
		initW(self.conv2d.weight, kind = kind)
		return

	def getConfigStr(self):
		""" define a nice condense pretty printing for ourself. Called from the model's getModelConfigStr.
		"""
		config_str = modelName(self)
		in_channels, out_channels, kernel_size, stride = self.conv2d.in_channels, self.conv2d.out_channels, self.conv2d.kernel_size, self.conv2d.stride
		return f"{config_str}{(in_channels, out_channels, kernel_size[0], stride[0])}"

	def forward(self, input):
		return self.conv2d.forward(input)


class MiscLayer(ABC):
	def __init__(self, name, layer):
		super().__init__()
		self.name = name
		self.layer = layer

	def getConfigStr(self):
		return self.name

	def forward(self, input):
		return self.layer(input)

class Activation(MiscLayer, nn.Module):
	def __init__(self, name, layer):
		super().__init__(name, layer)

class Pool(MiscLayer, nn.Module):
	def __init__(self, name, layer):
		super().__init__(name, layer)
