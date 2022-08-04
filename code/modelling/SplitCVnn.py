"""
Title: Split CVnn modules.

Description: These are Split CVnns motivated from Trabelsi et al., 2018.

Created on Sun Aug 16 17:44:29 2020

@author: Ujjawal.K.Panchal, Manny Ko.
"""
from functools import partial, reduce
from typing import Callable, List, Tuple, Optional, Union
from collections import OrderedDict
from operator import mul
from inspect import ismethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from cplxmodule import cplx
import cplxmodule.nn as cplxnn

from shnetutil.cplx import utils as coshutils
from shnetutil.cplx import dispatcher as dispatcher
from shnetutil.cplx import CVnn_base as CVnn_base
from shnetutil.cplx.layers import CplxSplitTTLayer, BatchReshape
from shnetutil.cplx import layers
import shnetutil.cplx.cplxlayer as cplxlayer

from shnetutil.utils import trace, torchutils
from shnetutil.utils.torchutils import modelName
from pyutils.classutils import method_exists



def isFC(key:str, layer):	
	""" Predicate for 'layer' being a FC type linear layer.	
		#TODO: do a proper check for a real 'fc' 
	"""
	return ("fc" in key) and method_exists(layer, 'initWeights')

def isCONV(key:str, layer):
	""" Predicate for 'layer' being a conv type layer.	
		#TODO: do a proper check for a real 'fc' 
	"""
	result = ("conv" == key[0:4]) and method_exists(layer, 'initWeights')
	return result

class CplxMLP(CVnn_base.CVnn_Base):
	""" Complex-valued MLP - i.e. nn.Sequential of 1 or more CplxLinear
	"""
	def __init__(self,
		config: List[tuple] = [(5 * 5 * 50, 500), (500, 10), (20, 10)], 
		linear 	= cplxlayer.CplxLinear, 
		initW: Callable	= cplxnn.init.cplx_trabelsi_standard_,
		initkind: str = "xavier",	#xavier|kaiming
		activation = cplxnn.CplxToCplx[nn.ReLU](),
		tracectx: Optional[trace.TraceContext] = None,
		probtype: str = 'real',
	):
		super().__init__(
			conv2d = None,
			linear = linear, 
			initW = initW,
			initkind = initkind,
			tracectx = tracectx,
			probtype = probtype,
			modelname = "CplxMLP",
		)
		self.config = config
		self.dict = None

		#assert(isinstance(activation_function, nn.Module))
		self.activation = activation
		self.forward_count = 0
		self.flatten_added = False

	def addBatchReshape(self, fc_creator):
		if not self.flatten_added: #for the first CplxLinear, put flattener.
			self.batchreshape = BatchReshape() #Flatten.
			self.addLayer("bs", self.batchreshape)
			self.flatten_added = True
		return

	def defineModel(self, device='cuda'):
		config = self.config
		layers = self.seqdict
		#1: prob real has .realfc
		n_fclayers = len(config)-1 if self.probtype == 'real' else len(config)
		
		#1st layer is flatten.
		for i, shape in enumerate(config[0:n_fclayers]):
			fc_create = dispatcher.linear_dispatch(shape) 	#TODO: do type checking.
			#print(f"CplxMLP {fc_create=}, {shape=}")
			fc = fc_create(*shape)

			act = cplxlayer.Activation(f"act-fc{i+1}", self.activation)
			self.addBatchReshape(fc_create) #put in BatchReshape if fc layer is CplxLinear.
			self.addLayer(f"fc{i+1}", fc)
			self.addLayer(f"a{i+1}", act)


		#2: insert our cplx->real layer
		cplx2real_create = dispatcher.Cplx2Real_dispatch[self.probtype]	#'real'|'complex'
		cplx2real = cplx2real_create(*config[-1])		#CplxComplex2Real|CplxComplex2magphase
		self.addLayer("cplx2real", cplx2real)

		self.seq = nn.Sequential(layers)
		#2: send ourself to the desired device	
		self.to(device)	

	def initWeights(self):
		for i, layer in enumerate(self.seqdict.items()):
			k, fc = layer
			if isFC(k, fc) and method_exists(fc, 'initWeights'): 	#TODO: unify initWeights for TT and linear
				fc.initWeights(self._initWeights, kind=self.initkind)
		return

	def forward(self, x) -> torch.Tensor:
		out = self.seq(x)

		self.forward_count +=1
		return out			

	def FCisTT(self, fc_n: int) -> bool:
		""" Predicate for whether .fc<n> is a TT layer """
		return type(self.config[fc_n]) == layers.TTDesc

	def FCisTCL(self, fc_n: int) -> bool:
		""" Predicate for whether .fc<n> is a TCL layer """
		return type(self.config[fc_n]) == layers.TDesc

	@property
	def FC1isTT(self) -> bool:
		""" Predicate for whether .fc1 is a TT layer """
		return self.FCisTT(0)

	def TT_init_str(self) -> str:
		config_str = ''	
		if self.FC1isTT:
			config_str = f", TT init scheme: {self.config[0].tt_init}"
		return config_str

	def getConfigStr(self) -> str:
		configstr = ''
		for k, fc in self.seqdict.items():
			if isFC(k, fc):
				configstr += dispatcher.getConfigStr(fc) + ", "
		configstr += str(self.activation)
		return configstr

	def __repr__(self):
		return str(self.seq)		

	def __str__(self):
		return str(self.seq)		


#Default Complex Valued Neural Networks.
class CoShCVNN(CVnn_base.CVnn_Base):
	""" Complex-valued convolutional net designed for CoShREM (complex-Shearlets)
	"""
	def __init__(self, 
		conv2d=cplxlayer.CplxConv2d,
		conv2d_config: list = [(20, 30, 5, 1), (30, 50, 5, 1)], 
		linear=cplxlayer.CplxLinear, 
		initW = cplxnn.init.cplx_trabelsi_standard_,		#cplx_trabelsi_standard_|cplx_trabelsi_independent_
		initkind: str = "xavier",	#xavier|kaiming
		tracectx: Optional[trace.TraceContext] = None,
		probtype: str = 'real',
		activation: Optional[str] = None, 
		num_pics = 3,
		mlp = None,
	):
		super().__init__(
			conv2d = conv2d,
			conv2d_config = conv2d_config,
			linear = linear, 
			initW = initW,
			initkind = initkind,
			tracectx = tracectx,
			probtype = probtype,
			modelname = "CoShCVNN",
		)
		#assert(len(conv2d_config) == 2)		#we use 2 conv layers
		#print(f"conv2d_config {conv2d_config}")
		self.settrace(tracectx)

		self.num_pics = num_pics
		self.pics = [0,0,0,0,0] #length should be equal to number of layers.
		assert(mlp is not None)
		self.mlp = mlp
		self.forward_count = 0
		self.activation = activation
		return

	def makeConvLayerDict(self, i, shape, dropout: float = 0.0):
		layer_dict = {}
		if isinstance(shape, CVnn_base.Conv2dDesc):
			convolution = self.conv2d if shape[2] > 1 else cplxlayer.CplxConv2d
			layer_dict[f"conv{i+1}"] = convolution(*shape)
			layer_dict[f"act-conv{i+1}"] = cplxlayer.Activation(f"act-conv{i+1}", self.activation)
		elif isinstance(shape, CVnn_base.AvgPool2dDesc):
			layer_dict[f"pool-conv{i}"] = cplxnn.CplxToCplx[F.avg_pool2d](*shape) #not +1 because pool belongs to the previous layer.
			if dropout > 0.0:
				layer_dict[f"dropout-conv{i}"] = cplxnn.modules.extra.CplxDropout(dropout)
		else:
			raise Exception(f"Description of {type(shape)=} is not supported. Supported types include ({CVnn_Base.Conv2dDesc}, {CVnn_Base.AvgPool2dDesc}).")
		return layer_dict


	def defineModel(self,
		device = 'cuda',
		weight_init:bool = True, 
		log_model:bool = True,
		dropout: float = 0.0, 
		**conv_kwargs
	):
		"""
		Defines model architecture(layers, activations, pool etc.).
		---
		args:
			1. weight_init (True|False): use weight initialization strategy (Y: True, N: False).
			2. log_model (True|False): log model architecture details (Y: True, N: False)?
			3. conv_kwargs: set of keyword arguments to pass to the convolutional layer
		"""
		#<class 'cplxmodule.nn.modules.base.<runtime type `CplxSplitLayerReLU`>'>
		self.activation = dispatcher.get_activation(self.activation, self.probtype)
			
		#1. Add convolutional layers.		
		for i, shape in enumerate(self.conv2d_config):
			[self.addLayer(name, layer) for name, layer in self.makeConvLayerDict(i, shape, dropout = dropout).items()]
			
		#2. our MLP is a full Model itself - construct it in GPU memory
		
		self.mlp.defineModel(device)
		self.addLayer("mlp", self.mlp)

		#3. Finalize full model.
		self.seq = nn.Sequential(self.seqdict)

		#4. weight initialization.
		if weight_init:
			self.initWeights()

		#5. upload to gpu & log if necessary.	
		self.to(device)
		if log_model:
			self.logModelDetails()
		return

	def logModelDetails(self): #TODO: Move Instance to baseclass.
		formatted_str = f"""
						===
						Model & Parameters:- 
						---
						{self.getModelConfigStr()}
						
						Architecture:-
						---
						{self}
						===
				   		"""
		self.logstr(formatted_str)	
		return

	def getConvConfigStr(self):
		configstr = ''
		for k, fc in self.seqdict.items():
			if isCONV(k, fc):
				configstr += dispatcher.getConfigStr(fc) + ", "
		return configstr

	def getFCConfigStr(self) -> str:
		configstr = self.mlp.getConfigStr()
		return configstr

	def get_init_str(self) -> str:
		""" Allow sub-class to over-ride and provide extra details for init string """
		return self.mlp.TT_init_str()		

#	def get_activation_str(self):
		""" Allow sub-class to over-ride and provide extra details for activation string """
#		return str(self.activation)		

	def initWeights(self):
		#init of conv layers.
		for i, t in enumerate(self.seqdict.items()):
			name, layer = t
			if isCONV(name, layer): 	#TODO: unify initWeights for TT and linear
				layer.initWeights(self._initWeights, kind=self.initkind)
		#init of mlp
		self.mlp.initWeights()
		return

	def forward(self, x) -> torch.Tensor:
		out = self.seq(x)

		self.forward_count +=1
		return out

	def getweights(self, layer):	
		ourtype = type(layer)
		weightsOp = dispatcher.getW_dispatch.get(ourtype, None)
		if weightsOp is None:
			raise Exception("Error<!>: Unknown layer in getweights()")
		weights = weightsOp(layer)
		
		return weights

	def __repr__(self):
		return str(self.seq)		

	def __str__(self):
		return str(self.seq)