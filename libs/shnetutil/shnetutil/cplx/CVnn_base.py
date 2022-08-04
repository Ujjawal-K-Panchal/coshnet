# -*- coding: utf-8 -*-
"""

Title: Shearlet based CNN common base class.
	
Created on Sun Sept 12 17:44:29 2021

@author: Manny Ko & Ujjawal.K.Panchal 

"""
import abc
from collections import namedtuple, OrderedDict
from typing import Callable, List, Tuple, Optional, Union

import torch.nn as nn

import cplxmodule.nn as cplxnn

from pyutils.classutils import method_exists
import shnetutil.cplx.cplxlayer as cplxlayer
from ..utils import torchutils, trace
from . import dispatcher

#Descriptor for a 2d-conv layer
Conv2dDesc=namedtuple("Conv2dDesc", ["in_channels", "out_channels", "kernel_size", "stride"])
AvgPool2dDesc=namedtuple("AvgPool2dDesc", ["input", "kernel_size"])

class Cnn_Base(nn.Module, trace.TraceMixin):
	"""
	Common base-class for all Cnn models. 
	"""
	def __init__(self,
		conv2d, 
		conv2d_config,
		linear, 
		initW,
		initkind: str = "xavier",	#xavier|kaiming
		tracectx: Optional[trace.TraceContext] = None,
		probtype: str = 'real',
		modelname="Cnn_Base",
	):					 
		super().__init__()
		trace.TraceMixin.__init__(self)
		self.settrace(tracectx)

		self.conv2d = conv2d
		self.conv2d_config = conv2d_config
		self.linear = linear
		self._initWeights = initW
		self.initkind = initkind
		self.probtype = probtype
		self.activation = None
		self._modelName = modelname 	#Note: ._modelName is what torchutils.modelname() look for
		self.seqdict = OrderedDict()	#for nn.Sequential which expects an OrderedDict

	def addLayer(self, name: str, layer: nn.Module):
		self.seqdict.update({f"{name}": layer})

	def getModelConfigStr(self):
		model_config_str = f"""
  1. Name: {self.modelname()}()
  2. Convolutional Layers: {self.getConvConfigStr()}.
  3. Linear Layers: {self.getFCConfigStr()}
  4. Initialization Scheme: {self._initWeights.__name__}; Kind: {self.initkind}.{self.get_init_str()}
  5. Activation: {self.get_activation_str()}
  6. Pooling: {self.get_pool_str()}
  """
		return  model_config_str

	def haslayer(self, attrname:str):	
		""" our layers are all in .seqdict now, not as a class attribute """
		return attrname in self.seqdict.keys()

	def getConvConfigStr(self):
		configstr = ''
		if self.haslayer('conv1'):
			configstr += dispatcher.getConfigStr(self.seqdict['conv1']) + ", "
		if self.haslayer('conv2'):
			configstr += dispatcher.getConfigStr(self.seqdict['conv2'])
		return configstr

	def getFCConfigStr(self):
		configstr = 'getFCConfigStr place holder'
		return configstr

	def get_init_str(self):
		""" Allow sub-class to over-ride and provide extra details for init string """
		return ''		

	def get_activation_str(self):
		""" Allow sub-class to over-ride and provide extra details for activation string """
		return str(self.activation)		

	def get_pool_str(self):
		posns = []
		for name, layer in self.seqdict.items():
			if "pool" in name:
				posns.append(int(name[-1]))
		return f"on layers: {posns}"

	def modelname(self):
		return torchutils.modelName(self)


class CVnn_Base(Cnn_Base):
	"""
	Common base-class for all CVnn models. 
	"""
	def __init__(self,
		conv2d=cplxlayer.CplxConv2d, 
		conv2d_config=[],
		linear=cplxlayer.CplxLinear, 
		initW = cplxnn.init.cplx_trabelsi_standard_,		#cplx_trabelsi_standard_|cplx_trabelsi_independent_
		initkind: str = "xavier",	#xavier|kaiming
		tracectx: Optional[trace.TraceContext] = None,
		probtype: str = 'real',
		modelname="CVnn_Base",
	):					 
		super().__init__(
			conv2d,
			conv2d_config,
			linear,
			initW,
			initkind,
			tracectx,
			probtype,
			modelname,
		)