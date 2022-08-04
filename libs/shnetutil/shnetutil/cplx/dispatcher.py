"""
Title: Common Dispatcher file for all CVnn modules.

Description: The common dispatcher file is imported in all architecture files. it is used to recognize layers and weight operations for logging gradients.

Created on Sun Aug 16 17:44:29 2020

@author: Ujjawal.K.Panchal
"""
from typing import Union, Optional
import inspect
#torch imports
import torch
import torch.nn as nn
#cplx imports
from cplxmodule import cplx
import cplxmodule.nn as cplxnn

from shnetutil.cplx import cplxlayer, tt_layer, layers
from shnetutil.utils import torchutils

#Weight getter from layers. for logging gradients.
getW_dispatch = {
	nn.Conv2d: 								lambda conv: conv.weight,
	nn.modules.linear.Linear: 				lambda fc: fc.weight,
}

#type disatch for getConfigStr() - mostly because cplxnn layers are not derived from our CplxLayer base class.
getConfigStr_dispatch = {
	cplxlayer.CplxLayer: lambda layer: layer.getConfigStr(),
	cplxlayer.CplxConv2d: lambda layer: layer.getConfigStr(),
}

bias_dispatch = {
	nn.modules.linear.Linear: lambda layer: layer.bias,
	cplxlayer.CplxLinear: 	lambda layer: layer.bias.real,
	tt_layer.TTLayer:		lambda layer: layer.bias,
	layers.CplxSplitTTLayer: lambda layer: layer.r_splitlayer.bias,		#TODO: return i_splitlayer.bias
}

activation_dispatch = {
	'relu': nn.ReLU,
	'elu': nn.ELU,
	'real': nn.ReLU,
}

def linear_dispatch(desc):
	linear = None

	if (type(desc) == tuple):
		linear = cplxlayer.CplxLinear 	#(w, h)
	else:
		assert(isinstance(desc, cplxlayer.TDesc))
		tensor_dispatch = {
			"tt":	layers.CplxSplitTTLayer,
		}
		linear = tensor_dispatch[desc.kind]
	return linear	

#Complex domain to real conversion layer
Cplx2Real_dispatch={
	'real':		layers.CplxComplex2Real,
}

def gettype(obj):
	return str(type(obj))

def hasmethod(myclass: object, methodname: str):
	""" predicate for 'myclass' has a method with 'methodname' """
	mymethod = getattr(myclass, methodname, None)
	return (mymethod is not None) and callable(mymethod)

def getModelConfigStr(model):
	ourstr = ''
	if hasmethod(model, "getModelConfigStr"):
		ourstr = model.getModelConfigStr()
	elif hasmethod(model, "getConfigStr"):
		ourstr = model.getConfigStr()
	else:
		ourstr = type(model)
	return str(ourstr)

def getConfigStr(layer):
	def def_configstr(layer):
		""" Default configstr for layers with .weight """
		config_str = f"{torchutils.modelName(layer)}{tuple(layer.weight.shape)}"
		return config_str

	if layer is None:
		return ''
	if isinstance(layer, cplxlayer.CplxLayer):
		getConfigStrOp = getConfigStr_dispatch.get(cplxlayer.CplxLayer)
	else:
		getConfigStrOp = lambda layer: gettype(layer)	#default to type(obj) for all unknown types
		if hasattr(layer, 'weight'):
			getConfigStrOp = def_configstr
 
	return getConfigStrOp(layer)

def get_bias(layer):
	biasop = bias_dispatch.get(type(layer))
	if biasop is None:
		print(f"unknown layer type {type(layer)}")
		assert(False)
	return biasop(layer)

def hasarg(func:callable, arg:str='alpha', alpha=1.):	
	return arg in inspect.signature(func).parameters

def get_activation(
	activation: Optional[str] = None, 
	probtype: Optional[str] = None, 
	tocplx=True
): 
	assert(not(activation == None and probtype == None)), f"Need at least one of `activation` or `probtype`\
															parameter to determine activation function."

	activation_function =  activation_dispatch.get(activation)

	if activation_function == None:
		activation_function = activation_dispatch[probtype]

	if tocplx:
		activation = cplxnn.CplxToCplx[activation_function] 	#cplxmodule.nn.modules.base._CplxToCplxMeta
		if hasarg(activation, 'alpha'):
			activation = activation(alpha=1.)
		else:
			activation = activation()	

	else:
		activaion = activation_function	
	return activation

def get_histogram(tensor: Union[torch.tensor, cplx.Cplx], bins:int = 10):
	hist = None
	if torch.is_tensor(tensor):
		hist = torch.histc(tensor, bins = 10)
	else:
		hist = [torch.histc(tensor.real, bins = 10), torch.histc(tensor.imag, bins = 10)]
	return hist

if __name__ == '__main__':
	import cplxmodule.nn as cplxnn
	from shnetutil.cplx import tt_layer

	#1: cplxnn.CplxLinear
	fc0 = cplxnn.CplxLinear(1250, 500)
	print(f"dispatcher.getConfigStr(fc0) = {getConfigStr(fc0)}")

	#2: cplxlayer.CplxLinear
	fc11 = cplxlayer.CplxLinear(1250, 500)
	fc11.initWeights()
	print(f"{fc11.getConfigStr()}")
	print(f"dispatcher.getConfigStr(fc11) = {getConfigStr(fc11)}")
	print(f" bias {type(get_bias(fc11))}")

	#3: TTLayer
	tt_fc1 = tt_layer.TTLayer(
		in_factors = (50, 25),
		out_factors = (20, 25),
		ranks = (8, ),
		ein_string = "nab, aoix, bipy",
	)
	print(f"{tt_fc1.getConfigStr()}")
	print(f"dispatcher.getConfigStr(tt_fc1) = {getConfigStr(tt_fc1)}")
	print(f" bias {type(get_bias(tt_fc1))}")
