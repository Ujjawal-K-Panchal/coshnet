"""
Title: CVnn - complex valued NN based on COShREM.
	
Created on Sun Aug 16 17:44:29 2020

@author: Ujjawal.K.Panchal & Manny Ko & Hector Andrade-Loarca.
"""
import copy
from collections import namedtuple
from collections.abc import KeysView
from typing import Callable, List, Tuple, Optional, Union
from pydantic import BaseModel, ValidationError, validator, root_validator

import torch

from cplxmodule.nn.init import cplx_trabelsi_standard_, cplx_trabelsi_independent_

#custom package imports.
from pyutils.enumutils import *
import shnetutil.cplx.dispatcher as dispatcher
import shnetutil.cplx.cplxlayer as cplxlayer
from shnetutil.cplx import layers, CVnn_base
from shnetutil.modelling import modelfactory
from shnetutil.utils import trace

from shnetutil.coshrem_xform import CoShREMConfig #coshrem xform config.

#adjascent file imports.
from .SplitCVnn import CplxMLP, CoShCVNN
from shnetutil.coshrem_xform import ksh_spec


trichannel_colorspaces = ["rgb", "lab"] #If/when want to support color images.

class FCLayerCompression(str, Enumbase):
	"""
	The type of compression to apply on the bottleneck
	fc layer.
	"""
	none = 'none'
	tt = 'tt'

class Colorspaces(str, Enumbase):
	"""
	The colorspace in which your dset exists.
	"""
	rgb = "rgb"
	lab = "lab"
	lum_lab = "lum_lab"
	grayscale = "grayscale"

class Activations(str, Enumbase):
	"""
	The activation you are using.
	"""
	kReLU = 'relu'
	kELU = 'elu'
	none = None

class ProbabilityKind(str, Enumbase):
	"""
	The type of output probability.
	"""
	kReal = "real"

class SupportedModels(Enumbase):
	"""
	Models which are supported in this
	release.
	"""
	kCoShCVNN = CoShCVNN #CoShCVNN aka. CoShNet.

class ConvLayers(Enumbase):
	"""
	Type of convolutional layers.
	"""
	kSplitConv = cplxlayer.CplxConv2d #simple complex conv 2d.
	kSplitDCF = layers.CplxSplitDCF #dcf complex conv 2d.

class LinearLayers(Enumbase):
	"""
	Type of linear layers.
	"""
	kSplitLinear = cplxlayer.CplxLinear #simple linear layer.
	kSplitTT = layers.CplxSplitTTLayer	#tensor train linear layer.

if False: 		#TODO: this does not work. But here for future support.
	class InitFuncs(Enumbase):
		kTrabelsi_standard 	  = cplx_trabelsi_standard_
		kTrabelsi_independent = cplx_trabelsi_independent_

kInitFuncs = [cplx_trabelsi_standard_, cplx_trabelsi_independent_]

class InitFuncError(Exception):
	""" Exception when a given init function is not a valid option we can use """
	pass

def getSupportedMetavar(supported_models):
	""" Convert the set of supported model into 'metavar=getSupportedMetavar()' argparse help """
	dispatch = {
		dict:	lambda supported: supported.keys(),
		list:	lambda supported: supported,
	}
	if type(supported_models) in dispatch.keys():
		names = dispatch[type(supported_models)](supported_models)
	else:
		assert(isinstance(Enumbase))
		names = supported_models.names()
	optionstr = ''
	for name in names:
		optionstr += f"{name}|"
	return optionstr

def getSupportedChoices(supported_models):
	""" Convert the set of supported model into 'choices=getSupportedChoices()' argparse help """
	metavar = getSupportedMetavar(supported_models)
	return tuple(metavar[:-1].split('|'))	

def getSupportedProbability():
	return getSupportedChoices(ProbabilityKind.values())

def getSupported_fcdecomp():
	return getSupportedChoices(FCLayerCompression.values())

def getSupportedActivation():
	return getSupportedChoices(Activations.values())

def getSupportedColorspaces():
	return getSupportedChoices(Colorspaces.values())


class CVnnMLPrecipe(modelfactory.Recipe_base):
	config: list 	 = [(5 * 5 * 50, 500), (500, 10), (20, 10)]
	linear: LinearLayers = LinearLayers.kSplitLinear
	init: Callable 	 	 = cplx_trabelsi_standard_
	probtype: ProbabilityKind = ProbabilityKind.kReal
	fc_comp: FCLayerCompression = FCLayerCompression.none
	tt_init: str = "tt-svd-1"		#"cores-init"|"tt-svd-1"|"tt-svd-2"
	activation: Optional[Activations] = None

#
# MLP/linear part of CVnn:
#
#standard recipe for our MLP sub-model used by older code as a starting template
kCVnnMLPrecipe = CVnnMLPrecipe(
	name  = 'CVnnMLPrecipe',
	config= [(5 * 5 * 50, 500), (500, 10), (20, 10)],
	linear= LinearLayers.kSplitLinear,
	init  = cplx_trabelsi_standard_,
	probtype = ProbabilityKind.kReal,
	fc_comp = FCLayerCompression.none,
	activation = Activations.none,
)

class CVnnMLPFactory(modelfactory.ModelFactory):
	""" Model factory for all variants of our CoShCVnn MLP models.
	"""
	def __init__(self, 
		recipe: CVnnMLPrecipe,
		tracectx: trace.TraceContext = None
	):
		super().__init__(recipe, tracectx)
		if tracectx:
			assert(isinstance(tracectx, trace.TraceContext))

	def makeModel(self, device='cuda') -> torch.nn.Module:
		recipe = self.recipe
		model = None
		modelclass = CplxMLP	#SupportedModels[recipe.modeltype.name].value
		activation = dispatcher.get_activation(recipe.activation, recipe.probtype)

		model = modelclass(
			recipe.config,
			recipe.linear.value, 
			recipe.init,
			probtype = recipe.probtype.value,
			activation = activation,
		)
		#print(f"makeModel {torchutils.modelName(model)}")
		model.defineModel()
		model.to(device)
		return model

def get_Linear2TTDescription(
	in_features: int,
	out_features: int,
	ranks: tuple = (8, ),
	tt_init: str = "tt-svd-1",
	tt_factor_method: str = "range",
	tt_factor_range: tuple = (20, 30)
):
	print(f"get_Linear2TTDescription {in_features=}")
	in_factors, out_factors = layers.CplxSplitTTLayer.Linear2InOut(in_features, out_features, tt_factor_method, tt_factor_range)
	return layers.TTDesc("tt", in_factors, out_factors, ranks, "nab, aoix, bipy", tt_init)

#===============================================================================
#
# Convolutional part of CVnn:
#
class CVnnRecipe(modelfactory.Recipe_base):
	""" model descriptor to drive our standard CoShCVnn model factory. 
		Most client code can use these defaults.
	"""
	modeltype: SupportedModels = SupportedModels.kCoShCVNN
	conv2d: ConvLayers	 = ConvLayers.kSplitConv		#defaults to Ivan's 
	conv_config: List[tuple] = [
								CVnn_base.Conv2dDesc(20, 30, 5, 1), CVnn_base.AvgPool2dDesc(2,2),
								CVnn_base.Conv2dDesc(30, 50, 5, 1), CVnn_base.AvgPool2dDesc(2,2)
					]
	linear: LinearLayers = LinearLayers.kSplitLinear
	init: Callable 	 	 = cplx_trabelsi_standard_
	probtype: ProbabilityKind = ProbabilityKind.kReal	#real or complex probability
	mlp_config = kCVnnMLPrecipe 	# = [(5 * 5 * 50, 500), (500, 10), (20, 10)]
	activation: Activations = Activations.kReLU
	colorspace: Colorspaces = Colorspaces.grayscale
	name: Optional[str] = None
	@root_validator
	@classmethod
	def check_settings(cls, values):
		if (not values.get('init') in kInitFuncs):
			raise(InitFuncError(f"{values.get('init').__name__} is not a valid func."))
		return values

#recipe for our standard CoShCVNN model
kCoShCVNNrecipe = CVnnRecipe(
	name 	= "CoShCVNNrecipe",
	conv2d 	= ConvLayers.kSplitConv,		#defaults to Ivan's                                
	activation= Activations.kReLU,
)
#recipe for our CoShCVNN model that use Split_DCF conv:
kSplit_DCFrecipe = CVnnRecipe(
	name 	= "Split_DCFrecipe",
	conv2d 	= ConvLayers.kSplitDCF, 
	init 	= cplx_trabelsi_standard_,
	activation= Activations.kReLU,
)

class CVnnFactory(modelfactory.ModelFactory):
	"""
	Model factory for all variants of our CoShCVnn models.
	---
	Args:
		1. recipe: CVNNRecipe object =  Which is to be made by the factory.
		2. tracectx: trace.TraceContext = the logger which will log details about this and store in logs/ folder.
	"""
	def __init__(self, 
		recipe: CVnnRecipe = kCoShCVNNrecipe,
		tracectx: trace.TraceContext = None,
	):
		super().__init__(recipe, tracectx)
		if tracectx:
			assert(isinstance(tracectx, trace.TraceContext))

	def makeModel(self, dropout:float = 0.0, device ='cuda') -> torch.nn.Module: #TODO: typecasting for device.
		recipe = self.recipe
		mlprecipe = recipe.mlp_config
		model = None
		modeltype = recipe.modeltype
		modelclass = SupportedModels[modeltype.name].value 	#CoShCVNN()
		activation_fn = dispatcher.get_activation(recipe.activation) 	#assume MLP and CVnn use same activation

		mlp = CplxMLP(
			config = mlprecipe.config, 			#Config Example = [(5 * 5 * 50, 500), (500, 10), (20, 10)],
			linear = mlprecipe.linear.value, 	#Exammple value = CVnn.LinearLayers.kSplitLinear.value,
			initW 	= recipe.init, 				#Example init = cplxnn.init.cplx_trabelsi_standard_,
			initkind = "xavier",				#Example initkind = xavier|kaiming
			activation = activation_fn,
			probtype = mlprecipe.probtype.value,
			tracectx = self.tracectx
		)

		model = modelclass(
			conv2d = recipe.conv2d.value,
			conv2d_config = recipe.conv_config,
			linear = recipe.linear.value, 
			initW = recipe.init,
			tracectx = self.tracectx,
			activation = recipe.activation,
			probtype = recipe.probtype.value,
			mlp = mlp,
		)
		#print(f"makeModel {torchutils.modelName(model)}") #TODO: control verbosity, remove comment.
		model.defineModel(device = device, dropout = dropout)
		return model


def makeTiny(coShNetRecipe:CVnnRecipe, fcdesc:Union[tuple, layers.TDesc], index=0):
	""" all tiny recipes use DCF for conv and some form of TT or TCL for .fc1 """
	tinyRecipe = copy.deepcopy(coShNetRecipe)
	tinyRecipe.conv2d = ConvLayers.kSplitDCF
	tinyRecipe.mlp_config.config[index] = fcdesc
	tinyRecipe.mlp_config.fc_comp = FCLayerCompression.tt
	return tinyRecipe


if __name__ == '__main__':
	print(f"colorspaces {trichannel_colorspaces}")
	print(f"ConvLayers.names() {ConvLayers.names()}")
	print(f"ConvLayers.values() {ConvLayers.values()}")
	print(f"{cplx_trabelsi_standard_} is in {kInitFuncs} {cplx_trabelsi_standard_ in kInitFuncs}")


