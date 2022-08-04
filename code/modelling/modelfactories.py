"""
Title: test_fashion.

Created on Sun Aug 16 17:44:29 2020

@author: Ujjawal.K.Panchal & Manny Ko & Hector Andrade-Loarca.
"""
from collections import namedtuple
from typing import Callable, List, Tuple, Union, Optional

#PyTorch
import torch, numpy as np

#our packages
from shnetutil.utils import trace
from shnetutil.modelling import modelfactory

from modelling import CVnn, fashionRecipes

class ModelFactory(modelfactory.ModelFactory):
	""" This is the factory to create 1 model for our training pipeline.
		#NOTE: only used by test_loadmodels during big refactoring.
	"""
	factories = {
		None: CVnn.CVnnFactory, 
		"nosh-cplx" : CVnn.CVnnFactory,
	}
	def __init__(self, 
		recipe: modelfactory.Recipe_base,		# model descriptor driving the factory.
		tracectx: trace.TraceContext=None,
		ablation_type: Optional[str] = None,
	) -> torch.nn.Module:
		assert(issubclass(type(recipe), modelfactory.Recipe_base))
		super().__init__(recipe, tracectx)
		self.recipe = recipe
		self.factory = ModelFactory.factories[ablation_type]
		return

	def makeModel(self, dropout: float = 0.0, device='cuda') -> torch.nn.Module:
		myfactory = self.factory(self.recipe, self.tracectx)
		model = myfactory.makeModel(device = device, dropout = dropout)
		#print(model) #TODO: control verbosity through self.parameter.
		#print(model._initWeights, model.mlp._initWeights)
		return model
