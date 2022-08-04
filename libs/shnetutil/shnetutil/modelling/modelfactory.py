# -*- coding: utf-8 -*-
"""
Title: Model Factory - 
	
Created on Fri Aug 21 16:01:29 2021

@author: Manny Ko & Ujjawal.K.Panchal
"""
import abc
#import argparse
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Optional, Union

#from enum import Enum
from pydantic import BaseModel, validator 

import torch.nn as nn

# tuple to config the Model for each training stage - used in ModelPipeline()
#Stage= namedtuple("Stage", "name base conv2d linear init probtype modeltype fc_comp")

class Recipe_base(BaseModel):
	""" Base-class for a Recipe - which is a model descriptor to drive a ModelFactory. """
	name: str                                              

	#https://pydantic-docs.helpmanual.io/usage/model_config/
	class Config:		#TypeError: metaclass conflict: the metaclass of a derived class must be a (non-strict) subclass of the metaclasses of all its bases
		pass
		#allow_mutation = False
		#arbitrary_types_allowed = True
		#underscore_attrs_are_private = True

class ModelFactory(metaclass=abc.ABCMeta):
	""" This is the factory to create 1 model in our training pipeline.
		Unlike standard PyTorch training where one create the model in the main. Factory
		enable us to have more than 1 training run with a fresh and potentially different variant 
		of the model in each run.
	"""
	def __init__(self, 
		recipe: Recipe_base,	#our model-recipe
		tracectx=None
	):
		self.recipe = recipe
		self.tracectx = tracectx

	def __repr__(self):
		return str(self.recipe)	

	@abstractmethod	
	def makeModel(self) -> nn.Module:
		return None

class ModelPipeline(metaclass=abc.ABCMeta):
	""" This is the factory to create all the stages for our training pipeline.
		The ARD and masked models must be trained sequentially after the regular model.
	"""
	def __init__(self, 
		stages: List[ModelFactory],
		tracectx=None
	):
		self.stages = stages
		self.tracectx = tracectx

	def __repr__(self):
		return str(self.stages)	

	@abstractmethod	
	def makeModels(self, device='cuda'):
		models = {"none": None}
		return models

#end of generic declarations
