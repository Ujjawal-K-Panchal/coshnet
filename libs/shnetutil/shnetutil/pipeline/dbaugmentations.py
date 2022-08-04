# -*- coding: utf-8 -*-
"""
Title: Augmentation pipeline. 
	
	Designed to run several training runs and still guarantee the same result as 
	training a single model.
	
Created on Mon Jul 21 16:01:29 2020

@author: Ujjawal.K.Panchal & Manny Ko
"""
from functools import partial
from pydantic import BaseModel
import time
from typing import List, Tuple, Dict, Union, Optional

from shnetutil.coshrem_xform import ksh_spec
from shnetutil.pipeline import batch, augmentation
from pyutils.testutil import time_spent

#from . import training

class CaptureAugmentation(augmentation.NullXform):
	""" Capture the result of a certain stage in our augmentation pipeline """
	def __init__(self, batchsize=512):
		augcache = batch.BatchCache(batchsize, batchbuilder=None, xform=None)
		augcache.setcapture(True)
		self.cache = augcache
		self.capture = False
		self.reset()
		self.setBatchsize(batchsize)

	def reset(self):
		self.isvalid = False
		self.batchindex = 0
		self.cache.reset()

	def rewind(self):
		self.batchindex = 0

	def setBatchsize(self, batchsize=512):
		self._batchsize = batchsize		

	@property
	def batchsize(self):
		return self._batchsize
		
	def setcapture(self, enable:bool):
		self.capture = enable	
		self.cache.setcapture(enable)

	def finalize(self):
		self.cache.finalize()
		self.capture = False
		self.isvalid = True

	def __str__(self):
		return f"CaptureAugmentation({self.capture})"	

	def __call__(self, sample):
		if not self.isvalid and (not self.capture): 	#no a valid cache AND not in capture mode
			return sample
		if self.capture:
			self.cache.insert(sample)
			return sample
		batchN = self.batchindex
		self.batchindex += 1
		return self.cache[batchN]

def locate(auglist:list, target:augmentation.Base) -> int:
	""" Locate the 'stage' in our pipeline and returns its index """
	index = -1
	for i, stage in enumerate(auglist):
		if (type(stage) == target):
			index = i
	return index

def CoShNetAugmentations(
	mean, std, 
	additional_xforms = None,
	batchsize:int = 512,
	denoise = False,
	device = 'cuda',
	coshrem_args: BaseModel = ksh_spec,
) -> augmentation.Sequential:
	""" The standard augmentation pipeline we use for our training CoShNet """
	auglist = [
		augmentation.Normalize(mean, std),		#10K1E: 79.7%, 10K4E: 86.9%, 10K10E: 87.6%	
		augmentation.Pad([(0,0), (2,2), (2,2)]),		  
		augmentation.CoShREM(coshrem_config = coshrem_args, device = device, tocplx = True),
		augmentation.ToTorchDims(),
		CaptureAugmentation(batchsize=batchsize),
	]
	if denoise:
		stage_idx = locate(auglist, augmentation.CoShREM)
		assert(stage_idx != -1)
		if (stage_idx != -1):
			sigma, RMS = 0.00712, auglist[stage_idx].RMS
			denoiser = augmentation.Denoise(thresholdingFactor=3, RMS=RMS, sigma=sigma, device=device)
			auglist.insert(stage_idx+1, denoiser)
			#print(auglist)

	capturecache = auglist[-1]
	assert(issubclass(type(capturecache), CaptureAugmentation))
	ourTransform = augmentation.Sequential(auglist, device=device)

	if additional_xforms:
		for position, xform in additional_xforms:
			ourTransform.xforms.insert(position, xform)
	return ourTransform

def enableCaching(pipeline: augmentation.Sequential):
	for stage in pipeline:
		if issubclass(type(stage), CaptureAugmentation):
			stage.setcapture(True)

def removeCaching(pipeline: augmentation.Sequential):
	indices = []
	for i, stage in enumerate(pipeline):
		if issubclass(type(stage), CaptureAugmentation):
			indices.append(i)
	assert(len(indices) <= 1)		
	for i in indices:
		del pipeline.xforms[i]		

def denoiseEnable(pipeline: augmentation.Sequential): #TODO: remove as not used.
	denoise = False
	for stage in pipeline:
		if issubclass(type(stage), augmentation.CoShREM):
			denoise = stage.denoise
	return denoise
			
def noShAblationAugs(
				mean,
				std,
				additional_xforms = None,
				denoise = False,
				real = False, device='cpu',
				coshrem_args: Optional[Dict] = None,
	) -> augmentation.Sequential:
	""" The augmentation pipeline we use for our training no shearlet ablation.
		Note: coshrem_args are added here just to satisfy the format of dispatcher.
	"""
	ourTransform = augmentation.Sequential([
		augmentation.Normalize(mean, std),
		augmentation.Pad([(0,0), (2,2), (2,2)]),		  
		augmentation.NoShAblation(20, real = real),
		augmentation.ToTorchDims(),
	])
	if additional_xforms:
		for position, xform in additional_xforms:
			ourTransform.xforms.insert(position, xform)
	return ourTransform

def augDispatcher(ablation_type: Optional[str] = None):
	augFunc = CoShNetAugmentations
	if ablation_type == "nosh-cplx":
		augFunc = noShAblationAugs
	elif ablation_type == "nosh-real":
		augFunc = partial(noShAblationAugs, real = True)
	return augFunc

def fashion_augmentations(
	datasetmean: float,
	datasetstd: float,
	denoise: bool = False,
	ablation_type: Optional[str] = None,
	device = 'cuda',
	train_additional_xforms: Optional[List] = None,
	test_additional_xforms: Optional[List] = None,
	coshrem_args: BaseModel = ksh_spec,
) -> tuple:
	"""
		Desc: 
			- Get fashion augmentations.
		Args:
		---
		ablation_type: selects one of augmentations based on the experiment of interest. supported:-
						1. None: ourAugmentations: Regular pipeline.
						2. "nosh-cplx": When CV output but no shearlets.
						3. "nosh-real": When RV output and no shearlets.
	"""
	ourTrainTransform, ourTestTransform = None, None
	augmentation_function = augDispatcher(ablation_type)

	ourTrainTransform = augmentation_function(
		datasetmean, datasetstd, 
		denoise = denoise,
		device = device,
		additional_xforms  = train_additional_xforms,
		coshrem_args = coshrem_args,
	)

	ourTestTransform = augmentation_function(
		datasetmean, datasetstd,
		denoise = False,
		device = device,
		additional_xforms = test_additional_xforms,
		coshrem_args = coshrem_args,
	)

	ourValidateTransform = augmentation_function(
		datasetmean, datasetstd,
		denoise = False,
		device = device,
		additional_xforms = test_additional_xforms,
		coshrem_args = coshrem_args,
	)
	return ourTrainTransform, ourTestTransform, ourValidateTransform

def DCF_augmentations(
	datasetmean: float,
	datasetstd: float,
	train_rot: float = 15,
	test_rot: float = 30,
	kCache:bool=False,	
):
	""" The augmentation pipeline we use for our training Fashion with DCF ROT layers."""
	ourTrainTransform = augmentation.Sequential([
		augmentation.RandomRot((-train_rot, train_rot), seed=99),
		augmentation.Normalize(datasetmean, datasetstd),		
		augmentation.ToTorchTensorDCF(),
	])
	
	ourTestTransform = augmentation.Sequential([
		augmentation.RandomRot((-test_rot, test_rot), seed=999),
		augmentation.Normalize(datasetmean, datasetstd),		
		augmentation.ToTorchTensorDCF(),
	])
	return ourTrainTransform, ourTestTransform, ourTestTransform
