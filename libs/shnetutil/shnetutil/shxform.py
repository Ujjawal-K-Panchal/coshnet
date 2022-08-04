# -*- coding: utf-8 -*-
import time
import numpy as np
from pydantic import BaseModel, validator
#from abc import ABC
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

import torch
#our packages
from .utils import torchutils

# ksh_spec = {
# 	'rows': 		256, 
# 	'cols': 		256, 
# 	'scales_per_octave': 2, 
# 	'shear_level': 	3, 
# 	'octaves': 		3.5, 
# 	'alpha': 		.8,		# Anisotropy level
# }

class CoShREMConfig(BaseModel):
	rows: int = 32
	cols: int = 32
	scales_per_octave: int = 2 
	shear_level: int = 3
	octaves: int = 1
	alpha: float = .5
	wavelet_eff_support: int = 7
	gaussian_eff_support: int = 14

	@validator('alpha')
	def check_alpha(cls, alpha):
		assert(alpha >=0 and alpha <=1), Exception(f"alpha {alpha} out of bounds; α ∈ [0, 1].")
		return alpha

class ShXform():    #metaclass=ABC
	""" complex Shearlet xform """
	def __init__(self, sh_spec):
		""" Prepare the shearlet system given by 'sh_spec' """
		self.started = False
		self.sh_spec = sh_spec	#this should be pickleable
		self._shearletSystem = None
		#Note: we on purpose delay call to start() to separate object construction and first use
		#This way we can pass a minimal Pickle object across process boundary
	
	def start(self, device):
		""" Generating the shearlet system specified by 'sh_spec' """
		self.device = device

		use_cuda = torch.cuda.is_available()
		use_cuda &= torchutils.is_cuda_device(device)
		torch.backends.cudnn.enabled = use_cuda
		#print(f"use_cuda: {use_cuda}")
		self.use_cuda = use_cuda
		#TODO: verified the cuda backend is inited correctly
		self.started = True

	def xform(self, image):
		""" image=(tensor, label) """
		assert(self.started)
		return None

	def invxform(self, coeffs):
		assert(self.started)
		return None	

	def xform_np(self, image):
		""" image=(ndarray, label): apply shearlet xform to ndarray 'd' """
		assert(self.started)
		return None

	def cleanup(self):
		self.started = False

	@property
	def dim(self):
		return self.sh_spec.rows, self.sh_spec.cols
	
	@property
	def shearletSystem(self):
		""" shearlets, shearletIdxs = coshxform.shearletSystem """
		return self._shearletSystem
	@property
	def shearlets(self):
		""" shearlets, shearletIdxs = coshxform.shearletSystem """
		return self._shearletSystem[0]
	@property
	def shearletIdxs(self):
		""" shearlets, shearletIdxs = coshxform.shearletSystem """
		return self._shearletSystem[1]


#visualization support moved to shnetutil.cplx.visual 

if __name__ == '__main__':
	device = torchutils.onceInit(kCUDA=True)
	xform = ShXform(device, ksh_spec)
