# -*- coding: utf-8 -*-
import os, time
import numpy as np

#import torch

from shnetutil import shearletxform, shxform

class PyShXform(shxform.ShXform):
	""" pyshearlab based ShXform """
	def __init__(self, sh_spec):
		""" Generating the shearlet system with pyCoShRem """
		super().__init__(sh_spec)
		#Note: we on purpose delay call to start() to separate object construction and first use
		#This way we can pass a minimal Pickle object across process boundary
	
	def start(self, device):
		super().start(device)
		#Note: regardless of 'device' we are using GPU for our compute
		self._shearletSystem, _ = shearletxform.onceInit(self.sh_spec, kCUDA=False)

	def xform(self, item):
		""" image=(tensor, label) """
		return shearletxform.shearletxform(self._shearletSystem, item)

	def xform_np(self, item):
		""" image=(ndarray, label): apply shearlet xform to ndarray 'd' """
		return shearletxform.shearletxform_np(self._shearletSystem, item)

class PyShXform4Covid(PyShXform):
	""" pyshearlab based ShXform for COVID """
	def __init__(self, sh_spec):
		""" Generating the shearlet system with pyshearlab """
		super().__init__(sh_spec)
	
	def start(self, device):
		super().start(device)
		#Note: regardless of 'device' we are using GPU for our compute
		self._shearletSystem, _ = shearletxform.onceInit(self.sh_spec, kCUDA=False)

	def xform(self, item):
		""" image=(tensor, label) """
		return shearletxform.shearletxform_np(self._shearletSystem, item)

if __name__ == '__main__':
	from shnetutil import torchutils

	#our default Shearlet system config:
	kShearletShape = (92, 92)
	kShearletScales = 2
	kShearlet_spec={
		'useGPU': 	0, 
		'rows':		kShearletShape[0],	#92
		'cols':		kShearletShape[1], 	#92
		'nScales':	kShearletScales,	#2
	}

	device = torchutils.onceInit(kCUDA=False)
	xform = PyShXform(kShearlet_spec)
	xform.start(device)
