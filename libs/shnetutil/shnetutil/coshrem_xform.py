# -*- coding: utf-8 -*-
"""

Title: Shearlet based CNN vs. simple MLP in Fashion MNIST.
	
Created on Mon Mar 16 17:44:29 2020

@author: Ujjawal.K.Panchal & Manny Ko.

debugged date: Sun Sep 19 14:07:17 2021.
"""
import os, time
import numpy as np
from PIL import Image
from coshrem.shearlet import construct_shearlet
from numpy.random import randint
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from skimage.transform import resize

#import numba

from coshrem.util.cone import cone_orientation
from coshrem.shearlet import yapuls, padarray, shear, construct_shearlet
from matplotlib.colors import hsv_to_rgb

import torch

from . import cplx, shxform
from .shxform import CoShREMConfig
import cplxmodule
from .utils import torchutils


"""
A port of Hector's Pytorch_PyCoshrem.ipynb
"""
# Default parameters for our Shearlet system
# https://www.math.colostate.edu/~king/software/CoShREM_Parameter_Guide.pdf
#

ksh_spec = CoShREMConfig(
			rows = 32,
			cols = 32,
			scales_per_octave = 2,
			shear_level = 3,
			octaves = 1,
			alpha = .5,	#   Anisotropy level
						#	alpha: coefficient determining level of anisotropy. 
						#	1 -> wavelets, 
						#	0.5 -> shearlets, 
						#	0 -> ridgelets... etc.

			wavelet_eff_support = 7,
			gaussian_eff_support = 14,
)

def get_sh_spec(rows:int, cols:int, alpha:float=0.5):
	""" Configure a sh_spec for our Shearlet system """
	sh_spec = ksh_spec.copy()
	sh_spec['rows'] = rows
	sh_spec['cols'] = cols
	sh_spec['alpha'] = alpha
	return sh_spec

def get_CoShXform(device:str, rows:int, cols:int, alpha:float=0.5):
	""" Start a CoShXform """
	sh_spec = get_sh_spec(rows, cols, alpha)
	coshxform = CoShXform(sh_spec)
	coshxform.start(device)
	return coshxform	

def power_of_two(target):
	""" round to next power-of-2 """
	if target > 1:
		for i in range(1, int(target)):
			if (2 ** i >= target):
				return 2 ** i
	else:
		return 1

class CoShXform(shxform.ShXform):
	def __init__(self,
		sh_spec = ksh_spec, 
		tocplx = False,
		topolar = False,
		phase_first = False
	):
		""" Generating the shearlet system with pyCoShRem """
		super().__init__(sh_spec=sh_spec)	#default complex Shearlet spec - usually provided by client
		nextp2 = power_of_two(self.dim[0])
		assert(nextp2 == self.dim[0])
		self.ourdim = (nextp2, nextp2)
		self.tocplx = tocplx
		self.topolar = topolar
		self.phase_first = phase_first
		#print(f"CoShXform: {sh_spec}")

	def __repr__(self):	
		return f"CoShXform({self.sh_spec})"	

	def __str__(self):
		return f"CoShXform({self.sh_spec})"	

	def start(self, device):
		super().start(device)

		#t = time.time()
		self._shearletSystem = getcomplexshearlets2D(**dict(self.sh_spec))
		#self.shearlets, self.shearletIdxs = self.shearletSystem
		#print(f"Elapsed time: getcomplexshearlets2D() {time.time()-t:3f}ms")

		#for pytorch deal with the imaginary and real part we separate them into two arrays
		shearlets_complex = to_torch_complex(self.shearlets)
		self.shearlets_complex = shearlets_complex
		self.torch_shearlets = torch.tensor(shearlets_complex[np.newaxis, :,:,:,:]).to(device).float()
	
	def xform(self, entry):
		"""
		Uses Broadcasting way to compute Shearlet Xform - - Fourier Conv. Theorem
		20x faster for single batch Sh. X. form.
		200x faster for large batches for Sh. Xform.
		 entry = (Tensor, label) - i.e. CustomDataset like
		"""
		#print("<.>: Using Broadcasted version of Xform")
		image, label = entry
		#print("type:", type(image), "shape:", image.shape, "ele type:", type(image[0,0]))
		torch_shearlets = self.torch_shearlets
		#print("type shearlets:", type(torch_shearlets), "shape:", torch_shearlets.shape, "ele type:", torch_shearlets[0,0,0,0,0].type())
		#for pytorch we separate imaginary and real part into two arrays 
		
		image_complex = to_torch_complex4Img(image)
		#print("image_complex:", type(image_complex), "shape:", image_complex.shape, "ele type:", type(image_complex[0,0,0]))

		torch_image = torch.FloatTensor(image_complex[np.newaxis, :,:,:]).to(self.device)
		#print("torch_image:", type(torch_image), "shape:", torch_image.shape, "ele type:", torch_image[0,0,0,0].type())
		#t = time.time()
		torch_coeffs = torchsheardec2D_broadcast(torch_image, torch_shearlets)
		#print("torch_coeffs type:", type(torch_coeffs), "shape:", torch_coeffs.shape, "ele type:", torch_coeffs[0,0,0,0,0].type())
		#print("\n\n\n")
		if (self.tocplx):
			assert (torch_coeffs.shape[4] == 2) #the 5th dim holds real and complex.
			torch_coeffs = cplxmodule.cplx.Cplx(torch_coeffs[:,:,:,0], torch_coeffs[:,:,:,1])
		
		if (self.topolar):
			mag, phase = cplx.utils.complex2magphase(batch_coeffs) #TODO: not use general functions.
			if (self.phase_first):
				torch_coeffs = torch.stack((phase, mag), dim = 3)
			else:
				torch_coeffs = torch.stack((mag, phase), dim = 3)
		return torch_coeffs

	def batch_xform(self, batch):
		"""
		Transform a whole batch (stacked vertically(axis = 0)) of images as ndarray(batchsize, w, h).
		"""
		batch_complex = to_torch_complex(batch) #it is 3 dimensional. So we can use to_torch_complex.
		torch_batch = torch.FloatTensor(batch_complex).to(self.device)
		batch_coeffs = torchsheardec2D_broadcast(torch_batch, self.torch_shearlets)
		if (self.tocplx):
			#assert (batch_coeffs.shape[4] == 2) #the 5th dim holds real and complex.
			batch_coeffs = cplxmodule.cplx.Cplx(batch_coeffs.real, batch_coeffs.imag)
			
		if (self.topolar):
			mag, phase = batch_coeffs.abs(), batch_coeffs.angle() #TODO: Not use General functions.
			if (self.phase_first):
				batch_coeffs = torch.stack((phase, mag), dim = 4)
			else:
				batch_coeffs = torch.stack((mag, phase), dim = 4)
		return batch_coeffs

	def xform2(self, image):
		""" image = ndarray """
		#NOTE: *** this is only here for running the unit test in here - not for real use
		torch_shearlets = self.torch_shearlets

		#for pytorch we separate imaginary and real part into two arrays 
		image_complex = to_torch_complex4Img(image)
		torch_image = torch.FloatTensor(image_complex[np.newaxis, :,:,:]).to(self.device)

		#t = time.time()
		torch_coeffs = torchsheardec2D(torch_image, torch_shearlets)
		#print(f"Elapsed time torchsheardec2D(): {time.time()-t:3f}ms")		
		return torch_coeffs

# ## Relevant CoShReM function

# Single shearlet computation
def _single_shearlet(
	rows, cols, 
	wavelet_eff_supp,
	gaussian_eff_supp, scales_per_octave,
	shear_level, alpha, 
	sample_wavelet_off_origin,
	scale, ori, _coneh, _ks, hilbert_matrix
):
	shearlet_f = construct_shearlet(rows, cols, wavelet_eff_supp,
								  gaussian_eff_supp, scales_per_octave,
								  shear_level, alpha,
								  sample_wavelet_off_origin, scale, ori, _coneh, _ks)

	if ori in _coneh:
		shearlet_f = shearlet_f + (hilbert_matrix[:, :, 0] * shearlet_f)
		# shearlet_f = np.fliplr(np.flipud(_hilbert_f(shearlet_f * -1)))
		# if not self.sampleWaveletOffOrigin:
		#     shearlet_f = np.roll(shearlet_f, -1, axis=0)
	else:
		if ori > np.max(_coneh):
			shearlet_f = -1 * (shearlet_f + hilbert_matrix[:, :, 1] * shearlet_f)
			# shearlet_f = _hilbert_f(shearlet_f.T * -1).T
			# shearlet_f = np.roll(shearlet_f, 1, axis=1)
		else:
			shearlet_f = shearlet_f + hilbert_matrix[:, :, 1] * shearlet_f
			# shearlet_f = _hilbert_f(shearlet_f.T).T
	return shearlet_f

def getcomplexshearlets2D(
	rows, 
	cols, 
	scales_per_octave, 
	shear_level, 
	octaves, alpha,
	wavelet_eff_support = 7,
	gaussian_eff_support = None,
):

	# Parameters precomputing
	gaussian_eff_support = gaussian_eff_support if gaussian_eff_support else wavelet_eff_support * 2 
	wavelet_eff_supp = np.min((rows, cols)) / wavelet_eff_support
	gaussian_eff_supp = np.min((rows, cols)) / gaussian_eff_support
	sampleWaveletOffOrigin= True

	hilbert_matrix = np.ones((rows, cols, 2))
	hilbert_matrix[:(rows//2), :, 0] = -1
	hilbert_matrix[:, (cols//2):, 1] = -1
	n_oris = 2 ** shear_level + 2
	scales = np.arange(1, (scales_per_octave * octaves) + 1)
	n_shearlets = len(scales) * n_oris

	normalize=True
	_, _coneh, _ks =  cone_orientation(shear_level)
	shearlets = np.zeros((rows, cols,n_shearlets), dtype=np.complex_)
	shearletidx = []
	# Computing shearlets
	for j, scale in enumerate(scales):
		for ori in range(n_oris):
			shearlet = _single_shearlet(rows, cols, wavelet_eff_supp,
						   gaussian_eff_supp, scales_per_octave,
						   shear_level, alpha, sampleWaveletOffOrigin,
						   scale, ori+1, _coneh, _ks, hilbert_matrix)

			if ori in _coneh:
				shearletidx.append([1,int(scale), _ks[ori]])
			else:
				shearletidx.append([2,int(scale), _ks[ori]])
			shearlets[:, :, n_oris * j + ori] = shearlet
	# Computing RMS (Root mean square)
	RMS = np.linalg.norm(shearlets[0], axis=(0, 1))/np.sqrt(rows*cols)
	dualFrameWeights = np.sum(np.power(np.abs(shearlets), 2), axis=2)
	return shearlets, shearletidx, RMS, dualFrameWeights


def sheardec2D(X, shearlets):
	"""Shearlet Decomposition function."""
	# ## Classical way to compute the coefficients

	# Classically one compute the coefficients by using the convolution
	# 
	# $$
	# \text{SH}(f)_{j,k,m}^{2D} (f) = \overline{\psi^{d}_{j,k,m}}\ast f = \widehat{\psi^{d}_{j,k,m}}\cdot \widehat{f}
	# $$
	# 
	# where the last equality is obtained by using the Fourier convolutional theorem. Since Fourier is fast to compute the second identity is better to use.

	coeffs = np.zeros(shearlets.shape, dtype=complex)
	Xfreq = fftshift(fft2(ifftshift(X)))
	for i in range(shearlets.shape[2]):
		coeffs[:, :, i] = fftshift(ifft2(ifftshift(Xfreq * np.conj(
								   shearlets[:, :, i]))))
	return coeffs

def shearrec2D(coeffs, shearlets, dualFrameWeights):
	"""Shearlet Reconstruction function."""
	Xfreq = np.zeros((coeffs.shape[0], coeffs.shape[1]), dtype=complex)
	for i in range(coeffs.shape[2]):
		Xfreq += fftshift(fft2(ifftshift(coeffs[:,:,i])))* np.conj(shearlets[:, :, i])
	InversedualFrameWeights = 1 /dualFrameWeights;
	InversedualFrameWeights[InversedualFrameWeights==np.inf] = 0.0
	return fftshift(ifft2(ifftshift(InversedualFrameWeights*Xfreq)))


# ## Compute the coefficients purely in pytorch

# Besides of being a deep learning framework, pytorch also serves as a platform to do differentiable programming, 
# which is at the same time scalable and multiplatform, allowing us to do computations in the gpu with no need to write a kernel.
# 
# FFT is implemented in pytorch, just not fftshift and ifftshift which deal with centering the zero frequencies

#import torch

# Since we need to use ifftshift and fftshift, we can easily implement it in torch using the roll function for circ shift
def roll_n(X, axis, n):
	f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
	b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
	front = X[f_idx]
	back = X[b_idx]
	return torch.cat([back, front], axis)

def batch_fftshift2d(x):
	""" shift with circular wrap (toroidal) """
	real, imag = torch.unbind(x, -1)
	for dim in range(1, len(real.size())):
		n_shift = real.size(dim)//2
		if real.size(dim) % 2 != 0:
			n_shift += 1  # for odd-sized images
		real = roll_n(real, axis=dim, n=n_shift)
		imag = roll_n(imag, axis=dim, n=n_shift)
	return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def batch_fftshift2d_broad(x):
	real, imag = torch.unbind(x, -1)
	for dim in range(1, len(real.size())-1):
		n_shift = real.size(dim)//2
		if real.size(dim) % 2 != 0:
			n_shift += 1  # for odd-sized images
		real = roll_n(real, axis=dim, n=n_shift)
		imag = roll_n(imag, axis=dim, n=n_shift)
	return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def batch_ifftshift2d(x):
	real, imag = x.real, x.imag
	for dim in range(len(real.size()) - 1, 0, -1):
		real = roll_n(real, axis=dim, n=real.size(dim)//2)
		imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
	return torch.complex(real, imag)  # last dim=2 (real&imag

def batch_ifftshift2d_broad(x):
	if x.dtype == torch.complex64:
		real, imag = x.real, x.imag
	else:
		real, imag = torch.unbind(x, -1)
	for dim in range(len(real.size()) - 2, 0, -1):
		real = roll_n(real, axis=dim, n=real.size(dim)//2)
		imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)

	return torch.complex(real, imag)  # last dim=2 (real&imag

def to_torch_complex(shearlets):
	"""for pytorch to deal with the imaginary and real part we separate them into two arrays """
	#print(f"to_torch_complex {shearlets.shape}")
	shearlets_complex = np.concatenate((
		shearlets.real[:,:,:,np.newaxis], 
		shearlets.imag[:,:,:,np.newaxis]), 
		3
	)
	return shearlets_complex

def to_torch_complex4Img(image):
	"""for pytorch to deal with the imaginary and real part we separate them into two arrays """
	#print(f" to_torch_complex4Img {image.shape}")
	image_complex = np.concatenate(
		(image[:,:,np.newaxis], np.zeros(image.shape)[:,:,np.newaxis]), 
		2
	)
	return image_complex

def dot_complex(x, y):
	xreal, ximag = x.real, x.imag
	yreal, yimag = y.real, y.imag
	
	xyreal =  xreal*yreal + ximag*yimag
	xyimag = -xreal*yimag + ximag*yreal
	return torch.complex(xyreal, xyimag)

def torchsheardec2D(torch_X,  torch_shearlets):
	"""Shearlet Decomposition function."""
	#print(f" torchsheardec2D {torch_X.shape}")
	coeffs = torch.zeros((torch_X.shape[0], torch_X.shape[1], torch_X.shape[2],
					 torch_shearlets.shape[3], torch_X.shape[3]))
	Xfreq = batch_ifftshift2d(torch.fft(batch_fftshift2d(torch_X), 2))
	for i in range(torch_shearlets.shape[3]):
		coeffs[:, :, :, i, :] = batch_ifftshift2d(torch.ifft(batch_fftshift2d(
			dot_complex(Xfreq, torch_shearlets[:,:,:,i,:])), 2))
	return coeffs


# port a signature like torch.fft() (from old pytorch) but computes using torch.fft.fft() (new pytorch).
def dispatch_my_fourier_args(signal_ndim, normalized = False):
	"""
	To port between torch's old fft function (torch 1.6) to new fft.fftn() function.
	"""
	if signal_ndim < 1 or signal_ndim > 3:
		print("Signal ndim out of range, was", signal_ndim, "but expected a value between 1 and 3, inclusive")
	
	dims = (-1)
	if signal_ndim == 2:
		dims = (-2, -1)
	if signal_ndim == 3:
		dims = (-3, -2, -1)
	
	norm = "backward"
	if normalized:
		norm = "ortho"
	return dims, norm

def my_fft(input, signal_ndim, normalized = False):
	dims, norm = dispatch_my_fourier_args(signal_ndim, normalized)
	return torch.fft.fftn(torch.view_as_complex(input), dim = dims, norm = norm)

def my_ifft(input, signal_ndim, normalized = False):
	dims, norm = dispatch_my_fourier_args(signal_ndim, normalized)
	return torch.fft.ifftn(input, dim = dims, norm = norm)


def torchsheardec2D_broadcast(torch_X,  torch_shearlets):
	"""Shearlet Decomposition function."""
	Xfreq = batch_ifftshift2d(my_fft(batch_fftshift2d(torch_X), 2))
	coeffs = batch_ifftshift2d_broad(
				my_ifft(
					batch_ifftshift2d_broad(
						dot_complex(Xfreq[:,:,:,np.newaxis],
						torch.view_as_complex(torch_shearlets))
					).permute(0,3,1,2),
					2
				).permute(0,2,3,1)
			)
	return coeffs

	
if __name__ == '__main__':
	import matplotlib.pyplot as plt
	from skimage import data
	from skimage.transform import resize

	size = 256
	print(type(data.coins()))
	image = resize(data.coins(), (size,size))

	plt.figure(figsize = (6,6))
	plt.axis("off")
	#plt.imshow(image, cmap = "gray")
	#plt.show()

	# Relevant parameters for our Shearlet system
	rows, cols = image.shape
	sh_spec = ksh_spec.copy()
	sh_spec['rows'] = rows
	sh_spec['cols'] = cols
	print(f"sh_spec: {sh_spec}")

	# Generating the shearlet system with pyCoShRem
	device = torchutils.onceInit(kCUDA=True)
	coshxform = CoShXform(sh_spec)
	coshxform.start(device)

	shearlets, shearletIdxs, RMS, dualFrameWeights = coshxform.shearletSystem

	# The shearlets have 56 slices
	print(f"shearlets.shape {shearlets.shape}")

	j = 3
	shearlet = shearlets[:,:,j]

	qUpper = np.percentile(np.absolute(shearlet), 98)
	#cplx.visual.complexImageShow(shearlet/qUpper)
	#plt.show()

	# ### Visualizing the shearlets in spatial domain
	shearlet_space = fftshift(ifft2(ifftshift(shearlet)))

	qUpper = np.percentile(np.absolute(shearlet_space), 98)
	#cplx.visual.complexImageShow(shearlet_space/qUpper)
	#plt.show()

	#3: Classical way to compute the coefficients
	t = time.time()
	coeffs = sheardec2D(image, shearlets)
	print(f"Elapsed time: sheardec2D() {time.time()-t:3f}ms")

	#3.1: The coefficients are
	j = 10
	coeff = coeffs[:,:,j]
	shearlet_space = fftshift(ifft2(ifftshift(shearlets[:,:,j])))

	qUpper = np.percentile(np.absolute(coeff), 98)
	#cplx.visual.complexImageShow(coeff/qUpper)

	#3.2: Coming from the filter
	qUpper = np.percentile(np.absolute(shearlet_space), 98)
	#cplx.visual.complexImageShow(shearlet_space/qUpper)
	#plt.show()

	#4: use pytorch 
	# In order to make pytorch deal with the imaginary and real part we need to separate them in two different arrays
	shearlets_complex = coshxform.shearlets_complex
	torch_shearlets	  = coshxform.torch_shearlets
	device = coshxform.device

	# CUDA for PyTorch
	if False:
		use_cuda = torch.cuda.is_available()
		device = torch.device("cuda:0")
		torch.backends.cudnn.enabled = True

		torch_shearlets = torch.tensor(shearlets_complex[np.newaxis, :,:,:,:]).to(device)

		# We need to do the same for the image
		image_complex = np.concatenate((image[:,:,np.newaxis], np.zeros(image.shape)[:,:,np.newaxis]), 2)
		torch_image = torch.tensor(image_complex[np.newaxis, :,:,:]).to(device)

		t = time.time()
		torch_coeffs = torchsheardec2D(torch_image, torch_shearlets);
		print(f"Elapsed time torchsheardec2D(): {time.time()-t:3f}ms")

	torch_coeffs = coshxform.xform((image, None))
#	torch_coeffs = coshxform.xform2(image)
	
	#Batch xform trials.
	batch_size = 64
	t1_batch = time.time()
	batch = np.repeat(image[np.newaxis,:,:], batch_size, axis = 0)
	print(f"batch type: {type(batch)}")
	print(f"my batch img shape: {batch.shape}")
	batch_torch_coeffs = coshxform.batch_xform(batch)
	print(f"Batch Xform took : {time.time() - t1_batch} s for a batch of size {batch_size}")
	
	# Comparison with the other for one batch
	t = time.time()
	coeffs = sheardec2D(image, shearlets)
	print(f"Elapsed time sheardec2D(): {time.time()-t:3f}ms")

	# ### Visualizing the coefficients produced by torch
	coeffs_numpy = torch_coeffs.detach().cpu().numpy()


	j = 10
	coeff = coeffs_numpy[:,:,j]
	qUpper = np.percentile(np.absolute(coeff), 98)
	#cplx.visual.complexImageShow(coeff/qUpper)

	# Comparison with the one computed classically
	j = 10
	coeff = coeffs[:,:,j]
	qUpper = np.percentile(np.absolute(coeff), 98)
	#cplx.visual.complexImageShow(coeff/qUpper)	

	# ## Larger Batch to show advantage of using the pytorch only method
	image_complex = np.concatenate((image[:,:,np.newaxis], np.zeros(image.shape)[:,:,np.newaxis]), 2)
	print(f"batch img complex shape: {image_complex.shape}")
	# Image batch done with repeating the same image 20 times
	batch_image = np.repeat(image_complex[np.newaxis,:,:,:], 20, axis=0)
	print(f"batch image shape: {batch_image.shape}")
	torch_image = torch.tensor(batch_image).to(device)

	#t = time.time()
	#torch_coeffs = torchsheardec2D(torch_image, torch_shearlets);
	#print(f"Elapsed time: torchsheardec2D() larger batch {time.time()-t:3f}ms")

	#  Comparison with doing the same thing outside pytorch
	t = time.time()
	for j in range(20):
		coeffs = sheardec2D(image, shearlets)
	print(f"Elapsed time: sheardec2D() larger batch {time.time()-t:3f}ms")

	#plt.show(block=True)
	#Complex Cplx testing.
	coshxform = CoShXform(sh_spec, tocplx = True)
	coshxform.start(device)
	#torch_coeffs = coshxform.xform((image, None))
	t1 = time.time()
	batch_coeffs = coshxform.batch_xform(batch)
	print(f"cplex batchxform took : {time.time() - t1} seconds.")