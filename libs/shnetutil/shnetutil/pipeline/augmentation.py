# -*- coding: utf-8 -*-
"""

Title: Shearlet based CNN vs. simple MLP in Fashion MNIST.
	
Created on Sun Aug 4 17:44:29 2020

@author: Manny Ko & Ujjawal.K.Panchal 

"""
import numpy as np, random
import abc
import skimage
import torch

from typing import Union

from functools import partial
from skimage import transform, restoration
from scipy import ndimage

from .patching import get_mp_patches, find_patch

from .. import coshrem_xform
from cplxmodule import cplx

from ..cplx import utils as shcplxutils


def transpose4Np(imgList):
	if (len(imgList.shape) == 3): #for gray imgs.
			imgList = imgList[:,np.newaxis, :, :]
	elif (len(imgList.shape) == 4): #support for colored images.
		imgList = np.transpose(imgList, (0,3,1,2))
	elif (len(imgList) == 5): #support for complex colored images.
		imgList = np.transpose(imgList, (0,4,3,1,2))
	else: #more dim support for some extra computation cost....
		dims = [dim for dim in range(len(imgList.shape))]
		top_dim = dims[0]
		bottom_dims = [dim for dim in dims[1:3]]
		channel_dims = list(dims[3:])[::-1]
		new_dims = []
		new_dims.append(top_dim)
		new_dims.extend(channel_dims)
		new_dims.extend(bottom_dims)
		imgList = np.transpose(imgList, tuple(new_dims))
	return imgList

def transpose4Torch(imgList):
	if (len(imgList.shape) == 3): #for gray image.
			imgList.unsqueeze_(1)
	elif (len(imgList.shape) == 4): #support for color imgages.
		imgList = imgList.permute(0,3,1,2)
	elif (len(imgList) == 5): #support for complex colored images.
		imgList = imgList.permute(0,4,3,1,2)
	else:
		dims = [dim for dim in range(len(imgList.shape))]
		top_dim = dims[0]
		bottom_dims = [dim for dim in dims[1:3]]
		channel_dims = list(dims[3:])[::-1]
		new_dims = []
		new_dims.append(top_dim)
		new_dims.extend(channel_dims)
		new_dims.extend(bottom_dims)
		imgList = imgList.permute(*new_dims) #args to open new_dims list.
	return imgList

def transpose4Cplex(imgList):	
	imgListReal = transpose4Torch(imgList.real)
	imgListImag = transpose4Torch(imgList.imag)
	imgList = cplx.Cplx(imgListReal, imgListImag)
	return imgList


class NumpyToTorchTensor(object):
	def __init__(self, dtype = torch.float32):
		self.dtype = dtype
		return
	def __call__(self, x):
		x = torch.tensor(x, dtype = self.dtype)
		return x

class SendToDevice(object):
	def __init__(self, device):
		self.device = device
		return
	def __call__(self, x):
		x = x.to(self.device)
		return x


class PrintDims(object):
	def __init__(self):
		pass
	def __call__(self, images):
		print(f"{images.shape=}")
		return images

class NumpyTranspose(object):
	def __init__(self, tpose_dims: tuple):
		self.tpose_dims = tpose_dims
		pass
	def __call__(self, images):
		images = images.transpose(self.tpose_dims)
		return images

class MakeContiguous(object):
	def __init__(self):
		pass
	def __call__(self, images):
		images = images.contiguous()
		return images

class ToTorchDims(object):
	"""
	Transposes the given List of Images' dims from : H,W,C -> C,H,W.
	"""
	def __init__(self):
		pass
	def __call__(self, imglist):
		return self.toTorchDims(imglist)

	#dispatch table for toTorchDims()	
	dispatch = {
		np.ndarray: 	transpose4Np,
		torch.Tensor: 	transpose4Torch,
		cplx.Cplx:		transpose4Cplex,
	}	
	def toTorchDims(self, imgList):
		"""
		Transpose of batch for training on torch.
		(H, W, C) -> (C, H, W)
		"""
		transposeOp = ToTorchDims.dispatch.get(type(imgList), None)
		if transposeOp is None:
			raise Exception("Error<!>: Unknown datatype in torch transposition!")
		imgList = transposeOp(imgList)

		return imgList
	def __str__(self):
		return f"ToTorchDims()"	

class ToTorchTensorDCF(object):
	"""
	converts the np array into torch tensor and adds channel dimension for DCF networks.
	"""
	def __init__(self):
		pass
	def __call__(self, imglist):
		imglist = imglist[:,np.newaxis, :, :]
		return torch.Tensor(imglist)


class Base(metaclass=abc.ABCMeta):
	""" Null xform 
	---import shnetutil.cplx as shcplx

	Args: (N/A).

	"""
	def __init__(self, **kwargs):
		self.kwargs = kwargs
		pass

	def __call__(self, sample):
		return sample

	def rewind(self):
		pass	

class NullXform(Base):
	""" Null xform 
	---import shnetutil.cplx as shcplx

	Args: (N/A).

	"""
	def __init__(self, **kwargs):
		self.kwargs = kwargs
		pass

	def __call__(self, sample):
		return sample

class Patch(Base):
	"""
	Patch a given set of Numpy Images.
	---
	Args:
		1. patch_size: size of patches to extract.
		2. inv_olap: inverse overlap factor for patching.
		3. num_workers (default (num of CPU cores)=None): number of processes for parallelization.
		4. verbose (default = 0): default verbose level. 
	"""
	def __init__(self, patch_size, stride, num_workers = None, verbose = 0, param_strategy = 0):
		self.patcher = partial(get_mp_patches, patch_func= find_patch, 
			   patch_size = patch_size, stride = stride, num_workers = num_workers, 
			   verbose = verbose, param_strategy = param_strategy)
	
	def __call__(self, images):
		return self.patcher(dset = images)


class Rescale(Base):
	"""Rescale the image in a sample to a given size.
	---
	Args:
		1. output_size (tuple or int): Desired output size. If tuple, output is
			matched to output_size. If int, smaller of image edges is matched
			to output_size keeping aspect ratio the same.
	"""

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	def __call__(self, sample):
		image, label = sample

		h, w = image.shape[:2]
		if isinstance(self.output_size, int):
			if h > w:
				new_h, new_w = self.output_size * h / w, self.output_size
			else:
				new_h, new_w = self.output_size, self.output_size * w / h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		img = skimage.transform.resize(image, (new_h, new_w))	#use skimage.transform

		return img, label

class RescaleAllAtOnce(Base):
	"""
	Takes an image and resizes it's H, W to given size.
	---
	Args:
		1. size := tuple of sizes wanted on dim of H and W. 
	"""
	def __init__(self, size):
		self.sizeH = size[0]
		self.sizeW = size[1]
	def __call__(self, batch, hdim =1, wdim = 2):
		factorH = self.sizeH/batch.shape[hdim]
		factorW = self.sizeW/batch.shape[wdim]
		batch = ndimage.zoom(batch, (1, factorH, factorW), order = 2)
		return batch


class RandomShift(Base):
	"""Randomly shift the images in a batch.
	---
	Args:
		1. shift_max: max value by which to shift along axes (float or list).
	"""
	def __init__(self, max_shift_perc: int = 10, seed = 42):
		self.max_shift_perc = max_shift_perc
		self.seed = seed
		self.setNumpyRandomState_()

		self.ops = {
		1: self.translate_up,
		2: self.translate_down,
		3: self.translate_right,
		4: self.translate_left,
		}
		return
	def __str__(self):
		return f"RandomShift({self.max_shift_perc=})"
		
	def translate_up(self, images, shift_perc: int = 10):
		shift = int(images.shape[1] * (shift_perc / 100))
		if shift == 0:
			return images
		#print(f"{shift=}")
		up_roll = images[:, :shift, ...].copy()
		images[:, :-shift, ...] = images[:, shift:, ...]
		images[:, -shift:, ...] = up_roll
		return images

	def translate_down(self, images, shift_perc: int = 10):
		shift = int(images.shape[1] * (shift_perc / 100))
		if shift == 0:
			return images
		#print(f"{shift=}")
		down_roll = images[:, -shift:, ...].copy()
		images[:, shift:, ...] = images[:, :-shift, ...]
		images[:, :shift, ...] = down_roll
		return images

	def translate_right(self, images, shift_perc: int = 10):
		shift = int(images.shape[2] * (shift_perc / 100))
		if shift == 0:
			return images
		#print(f"{shift=}")
		right_roll = images[:, :, -shift:, ...].copy()
		images[:, :, shift:, ...] = images[:, :, :-shift, ...]
		images[:, :, :shift, ...] = np.flip(right_roll, axis = 2)
		return images

	def translate_left(self, images, shift_perc: int = 10):
		shift = int(images.shape[2] * (shift_perc / 100))
		if shift == 0:
			return images
		#print(f"{shift=}")
		left_roll = images[:, :, :shift, ...].copy()
		images[:, :, :-shift, ...] = images[:, :, shift:, ...]
		images[:, :, -shift:, ...] = left_roll
		return images

	def setNumpyRandomState_(self, seed: Union[None, int] = None):
		s = seed if seed is not None else self.seed
		self.ran = np.random.RandomState(s)
		return

	def __call__(self, images):
		direction = self.ran.randint(1, 4 + 1)
		shift_perc = self.ran.randint(0, self.max_shift_perc + 1)
		#print(f"{shift_perc=}")
		shifted_images = self.ops[direction](images, shift_perc) if shift_perc > 0 else images
		return shifted_images

class RandomRot(Base):
	"""Randomly rotate the image in a batch.
	---
	Args:
		1. rot_range (tuple or int): Desired range of rotating angles. 
	"""
	def __init__(self, rot_range: Union[tuple, int] = (0, 360), seed: int = 0):
		if isinstance(rot_range, int):
			self.rot_range = (0, rot_range)
		else:
			assert len(rot_range) == 2
			self.rot_range = rot_range
		self.seed = seed
		self.setNumpyRandomState_(seed)
		return

	def setNumpyRandomState_(self, seed: Union[None, int] = None):
		s = seed if seed is not None else self.seed
		self.ran = np.random.RandomState(s)
		return

	def __call__(self, images):
		#1. get angle.
		angle = self.ran.randint(*self.rot_range)
		new_images = ndimage.rotate(images, angle, axes = (1,2), reshape = False)
		return new_images

	def __str__(self):
		return f"RandomRot({self.rot_range})"

class RandomFlip(Base):
	def __init__(self, seed = 42):
		self.flip_ops = {
			0: self.noop,
			1: self.flip_ud,
			2: self.flip_lr,
		}
		self.seed = seed
		self.setNumpyRandomState_(seed)
		return
	def __str__(self):
		return f"RandomFlip({self.seed=})"
	def setNumpyRandomState_(self, seed: Union[None, int] = None):
		s = seed if seed else self.seed
		self.ran = np.random.RandomState(s)
		return
	
	def noop(self, images):
		return images
	
	def flip_ud(self, images):
		return np.flip(images, axis = 1)
	
	def flip_lr(self, images):
		return np.flip(images, axis = 2)
	
	def __call__(self, images):
		op_index = self.ran.randint(0, 3)
		return self.flip_ops[op_index](images)

class RandomBlurSharpen(Base):
	def __init__(self, maxBlurSigma = 5, maxSharpenAlpha = 50, seed = 42):
		self.ops = {
			0: self.blur,
			1: self.sharpen
		}
		assert maxBlurSigma >=1, f"maxBlurSigma less than 1 doesn't give meaningful result. Got {maxBlurSigma}"
		self.maxBlurSigma = maxBlurSigma
		self.maxSharpenAlpha = maxSharpenAlpha
		self.seed = seed
		self.setNumpyRandomState_()
		return

	def __str__(self):
		return f"RandomBlurSharpen({self.maxBlurSigma=}, {self.maxSharpenAlpha=}, {self.seed=})"

	def setNumpyRandomState_(self, seed: Union[None, int] = None):
		s = seed if seed else self.seed
		self.ran = np.random.RandomState(s)
		return
	
	def blur(self, images, sigma, alpha = 0):
		"""
		alpha unused arg.
		"""
		return ndimage.gaussian_filter(images, sigma = sigma)

	def sharpen(self, images, sigma, alpha):
		blurred = self.blur(images, sigma = sigma + 2)
		less_blurred = self.blur(images, sigma = sigma)
		return blurred + alpha * (blurred - less_blurred)

	def __call__(self, images):
		rand_op = self.ran.randint(2)
		rand_sigma = self.ran.randint(1, self.maxBlurSigma +1)
		rand_alpha = self.ran.randint(10, self.maxSharpenAlpha + 1)
		return self.ops[rand_op](images, rand_sigma, rand_alpha)

class RandomCrop(Base):
	"""Crop randomly the image in a sample.
	---
	Args:
		1. output_size (tuple or int): Desired output size. If int, square crop
			is made.
	"""

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size

	def __call__(self, images):
		#image, label = sample

		h, w = images.shape[1:3]
		new_h, new_w = self.output_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)

		images = images[:,top: top + new_h,
					  left: left + new_w, :]
		return images

	def __str__(self):
		return f"RandomCrop({self.output_size})"	

class Rotate(Base):
	"""
	Rotate all images in a given image.
	---
	Args:
		1. angle: (int): angle in degrees by which you want to rotate your image.
	"""
	def __init__(self, angle: int):
		self.angle = angle
		
	def __call__(self, images):
		new_images = ndimage.rotate(images, self.angle, axes = (1,2), reshape = False)
		return np.array(new_images)

	def __str__(self):
		return f"Rotate({self.angle})"

class Sequential(Base):
	"""
	A Sequential class to do multiple xforms one after another.
	---
	Args:
		1. List of augmentation Xform objects.
	"""
	def __init__(self, xforms=[], device='gpu'):
		self.xforms = xforms
		self.device = device

	def __call__(self, x):
		for xform in self.xforms:
			x = xform(x)
			#print("image became:",x.shape, type(x))
		return x
	def __getitem__(self, index):
		return self.xforms[index]

	def __str__(self):
		mystr = f"Sequential Transform({self.device}):\n"
		for i, xform in enumerate(self.xforms):
			attrib_str = ""
			mystr += f"  {i}: {xform.__str__()}\n"
		return mystr

	def locate(target:Base) -> int:
		""" Locate the 'stage' in our pipeline and returns its index """
		index = -1
		for i, stage in enumerate(self.xforms):
			if (type(stage) == type(target)):
				index = i
		return index		

class CoShREM(Base):
	"""
	A class for implementing CoShREM based edge and ridge detector [1].
	---
	ref:
		[1]. Reisenhofer R., King E. (2019). Edge, Ridge and Blob detection with symmetric molecules.  
	---
	Args:
		1. **kwargs corresponding to ksh_spec for your Shearlet system.
			1.a. rows: H of the input matrix.
			1.b. cols: W of the input matrix.
			1.c. scales_per_octave: number of scales in frequency octaves.
			1.d. octaves: number of octaves.
			1.e. alpha: coefficient determining level of anisotropy. 
						0 -> wavelets, 
						0.5 -> shearlets, 
						1 -> ridgelets... etc.
	"""
	def __init__(self, 
		device = "cuda", 
		tocplx = False, 
		topolar = False, phase_first = False, 
		colorspace = "grayscale",
		coshrem_config = coshrem_xform.ksh_spec.copy(),
	):
		color2dims = {
			"grayscale": 1,
			"lum_lab": 1,
			"rgb": 3,
			"lab": 3,
		}
		self.device = device
		self.ksh_spec = coshrem_config
		self.coshxform = coshrem_xform.CoShXform(
									self.ksh_spec, tocplx = tocplx,
									topolar = topolar, phase_first = phase_first
						)
		self.coshxform.start(device)
		self.coshxform.color_aware_xform = self.graybatch_xform if color2dims[colorspace] == 1 else self.colorbatch_xform
		return

	def __call__(self, x):
		return self.coshxform.color_aware_xform(x)

	def graybatch_xform(self, x):
		""" 1-channel xform """
		x = self.coshxform.batch_xform(x)
		return x

	def colorbatch_xform(self, x):
		""" 3-channel xform """
		l1_real, l1_imag = [], []
		for i in range(x.shape[-1]):
			z = self.coshxform.batch_xform(x[...,i])
			l1_real.append(z.real)
			l1_imag.append(z.imag)
		x_real = torch.cat(l1_real, dim = 3)
		x_imag = torch.cat(l1_imag, dim = 3)
		x = cplx.Cplx(x_real, x_imag)
		return x

	def __str__(self):
		return str(self.coshxform)	

	@property
	def RMS(self):
		shearlets, shearletidxs, ourRMS, *_ = self.coshxform._shearletSystem
		return ourRMS

class GaussianNoise(Base):
	"""
	Apply Random Noise to Batch of Images.
	"""
	def __init__(self, mean: float = 0.0, variance: float = 0.0, device = 'cpu', seed = 0):
		self.mean = mean
		self.sigma = variance ** 0.5
		self.device = device
		self.seed = seed
		self.ran = np.random.RandomState(self.seed)
		return

	def __str__(self):
		return f"GaussianNoise(mean = {self.mean}, sigma = {self.sigma}, seed = {self.seed})"

	def __call__(self, x):
		x = x + self.ran.normal(self.mean, self.sigma, (x.shape[0], x.shape[1], x.shape[2]))
		return x

class GaussianBlur(Base):
	"""
	Apply Gaussian Noise to Batch of Images.
	"""
	def __init__(self, max_sigma: float = 0.0, device = 'cpu', seed = 0):
		self.max_sigma = max_sigma
		self.seed = seed
		self.ran = np.random.RandomState(self.seed)
		return

	def __str__(self):
		return f"GaussianBlur(max_sigma = {self.max_sigma}, seed = {self.seed})"

	def __call__(self, x):
		x = ndimage.gaussian_filter(x, sigma = self.ran.uniform(0.0, self.max_sigma))
		return x


class Denoise(Base):
	"""
	Class to implement one-step hard thresholding using the RMS for Sh Coeff. 
	Denoising.

	"""
	def __init__(self, thresholdingFactor, RMS, sigma = 0.00712, device = 'cpu', enablestats=True):
		self.thresholdingFactor = thresholdingFactor
		self.RMS = RMS
		self.sigma = sigma
		self.device = device
		self.enablestats = enablestats
		self.zeros = 0
		self.total = 0
#		print(f"Denoise(sigma {sigma})")

	def __call__(self, x):
		if self.sigma is None:
			self.sigma = restoration.estimate_sigma(cplx.to_concatenated_real(x.clone()).cpu().numpy())
		#print(f"sigma {self.sigma}")
		#T = self.thresholdingFactor * self.RMS * torch.ones(x.shape, requires_grad=False) * self.sigma
		T = self.thresholdingFactor * self.RMS * np.ones(x.shape) * self.sigma 	#for now keep np.ones() because the above has small numerical difference

		torch_T = torch.tensor(T[np.newaxis,:,:,:]).to(self.device)
		torch_mag, torch_phase = shcplxutils.cplx_complex2magphase(x)

		torch_mag_denoised = torch.relu(torch.relu(torch_mag-torch_T)).float().squeeze(0)

		if self.enablestats:
			coeffs_numpy = torch_mag_denoised.cpu().numpy()
			self.total += coeffs_numpy.size
			self.zeros += (coeffs_numpy.size - np.count_nonzero(coeffs_numpy))

		torch_coeffs_denoised = shcplxutils.cplx_magphase2complex(torch_mag_denoised, torch_phase)
		return torch_coeffs_denoised

	def __str__(self):
		return f"Denoise(sigma={self.sigma})"	


class MinMaxScaler(Base):
	"""
	MinMaxScale the image in a sample to [0,1].

	---
	Args:
		1. min: min value of the dataset.
		2. max: max value of the dataset.
	"""
	def __init__(self, min = 0, max = 255):
		self.min = min
		self.max = max
	def __call__(self, x):
		x = x.astype('float32')
		return (x - self.min) / (self.max - self.min)


class Normalize(Base):
	"""
	Rescale the image in a sample to [-1,1].

	---
	Args:
		1. mean (float): mean of the dataset - for centering.
		2. std (float): standard deviation.
	"""

	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def __call__(self, x):
		x = x.astype('float32')/255
		return (x - self.mean) / (self.std)

	def __str__(self):
		return f"Normalize({self.mean}, {self.std})"	

class ComplexNormalize(Base):
	"""
	Complex Normalization of the given image.
	x:= a+ib;
	mag:= sqrt(a^2 + b^2);
	x = x/mag = a/mag + i (b/mag)
	
	---
	Args:
		1. avg_mag := estimate of the average magnitude with which to normalize.
	"""
	def __init__(self, avg_mag):
		self.avg_mag = avg_mag

	def __call__(self, cplxnums):
		real = cplxnums.real / self.avg_mag
		imag = cplxnums.imag / self.avg_mag
		return cplx.Cplx(real, imag)

class RgbExtract(Base):
	"""
	Extract 1 channel from Rgb
	---
	Args:
		1. channel (int): channel index.
	"""
	def __init__(self, channel:int = 0):
		self.channel = 0

	def __call__(self, x):
		return x[:,:,self.channel]

	def __str__(self):
		return f"RgbExtract({self.channel})"	


class Pad(object):
	"""
	Pad the image by a given amount.

	---
	Args:
		1. sizes = pad sizes on respective axes.
		2. padval = value with which to conduct padding. (default is zero padding).
	"""
	def __init__(self, sizes, padval = 0):
		self.sizes = sizes
		self.padval = padval
		self.mode = 'constant'

	def __call__(self, x):
		x = np.pad(x, self.sizes, mode = self.mode, constant_values = self.padval)
		return x
	def __str__(self):
		return f"Pad({self.sizes}, {self.mode})"	

class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):
		image, label = sample

		# swap color axis because
		# numpy image: H x W x C
		# torch image: C X H X W
		image = image.transpose((2, 0, 1))
		return torch.from_numpy(image), torch.from_numpy(label)

	def __str__(self):
		return f"ToTensor()"	


class RepeatDepth(object):
	def __init__(self, n_times: int = 3, device= "cpu"):
		self.n_times = n_times
		self.device = device
	def __call__(self, x):
		x = torch.stack(self.n_times * [torch.tensor(x).clone().to(self.device)], dim = -1)
		return x


class NoShAblation(object):
	"""
	Repeat the object n times, make it cplx and pass it through the network, to replicate shearlet output dimension.
	---
	Args:
		1. n_channels = number of channels in the output cplx object.
	"""
	def __init__(self, n_channels: int = 10, device = "cpu", real = False):
		self.n_channels = n_channels
		self.device = device
		self.real = real

	def make_nChannelComplex(self, x):
		x_real, x_imag = torch.stack(self.n_channels * [torch.tensor(x).clone().to(self.device)], dim = -1),\
						 torch.stack(self.n_channels * [torch.tensor(x).clone().to(self.device)], dim = -1)
		x_cplx = cplx.Cplx(x_real, x_imag)
		return x_cplx

	def make_nChannelReal(self, x):
		x = torch.stack(self.n_channels * [torch.tensor(x).clone().to(self.device)], dim = -1)
		return x 
		
	def __call__(self, x):
		if self.real:
			x = self.make_nChannelReal(x)
		else:
			x = self.make_nChannelComplex(x)
		return x


class CaptureAugmentation(NullXform):
	""" Capture augmentations into a replay buffer """
	def __init__(self, cache):
		self.cache = cache
		self.capture = True
		self.reset()

	def reset(self):
		self.batchindex = 0

	def __call__(self, sample):
		if self.capture:
			self.cache.insert(sample)
			return sample
		batchN = self.batchindex
		self.batchindex += 1
		return self.cache[batchN]
	