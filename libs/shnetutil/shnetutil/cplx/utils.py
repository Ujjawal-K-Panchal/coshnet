# -*- coding: utf-8 -*-
"""

Title: Complex Utilities.
	
Created on Mon Mar 16 17:44:29 2020

@author: Hector Andrade-Loarca & Manny Ko & Ujjawal.K.Panchal
"""
import numpy as np
import matplotlib.pyplot as plt

from cplxmodule import cplx

import torch
# from_real|to_real: Cplx to and from interleaved_real

#TODO: original written for np.complex
#  cplx.Cplx which also has .imag and .real. It might just work (need to test)

#for numpy.
def np_complex2magphase(z):
	""" complex to (mag, phase) 'z' is np.complex """
	mag = np.abs(z)
	phase = np.arctan2(z.imag, z.real)	#this handles the divide by 0
	#phase[np.isnan(phase)] = np.arctan(np.inf)
	return mag, phase

def np_magphase2complex(mag, phase):
	""" (mag, phase) -> complex """
	return mag*(np.cos(phase) + 1j*np.sin(phase))

#for cplx. (From Ivannz's repo.)
def cplx_complex2magphase(z):
	real, imag = z.real, z.imag #adjusted to work with cplx | Torch.
	torch_phase = z.angle
	torch_mag = torch.sqrt(real**2+imag**2)
	return torch_mag, torch_phase

def cplx_complex2mag(z):
	#refer: https://en.wikipedia.org/wiki/Probability_amplitude
	#x_mag := |\Phi|^2$ or |\Phi|
	mag, _ = cplx_complex2magphase(z)
	return mag

def cplx_magphase2complex(mag, phase):
	return cplx.Cplx(mag * torch.cos(phase), mag * torch.sin(phase))

#for torch tensors.
def torch_complex2magphase(z):	
	real, imag = z[...,0], z[...,1]
	torch_phase = torch.atan2(imag, real)
	torch_mag = torch.sqrt( (real ** 2) + (imag ** 2))
	return torch_mag, torch_phase

def torch_magphase2complex(torch_mag, torch_phase):
	real = torch_mag*torch.cos(torch_phase)
	imag = torch_mag*torch.sin(torch_phase)
	coeffs = torch.stack((real, imag), dim = len(real.shape) - 1)
	return coeffs #adjusted to work with cplx | torch.

#type modifiers.
def numpy2cplx(z):
	"""
	Converts numpy complex array to Cplx tensor.
	"""
	return cplx.Cplx(torch.tensor(z.real), torch.tensor(z.imag))

def cplx2numpy(z):
	"""
	Converts a cplx.Cplx number to numpy complex.
	"""
	return z.real.cpu().detach().numpy() + 1j *  z.imag.cpu().detach().numpy()

#general.
def complex2magphase(z):
	if isinstance(z, np.ndarray):
		return np_complex2magphase(z)
	elif torch.is_tensor(z):
		return torch_complex2magphase(z)
	elif isinstance(z, cplx.Cplx):
		return cplx_complex2magphase(z)
	else:
		raise Exception("Error <!>: input type is not supported.")
	return None

def magphase2complex(mag, phase):
	if isinstance(mag, np.ndarray) and isinstance(phase, np.ndarray):
		return np_magphase2complex(mag, phase)
	elif torch.is_tensor(mag) and torch.is_tensor(phase):
		return torch_magphase2complex(mag, phase)
	else:
		raise Exception("Error <!>: input type is not supported.")
	return None

def cplx_bmm(a, b):
	"""
	The batchwise matrix multiplication extended for cplx numbers. 
	"""
	real = torch.matmul(a.real, b.real) - torch.matmul(a.imag, b.imag)
	imag = torch.matmul(a.real ,b.imag) + torch.matmul(a.imag, b.real)
	return cplx.Cplx(real, imag)

def cplx_pow(z, n):
    """
    z^n = (re^{Iθ})^n = r^n e^{inθ}
    """
    #1. to magnitude and phase.
    r, theta = cplx_complex2magphase(z)
    #2. calculating individual components.
    r_n = r ** n
    n_theta = theta * n
    #3. returning to cplx.
    z_n = cplx_magphase2complex(r_n, n_theta)
    return z_n

def cplx_plot_hist(cplxTensor: cplx.Cplx, title: str = "bias"):
	plt.title(f"{title} real valued histogram.")
	plt.hist(cplxTensor.real.cpu().detach().numpy())
	plt.show()
	plt.title(f"{title} imaginary valued histogram.")
	plt.hist(cplxTensor.imag.cpu().detach().numpy())
	plt.show()
	return