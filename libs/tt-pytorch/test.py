"""
Title: tensor layers.

Description: These are Split CVnns derived from Trabelsi et al., 2018.

Created on Thur Jul 29 14:48:29 2021

@author: Ujjawal.K.Panchal.
"""
import cmath, numpy as np
import torch
import t3nsor
import cplxmodule.nn as cplxnn
from cplxmodule import cplx
from shnetutil.cplx import cplxlayer, layers
from typing import Iterable
from shnetutil import torchutils

def copy_tt_cores(dst: Iterable, src: Iterable):
	assert (len(src) == len(dst)), "Error<!>: Length of source not the same as length of destination."

	for i in range(len(dst)):
		print(f"dst: {dst[i].shape}, src: {src[i].shape}")
		dst[i] = torch.nn.Parameter(src[i].permute(1, 0, 3, 2))# src[i].reshape(*dst[i].shape)
	return

if __name__ == "__main__":
	torchutils.onceInit()
	#1. initialize split complex FC Layer with weights of choice.
	fc_shape = (1250, 500)
	fc = cplxlayer.CplxLinear(1250, 500)
	fc.initWeights(cplxnn.init.cplx_trabelsi_standard_, kind = "xavier")

	#2. get real and imaginary tensor trains corresponding to fc layer's real and imaginary split.
	in_factors = (50, 25)
	out_factors = (20, 25)
	ranks = (100,)

	tt_real = t3nsor.decompositions.to_tt_matrix(fc.weight.real, shape = [[50, 25], [20, 25]], max_tt_rank = 100, epsilon = 0.0001) #is this legit?
	tt_imag = t3nsor.decompositions.to_tt_matrix(fc.weight.imag, shape = [[50, 25], [20, 25]], max_tt_rank = 100, epsilon = 0.0001) #is this legit?

	#3. show details of the corresponding tensor trains.
	print("TT SVD train:")
	print(f"type: {type(tt_real)}, num_cores: {len(tt_real.tt_cores)}")
	for i, core in enumerate(tt_real.tt_cores):
		print(f"core#{i} core_shape: {core.shape}")

	print(f"tt ranks: {tt_real.ranks}")




	#4. copy cores to SplitTTLayer approximation of choice.
	print("Approximating using CplxSplitTTLayer:")
	tt_layer = layers.CplxSplitTTLayer(
				in_factors = in_factors,
				out_factors = out_factors,
				ranks = ranks,
				ein_string = "nab, aoix, bipy",
				split = True
			)

	print("CplxSplitTTLayer details:")
	print(f"type: {type(tt_layer)}, num_cores: {len(tt_layer.r_splitlayer.cores)}")
	for i, core in enumerate(tt_layer.r_splitlayer.cores):
		print(f"core#{i} core_shape: {core.shape}")



	#5. copying weights.
	before_real_tt_cores = torch.cat([core.flatten() for core in tt_layer.r_splitlayer.cores]).clone()
	#tt_layer.r_splitlayer.cores = tt_real.to_parameter().parameter
	#tt_layer.i_splitlayer.cores = tt_imag.to_parameter().parameter
	copy_tt_cores(dst = tt_layer.r_splitlayer.cores, src = tt_real.tt_cores)
	copy_tt_cores(dst = tt_layer.i_splitlayer.cores, src = tt_imag.tt_cores)
	after_real_tt_cores = torch.cat([core.flatten() for core in tt_layer.r_splitlayer.cores])
	
	print(f"Initialization takes effect if this value is non-zero: {sum(before_real_tt_cores - after_real_tt_cores)}")


	#6. Check near-equality of response maps.
	real_x, imag_x = torch.randn(1, 1250), torch.randn(1, 1250)

	x = cplx.Cplx(real_x.clone(), imag_x.clone())
	clone_x = cplx.Cplx(real_x.clone(), imag_x.clone())

	op_tt_layer = tt_layer(clone_x)
	op_linear_layer = fc(x)	

	op_linear_layer = op_linear_layer.real.detach().flatten().numpy() + 1j* op_linear_layer.imag.detach().flatten().numpy()
	op_tt_layer = op_tt_layer.real.detach().flatten().numpy() + 1j* op_tt_layer.imag.detach().flatten().numpy()

	difference = op_linear_layer - op_tt_layer


	equality = [cmath.isclose(z1, z2) for z1, z2 in zip(op_linear_layer, op_tt_layer)]
	print(f"equality of complex output of linear and tt layer: {np.all(equality)}")	

	print(f"mean difference between weight initialized linear layer and it's tensor train decomposition: mean: {difference.mean()}, median: {np.median(difference)}")
	print(f"for comparison the mean of linear layer output is: mean: {op_linear_layer.mean()}, median: {np.median(op_linear_layer)}")
	print(f"for comparison the mean of TT layer output is: mean: {op_tt_layer.mean()}, median: {np.median(op_tt_layer)}")	