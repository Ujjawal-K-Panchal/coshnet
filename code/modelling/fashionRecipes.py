"""
Title: fashionRecipe - complex valued NN based on COShREM.
	
Created on Tue Dec 7 10:44:29 2021

@author:  Manny Ko, Ujjawal K. Panchal.
"""
from collections import namedtuple
from collections.abc import KeysView
from typing import Callable, List, Tuple, Optional, Union
from pydantic import BaseModel, ValidationError, validator, root_validator

from cplxmodule.nn.init import cplx_trabelsi_standard_, cplx_trabelsi_independent_

from pyutils.enumutils import *
from shnetutil.cplx import layers

from . import CVnn


""" all the supported variations for CoShCVNN in Fashion with names """
kCoShMLPRecipe = CVnn.CVnnMLPrecipe(
	name  = 'CoShNetMLPrecipe',
	config= [(5 * 5 * 50, 500), (500, 10), (20, 10)],
	linear= CVnn.LinearLayers.kSplitLinear,
	init  = cplx_trabelsi_standard_,
	probtype = CVnn.ProbabilityKind.kReal,
	fc_comp = CVnn.FCLayerCompression.none,
	activation = CVnn.Activations.kReLU,
)
kGiantCoShMLPRecipe = CVnn.CVnnMLPrecipe(
	name = 'GiantCoShNetMLPrecipe',
	config = [(5 * 5 * 90, 900), (900, 10), (20, 10)],
	linear = CVnn.LinearLayers.kSplitLinear,
	init = cplx_trabelsi_standard_,
	probtype = CVnn.ProbabilityKind.kReal,
	fc_comp = CVnn.FCLayerCompression.none,
	activation = CVnn.Activations.kReLU,
)

kCoShNetRecipe = CVnn.CVnnRecipe(
	modeltype = CVnn.SupportedModels.kCoShCVNN,
	conv2d = CVnn.ConvLayers.kSplitConv,
	conv_config = [
			CVnn.CVnn_base.Conv2dDesc(20, 30, 5, 1), CVnn.CVnn_base.AvgPool2dDesc(2,2),
			CVnn.CVnn_base.Conv2dDesc(30, 50, 5, 1), CVnn.CVnn_base.AvgPool2dDesc(2,2)
	],
	linear = CVnn.LinearLayers.kSplitLinear,
	init = cplx_trabelsi_standard_,
	probtype = CVnn.ProbabilityKind.kReal,
	mlp_config = kCoShMLPRecipe,
	activation = CVnn.Activations.kReLU,
	colorspace = CVnn.Colorspaces.grayscale,
	name = "kCoShNetRecipe",
)

kGiantCoShNetRecipe = CVnn.CVnnRecipe(
	modeltype = CVnn.SupportedModels.kCoShCVNN,
	conv2d = CVnn.ConvLayers.kSplitConv,
	conv_config = [
			CVnn.CVnn_base.Conv2dDesc(60, 90, 5, 1), CVnn.CVnn_base.AvgPool2dDesc(2,2),
			CVnn.CVnn_base.Conv2dDesc(90, 75, 1, 1),
			CVnn.CVnn_base.Conv2dDesc(75, 60, 1, 1),
			CVnn.CVnn_base.Conv2dDesc(60, 90, 5, 1), CVnn.CVnn_base.AvgPool2dDesc(2,2),
	],
	linear = CVnn.LinearLayers.kSplitLinear,
	init = cplx_trabelsi_standard_,
	probtype = CVnn.ProbabilityKind.kReal,
	mlp_config = kGiantCoShMLPRecipe,
	activation = CVnn.Activations.kReLU,
	colorspace = CVnn.Colorspaces.grayscale,
	name = "kGiantCoShNetRecipe",
)

#
# TT/tiny support
#

#our TDesc for 'fc1' for tiny (49990)
kTT_fc1_desc1 = layers.TTDesc("tt", (50, 25), (20, 25), (8, ), "nab, aoix, bipy", "tt-svd-1")

#our TDesc for 'fc1' for tiny2 (1024150)
kTT_fc1_desc2 = layers.TTDesc("tt", (5, 250), (2, 250), (8, ), "nab, aoix, bipy", "tt-svd-1")

# rank-8(1024150) 89.2, 7(899130) 89.0, 6(774110) 89.0, 5(649090) 89.0, 4(524070) 88.5
kTT_fc1_desc3 = layers.TTDesc("tt", (10, 125), (4, 125), (5, ), "nab, aoix, bipy", "tt-svd-1")

# rank-8( 274630) 88.8, 7(243300) 88.6, 6(211970) 88.8, 5(180640) 89.0, 4(149310) 88.5
kTT_fc1_desc4 = layers.TTDesc("tcl", (50, 5, 5), (500, 1, 1))

ttdesc_mapper = {
	"tiny": kTT_fc1_desc1,
	"tiny2": kTT_fc1_desc2,
	"tiny3": kTT_fc1_desc3,
	"tiny4": kTT_fc1_desc4,
}

kTiny1Recipe = CVnn.makeTiny(kCoShNetRecipe, kTT_fc1_desc1)
kTiny2Recipe = CVnn.makeTiny(kCoShNetRecipe, kTT_fc1_desc2)
kTiny3Recipe = CVnn.makeTiny(kCoShNetRecipe, kTT_fc1_desc3)
kTiny4Recipe = CVnn.makeTiny(kCoShNetRecipe, kTT_fc1_desc4)

default_coshrem_config = CVnn.CoShREMConfig(
							rows = 32,
							cols = 32,
							scales_per_octave = 2,
							shear_level = 3,
							octaves = 1,
							alpha = .5,
							wavelet_eff_support = 7,
							gaussian_eff_support = 14, 
						)
giant_coshrem_config = CVnn.CoShREMConfig(
							rows = 32,
							cols = 32,
							scales_per_octave = 3,
							shear_level = 3,
							octaves = 2,
							alpha = .5,
							wavelet_eff_support = 7,
							gaussian_eff_support = 14, 
						)

#recipe mapper.
recipe_mapper = {
	"GiantCoShCVNN": (giant_coshrem_config, kGiantCoShNetRecipe,),
	"CoShCVNN": (default_coshrem_config, kCoShNetRecipe),
	"tiny":  (default_coshrem_config, kTiny1Recipe),
	"tiny2": (default_coshrem_config, kTiny2Recipe),
	"tiny3": (default_coshrem_config, kTiny3Recipe),
	"tiny4": (default_coshrem_config, kTiny4Recipe),
}

if __name__ == '__main__':
	supported = SupportedModels
	for s in supported:
		print(type(s), s.name)

	print(f"ConvLayers.names() {ConvLayers.names()}")
	print(f"ConvLayers.values() {ConvLayers.values()}")
	print(f"{cplx_trabelsi_standard_} is in {kInitFuncs} {cplx_trabelsi_standard_ in kInitFuncs}")


