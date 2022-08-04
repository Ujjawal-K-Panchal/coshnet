# -*- coding: utf-8 -*-
"""

Title: Project configurations - folder and paths for now
    
Created on Thurs July 6 17:44:29 2020

@author: Manny Ko & Ujjawal.K.Panchal & Hector Andrade-Loarca

"""
from cplxmodule.nn.utils.sparsity import sparsity

def is_ard(stagename):
	return stagename in {'ard', 'masked'}

def get_sparsity(model, threshold, hard=True, verbose=True):
	if verbose:
		f_sparsity = sparsity(model, hard=hard, threshold=threshold)
	else:
		f_sparsity = float("nan")
	return f_sparsity
		