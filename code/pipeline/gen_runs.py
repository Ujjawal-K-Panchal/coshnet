"""
Title: test_fashion.

Created on Sun Aug 16 17:44:29 2020

@author: Manny Ko & Ujjawal.K.Panchal.
"""

#
# All the test runs cited in the paper and testing the model.
#
from typing import List, Tuple, Optional, Callable
import numpy as np

from shnetutil.pipeline import trainutils


def generateRuns(generator:Callable, train_params, epochs:int=2):
	training_runs = []
	parmlist = []

	for overrides, parm in generator:
		parmlist.append(parm)
		training_runs.append(trainutils.OneRun(train_params, overrides=overrides, runname="", indep=True))
	return training_runs, parmlist	

def wrapTuple(atuple:tuple, key:str, tag, epochs):
	""" wrap a user supplied tuple of parameter values for ['key'] """
	for ent in atuple:
		overrides = {
			tag:	{'epochs': epochs, key: ent},
		}
		yield overrides, ent

def gen1rand(number_of_runs:int, epochs:int=4, tag='seeds'):
	""" Generator for 1 random seed run """
	picard_magic = 3407	

	ss = np.random.SeedSequence(12345)
	rng = np.random.Generator(np.random.PCG64(ss))	#PCG64|MT19937
	seeds = rng.integers(low=0, high=99999999, size=number_of_runs+1)
	seeds[-1] = picard_magic

	for seed in seeds:
		overrides = {
			tag:	{'epochs': epochs, 'seed': seed},
		}
		yield overrides, seed

def generateRandomSeeds(number_of_runs:int, train_params:trainutils.TrainingParams, epochs:int=4):
	gen = gen1rand(number_of_runs, epochs, tag="seeds")

	training_runs, seeds = generateRuns(gen, train_params, epochs)

	return training_runs, seeds

def gen1bsize(number_of_runs:int, bsize=32, step=64, epochs:int=2, tag="bsize"):
	""" Generator for 1 batchsize run """
	bs = bsize

	for i in range(number_of_runs):
		overrides = {
			tag:	{'epochs': epochs, 'batchsize': bs},
		}
		yield overrides, bs
		bs += step

def batchsizeRuns(number_of_runs:int, train_params:trainutils.TrainingParams, bsize=32, step=64, epochs:int=2):
	if type(bsize) == tuple:
		gen = wrapTuple(bsize, "batchsize", tag="bsize", epochs=epochs)
	else:	#create our generator for range(bsize,,step)
		gen = gen1bsize(number_of_runs, bsize, step, epochs)

	training_runs, bsizes = generateRuns(gen, train_params, epochs)

	return training_runs, bsizes	

def gen1lr(number_of_runs:int, lr_0=.001, step=.001, epochs:int=2, tag="lr"):
	""" Generator for 1 lr run """
	lr = lr_0

	for i in range(number_of_runs):
		overrides = {
			tag:	{'epochs': epochs, 'lr': lr},
		}
		yield overrides, lr
		lr += step

def lr_rateRuns(number_of_runs:int, train_params, lr_0=.001, step=.001, epochs:int=2):
	gen = gen1lr(number_of_runs, lr_0, step, epochs, tag="lr")

	training_runs, lrs = generateRuns(gen, train_params, epochs)

	return training_runs, lrs	
