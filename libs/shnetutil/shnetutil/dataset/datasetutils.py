# -*- coding: utf-8 -*-
"""

Title: Dataset loader and conditioner for COVID dataset
    
Created on Thurs July 6 17:44:29 2020

@author: Manny Ko & Ujjawal.K.Panchal

"""
from collections import Counter, namedtuple
from operator import itemgetter
import numpy as np 

from shnetutil.dataset import dataset_base
from shnetutil.math import sampling

TrainingSet=namedtuple("DataSet", "train test validate train_aug test_aug validate_aug")


def getBalancedSubset(
	dataset, 
	fraction=0.5, 
	offset=0, 
	kLogging=True, 
	useCDF=False, 
	name="balanced",
	seed=1103,
):
	""" Sample a subset of 'dataset' while maintain class balance """
	subsetsize = int(len(dataset)*fraction)
	print(f" getBalancedSubset({name=}: {dataset.name=}, {len(dataset)}, {useCDF=})")

	# for some data sets (e.g. Fashion) the input is already so well scrambled we can just take a slice.
	if not useCDF:
		dbchunk = dataset_base.CustomDatasetSlice(dataset, (offset, subsetsize))
	else:
		sorteddb = dataset_base.SortedDataset(dataset)
	
		dbchunk = sampling.getBalancedSubsetCDF(
			sorteddb,
			fraction=fraction,
			bins=0,
			rng = np.random.Generator(np.random.PCG64(seed)),	#PCG64|MT19937|PCG64DXSM
			name = name,
		)
	
	if kLogging:
		dbstats = dataset_base.DatasetStats(dbchunk)
		dbstats.dumpDatasetInfo()
	return dbchunk

#
# unit test:
#
def test_getBalancedSubsetCDF():
	# code taken from t_balsubset.py	
	from shnetutil import projconfig
	from shnetutil.pipeline import loadMNIST

	mnistdir = projconfig.getMNISTFolder()
	mnist_test   = loadMNIST.getdb(mnistdir, istrain=False, kTensor = False)

	mnist_test = dataset_base.SortedDataset(mnist_test)
	print(f"mnist_test {len(mnist_test)} from {mnistdir}")

	mnist_stats = dataset_base.DatasetStats(mnist_test)
	print(mnist_stats.labelCounts(sort=True))

	subset = sampling.getBalancedSubsetCDF(
		mnist_test,
		fraction=.10,
		bins=10,
		rng = np.random.Generator(np.random.PCG64(1103)),	#PCG64|MT19937|PCG64DXSM
	)
	print(f"{type(subset)=}")
	subset_stats = dataset_base.DatasetStats(subset)
	print(f"{subset_stats.labelCounts(sort=True)=}")
	#subset_stats.dumpDatasetInfo()


if __name__ == '__main__':
	test_getBalancedSubsetCDF()
