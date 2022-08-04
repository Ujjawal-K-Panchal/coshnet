"""
Title: fashionRecipe - complex valued NN based on COShREM.
	
Created on Tue Dec 7 10:44:29 2021

@author:  Manny Ko.
"""
from collections import Counter
from operator import itemgetter
from typing import List, Union
import numpy as np
import scipy.interpolate as interpolate

from shnetutil.dataset import dataset_base

kTestRand=False
#
# https://tmramalho.github.io/blog/2013/12/16/how-to-do-inverse-transformation-sampling-in-scipy-and-numpy/
# http://usmanwardag.github.io/python/astronomy/2016/07/10/inverse-transform-sampling-with-python.html
#
def getHistogram(dataset:dataset_base.DataSet, bins=10, tag='', kStats=True) -> tuple:
	""" generate a histogram for the labels in 'dataset' """
	db_stats = dataset_base.DatasetStats(dataset)
	n_classes = db_stats.numClasses()

	if kStats:
		cnts = db_stats.labelcnts
		print(f"{tag} labels {db_stats.labelCounts(sort=True)}")

	labels = db_stats.labels	
	bins = bins if bins else n_classes 	#use same number bins as classes
	# https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
	hist, bin_edges = np.histogram(labels, bins, range=(0, n_classes), density=True)

	return hist, bin_edges

# https://tmramalho.github.io/blog/2013/12/16/how-to-do-inverse-transformation-sampling-in-scipy-and-numpy/
def inverse_transform_sampling(
	dataset:dataset_base.DataSet, 
	n_samples=100,
	n_bins=20,
	rng:np.random.Generator=None, 
):
	""" inverse-CDF xform sampling """
	assert(dataset.isSorted)
	hist, bin_edges = getHistogram(dataset, bins=n_bins, kStats=False)
	#print(f"{bin_edges.shape=}")
	cum_values = np.zeros(bin_edges.shape)
	cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))		#[0.0..1.0]
	#print(f"{bin_edges=}, {cum_values=}")

	inv_cdf = interpolate.interp1d(cum_values, bin_edges, kind='quadratic', assume_sorted=True)

	rng = rng if rng else np.random.Generator(np.random.PCG64(1103))	#PCG64|MT19937|PCG64DXSM
	rand_samples = rng.uniform(0, 1, n_samples)
	samples = inv_cdf(rand_samples)
	return samples	

if False:
	def inverse_transform_sampling(
		dataset:dataset_base.DataSet, 
		n_samples:int=100,		#size of the returned subset
		n_bins:int=10, 			#number of histogram bins to approximate the label distribution
		rng:np.random.Generator=None, 
	) -> list:
		hist, bin_edges = getHistogram(dataset, bins=n_bins)

		cum_values = np.zeros(bin_edges.shape)
		cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
		inv_cdf = interpolate.interp1d(cum_values, bin_edges)

		rng = rng if rng else np.random.Generator(np.random.PCG64(1103))	#PCG64|MT19937|PCG64DXSM
		rand_samples = rng.uniform(0, 1, n_samples)
		#indices = inv_cdf(rand_samples)
		#indices = np.round(indices)

		#http://usmanwardag.github.io/python/astronomy/2016/07/10/inverse-transform-sampling-with-python.html
		# 
		indices = [int(np.argwhere(cum_values == min(cum_values[(cum_values - r) > 0]))) for r in rand_samples]
		
		return indices
#
# Efficient class + within-class sampling. Inspired by many-light sampling in raytracing
# "Monte Carlo techniques for direct lighting calculations" ACM Trans. Graph. 1996
#
def manyLightsSlow(dataset, samples, tag=""):
	cnt = Counter()

	db_stats = dataset_base.DatasetStats(mnist_test)
	labelcnts = db_stats.labelCounts(sort=True)
	labels, cnts = zip(*labelcnts)

	cumsum = np.cumsum(cnts).tolist()
	cumsum.insert(0, 0)
	#print(labels, cumsum, cnts)

	subset = []
	for val in samples:
		b = np.trunc(val).astype(int)
	 	#the fractional part will be used to sample each bin 
		frac = val - b 			#this is the secret sauce from the many lights paper
		offset = cumsum[b]
		perclass_i = np.trunc(frac * cnts[b]).astype(int)
		subset.append(dataset[offset + perclass_i]) 
		cnt.update([b])
	print(f"{tag}: {sorted(cnt.items(), key=itemgetter(0))}", end="")
	print(f"  min, max {min(cnt.values())}, {max(cnt.values())}")

	return subset	

#
# Efficient class + within-class sampling. Inspired by many-light sampling in raytracing
# "Monte Carlo techniques for direct lighting calculations" ACM Trans. Graph. 1996
#
def manyLightsFast(dataset, samples, tag="") -> List[int]:
	assert(dataset.isSorted)
	db_stats = dataset_base.DatasetStats(dataset)
	labelcnts = db_stats.labelCounts(sort=True)
	labels, cnts = zip(*labelcnts)

	cumsum = np.cumsum(cnts).tolist()
	cumsum.insert(0, 0)
	cumsum = np.asarray(cumsum)

	subset = np.ndarray((len(samples),), dtype=int) 	#collect our indices
	indices = np.trunc(samples).astype(int)
	fractions = samples - indices
	offsets = cumsum[indices]	

	for i, bn in enumerate(indices):
		offset = offsets[i]
	 	#the fractional part will be used to sample each bin - secret sauce from 'many lights'
		frac = fractions[i]
		perclass_i = np.trunc(frac * cnts[bn]).astype(int) 	#this is the many-lights sampling
		subset[i] = offset + perclass_i

	return subset	

def inverseCDF_byLabels(
	dataset:dataset_base.DataSet, 
	n_samples:int, 
	bins=0,		#defaults to same # of bins as classes
	rng:np.random.Generator=np.random.Generator(np.random.PCG64(1103)),
):
	""" Sample a subset and maintain the same class distribution efficiently """
	samples = inverse_transform_sampling(
		dataset, 
		n_samples=n_samples, 
		n_bins=bins,
		rng = rng,		#PCG64|MT19937|PCG64DXSM
	)
	#
	# Efficient class + within-class sampling. Inspired by many-light sampling in raytracing
	# "Monte Carlo techniques for direct lighting calculations" ACM Trans. Graph. 1996
	#
	#subset1 = manyLightsSlow(dataset, samples, tag=f"bins[{bins}]")
	subset = manyLightsFast(dataset, samples, tag=f"bins[{bins}]")
	return subset

def getBalancedSubsetCDF(
	dataset:dataset_base.DataSet, 
	fraction=.1, 
	bins:int=0,		#defaults to same # of bins as classes
	rng:np.random.Generator=np.random.Generator(np.random.PCG64(1103)),
	name="balancedCDF",
	withreplace=False,		#sample without replacement
	shuffle=True,	
	tag="" 
):
	""" Sample a subset and maintain the same class distribution efficiently using inverse-CDF sample.
		Note: this by itself does not make highly imbalanced dataset balanced.  
	"""
	n_samples = int(len(dataset)*fraction), 
	subset = inverseCDF_byLabels(dataset, n_samples, bins, rng)

	if shuffle:
		#rng = np.random.Generator(np.random.MT19937(3409))	#PCG64|MT19937
		#rng.shuffle(subset)
		pass

	if not withreplace:	
		uniques = np.unique(subset)		#remove duplicates
		#print(f"{len(subset)=}, {uniques.shape=}")
		n_uniques = uniques.shape[0]
		remain = len(subset) - n_uniques
		subset2 = inverseCDF_byLabels(dataset, remain, bins, rng)

		subset[0:n_uniques] = uniques	#subset = np.concatenate(uniques, np.asarray(subset2))
		subset[n_uniques:]  = subset2
		#print(f"{np.unique(subset).shape}")

	dbchunk = dataset_base.CustomDatasetSubset(dataset, np.asarray(subset), name=name)

	return dbchunk

#
# testing code
#
def test_bins(
	dataset, 
	bin_range=(10, 100, 10), 
	rng:np.random.Generator=np.random.Generator(np.random.PCG64(1103)), 
	tag="PCG64",
):
	for bins in range(*bin_range):
		rng = np.random.Generator(np.random.PCG64(1103)) 	#PCG64|MT19937|PCG64DXSM
		balanced = inverse_transform_sampling(
			dataset, 
			n_samples=int(len(dataset)*0.1), 
			n_bins=bins,
			rng = rng,		#PCG64|MT19937|PCG64DXSM
		)
		cnt = Counter()
		for v in balanced:
			val = v
			b = np.trunc(val).astype(int)
			frac = val - b 	#the fractional part will be used to sample each bin
			cnt.update([b])
		print(f"{tag} bins[{bins}]: {sorted(cnt.items(), key=itemgetter(0))}", end="")
		print(f"  min, max {min(cnt.values())}, {max(cnt.values())}")
		#print(balanced)

def test_getBalancedSubsetCDF():
	# code taken from t_balsubset.py	
	from shnetutil import projconfig
	from shnetutil.pipeline import loadMNIST

	mnistdir = projconfig.getMNISTFolder()
	fashiondir = projconfig.getFashionMNISTFolder()
	mnist_test   = loadMNIST.getdb(mnistdir, istrain=False, kTensor = False)
	fashion_test = loadMNIST.getdb(fashiondir, istrain=False, kTensor = False)

	mnist_test = dataset_base.SortedDataset(mnist_test)
	print(f"mnist_test {len(mnist_test)} from {mnistdir}")

	hist, bin_edges = getHistogram(mnist_test, bins=10, tag='mnist')
	print(f"{type(mnist_test)}: hist = {hist}")

	mnist_stats = dataset_base.DatasetStats(mnist_test)
	print(mnist_stats.labelCounts(sort=True))

	if kTestRand:
		test_bins(
			fashion_test, 
			bin_range=(0, 50, 10), 
			rng = np.random.Generator(np.random.PCG64(1103)),	#PCG64|MT19937|PCG64DXSM
			tag="PCG64"
		)

	subset = getBalancedSubsetCDF(
		mnist_test,
		fraction=.10,
		bins=0,
		rng = np.random.Generator(np.random.PCG64(1103)),	#PCG64|MT19937|PCG64DXSM
		withreplace=False,		#sample with replacement
	)

	subset_stats = dataset_base.DatasetStats(subset)
	print(f"{subset_stats.labelCounts(sort=True)=}")
	subset_stats.dumpDatasetInfo()


if __name__ == '__main__':
	test_getBalancedSubsetCDF()
