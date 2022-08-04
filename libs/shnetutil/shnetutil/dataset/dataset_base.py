# -*- coding: utf-8 -*-
"""

Title: Dataset loader and conditioner for COVID dataset
    
Created on Thurs July 6 17:44:29 2020

@author: Manny Ko & Ujjawal.K.Panchal

"""
#import abc
from collections import Counter, namedtuple
from typing import List, Union
from operator import itemgetter
import numpy as np 

import torch

ImageDesc = namedtuple("ImageDesc", "coeffs label")

class DataSet():	#this is compatible with torch.utils.data.Dataset
	def __init__(self, name='generic', colorspace = "grayscale", sorted=False):
		self.images = []
		self.labels = []
		self.name = name
		self._colorspace = colorspace
		self._isSorted = sorted		
		self.stats = None

	def start(self):
		""" get ourself prepared to be used as a dataset """	
		return

	@property
	def colorspace(self):
		return self._colorspace

	@property
	def isSorted(self):
		return self._isSorted

	def __getitem__(self, index) -> ImageDesc:
		if index >= len(self):
			return None
		return ImageDesc(self.images[index], self.labels[index])

	def __len__(self):
		return len(self.images)

	def __next__(self) -> ImageDesc:
		index = self.index
		self.index += 1
		if index >= len(self): 
			raise StopIteration 
		return self.__getitem__(index)

	def __iter__(self):
		""" """
		self.index = 0
		return self

	def __str__(self):
		return f'DataSet({self.name})'

	def __repr__(self):
		return f'DataSet({self.name})'

	def getstats(self, recompute=False) -> tuple:
		if not self.stats:
			imgs = np.asarray(getCoeffs(self))
			#arr.sum(axis=tuple(range(arr.ndim-1)))
			self.stats = np.mean(imgs, axis=(0,1,2)), np.std(imgs, axis=(0,1,2))
		return self.stats

#end of DataSet...

def createDummyDataset(name="dummy", size=10000):
	dataset = DataSet(name=name)
	#TODO: fill in some random data
	return dataset

def getLabels(dataset):
	#assert(isinstance(dataset, dataset_base.DataSet))
	labels = [item.label for item in dataset]
	return labels

def getCoeffs(dataset):
	coeffs = [item.coeffs for item in dataset]
	return coeffs

def qSameClass(labels):	
	""" predicate for all labels in 'labels' are of the same class """ 
	unique = np.unique(labels)
	return len(unique) == 1

def gatherByIndices(dataset, indices: Union[list, np.ndarray]) -> tuple:
	""" returns two parallel list of (images+, labels+) """
	#TODO: apply zip() ?
	indxs = np.asarray(indices)
	return dataset[indxs]

def getCoeffsByIndices(dataset, indices: Union[list, np.ndarray]):
	return [dataset[idx].coeffs for idx in indices]

def getLabelsByIndices(dataset, indices):
	return [dataset[idx].label for idx in indices]

def getClassIndices(dataset):
	""" return the indices for each class """
	#2.2: wonder if numexpr is helpful here
	labels =  np.asarray(getLabels(dataset))
	unique = np.unique(labels)
	result = []
	for label in unique:
		bindices = (label == labels)
		result.append(np.arange(len(labels))[bindices]) 
	return result

def verifyChunk(dbchunk, dataset):
	""" assume each entry is an ndarray """
	offset = dbchunk.offset
	result = True
	count = 0

	for i, item in enumerate(dbchunk):
		#print(i)
		img0, label0 = dataset[offset+i]
		img1, label1 = item
		result &= np.array_equal(img0, img1)
		result &= label0 == label1
		count += 1
		assert(result)
	assert(count == len(dbchunk))	
	return result

class CustomDatasetSlice(torch.utils.data.Dataset):		#torch.utils.data.Dataset
	def __init__(
		self, 
		dataset,		#conforming impl. of a torch.utils.data.Dataset 
		ourslice,		#ourslice=(<offset>, <size>)
		name="custom",
		kVerf=False
	):
		""" a 1D slice (subset) of a CustomDataset """
		self.dataset = dataset
		self.offset  = ourslice[0]		#index offset
		assert(self.offset < len(dataset))
		self.size = min((len(dataset)- ourslice[0]), ourslice[1])  #assume 'offset' is valid
		self.name = name
		if kVerf:
			verifyChunk(self, dataset)

	def __getitem__(self, index) -> ImageDesc:
		""" 1. Return a data pair (e.g. image and label). """
		return self.dataset[self.offset + index]

	def __len__(self):
		return self.size

	def __iter__(self):
		""" return a new iterator """
		self.reset()
		return self

	def __next__(self) -> ImageDesc:
		""" main iteration interface """
		if self.n < self.size:
			index = self.n
			self.n += 1
			return self.__getitem__(index)
		else:
			raise StopIteration

	def reset(self):
		""" reset the iteration counter to 0 """
		self.n = 0

class CustomDatasetSubset(torch.utils.data.Dataset):		#torch.utils.data.Dataset
	def __init__(
		self, 
		dataset,		#conforming impl. of a torch.utils.data.Dataset 
		indices,		#our index set
		name="custom",
		kVerf=False
	):
		""" a 1D slice (subset) of a CustomDataset """
		self.dataset = dataset
		self.indices = indices
		self.size = indices.shape[0]
		self.name = name

	def __getitem__(self, index) -> ImageDesc:
		""" 1. Return a data pair (e.g. image and label). """
		return self.dataset[self.indices[index]]

	def __len__(self):
		return self.size

	def __iter__(self):
		""" return a new iterator """
		self.reset()
		return self

	def __next__(self) -> ImageDesc:
		""" main iteration interface """
		if self.n < self.size:
			index = self.n
			self.n += 1
			return self.__getitem__(index)
		else:
			raise StopIteration

	def reset(self):
		""" reset the iteration counter to 0 """
		self.n = 0

def sortDataset(dataset: Union[torch.utils.data.Dataset, DataSet]):
	allpairs = (ent for ent in dataset)
	sortedpairs = sorted(allpairs, key=itemgetter(1))
	return sortedpairs

class SortedDataset(torch.utils.data.Dataset):		#
	def __init__(
		self, 
		dataset,		#conforming impl. of a torch.utils.data.Dataset 
		name="sorted",
	):
		""" A dataset sorted by its labels """
		self.dataset = dataset
		self.size = len(dataset)
		self.name = name
		self.sortedpairs = sortDataset(dataset)

	@property
	def isSorted(self):
		return True

	def __getitem__(self, index) -> ImageDesc:
		""" 1. Return a data pair (e.g. image and label). """
		return self.sortedpairs[index]

	def __len__(self):
		return self.size

	def __iter__(self):
		""" return a new iterator """
		self.reset()
		return self

	def __next__(self) -> ImageDesc:
		""" main iteration interface """
		if self.n < self.size:
			index = self.n
			self.n += 1
			return self.__getitem__(index)
		else:
			raise StopIteration

	def reset(self):
		""" reset the iteration counter to 0 """
		self.n = 0

class DatasetStats():
	""" Support various types of stats on a dataset """
	def __init__(self, dataset):
		self.dataset = dataset
		self._labels = None
		self._labelcnts = None

	@property
	def size(self):
		return self.dataset.size
		
	@property
	def labels(self):
		""" lazily generating the labels """
		if (self._labels is None):
			self._labels = getLabels(self.dataset)
		return self._labels

	@property
	def labelcnts(self) -> Counter:
		""" lazily generating the counts """
		if not self._labelcnts:
			self._labelcnts = Counter(self.labels)
		return self._labelcnts

	def labelCounts(self, sort=False):
		counts = self.labelcnts
		if sort:
			return sorted(counts.items(), key=itemgetter(0))
		else:
			return counts

	def qIsBinary(self):
		""" predicate for being a binary dataset """
		labelcnts = self.labelcnts
		return len(labelcnt.keys()) == 2

	def numClasses(self):
		labelcnts = self.labelcnts
		return len(labelcnts.keys())

	def verifyClassDist(self):
		classindices = getClassIndices(self.dataset)
		print(len(classindices))

		for idx, indices in enumerate(classindices):
			labels = getLabelsByIndices(self.dataset, indices)
			assert(qSameClass(labels))
			unique = np.unique(labels)
			print(f"class[{idx}] = {len(indices)}", unique)
			assert(unique[0] == idx)

	def dumpDatasetInfo(self, sort=True):
		dataset = self.dataset
		print(type(dataset), len(dataset), dataset[0][0].shape, f"numclasses {self.numClasses()}")
		labelcnt = self.labelCounts(sort=sort)
		print(f"dataset labelcnt {labelcnt} ")

def cumsum(db_stats:DatasetStats) -> tuple:
	labelcnts = db_stats.labelCounts(sort=True)
	labels, cnts = zip(*labelcnts)
	#print(labels, cnts)
	return labels, np.cumsum(cnts)

if False:
	def get_mean_and_std(dataset):
	    '''Compute the mean and std value of dataset.'''

	    mean = torch.zeros(3)
	    std = torch.zeros(3)
	    print('==> Computing mean and std..')
	    for inputs, targets in dataloader:
	        for i in range(3):
	            mean[i] += inputs[:,i,:,:].mean()
	            std[i] += inputs[:,i,:,:].std()
	    mean.div_(len(dataset))
	    std.div_(len(dataset))
	    return mean, std

def testGather(bigtest):
	seed = 99
	num_workers = 1
	ss = np.random.SeedSequence(seed)
	rng = np.random.Generator(np.random.PCG64(ss))	#PCG64|MT19937

	#2: numpy
	labels = np.asarray([item.label for item in bigtest])
	print(labels.shape)

	num = 20
	indices = rng.integers(low=0, high=len(bigtest), size=num)
	print(indices)

	labels1 = np.asarray([bigtest[i].label for i in indices])
	print(f"labels1 {labels1}")

	labels2 = labels[indices]
	print(f"np labels[indices]: {labels2}")

	#3: torch
	torchlabels = torch.from_numpy(labels)
	labels22 = torchlabels[indices]
	print(f"torchlabels[indices]: {labels22}")

	abatch = gatherByIndices(bigtest, indices)
	#label33 = dataset_base.getLabels(abatch)
	print(f" gatherByIndices(bigtest, indices): {abatch[0].shape} {type(abatch[0].shape)}")
	print(abatch[1])


if __name__ == "__main__":
	...
