# -*- coding: utf-8 -*-
"""

Title: Shearlet based CNN vs. simple MLP in Fashion MNIST.
	
Created on Mon June 30 17:44:29 2020

@author: Manny Ko & Ujjawal.K.Panchal 

"""
#import logging
from cplxmodule import cplx #new import.
import os, sys, time
import abc
import numpy as np
from collections import namedtuple, Counter
import asyncio

#our modules
from shnetutil import projconfig
from shnetutil.dataset import dataset_base
from shnetutil.utils import torchutils

from pyutils.testutil import time_spent


kVerifyResults=False

# See Docs/DataPipeline/mk_datapipelin.pdf
# Key Influences: 
#	- Batch Augmentation: https://arxiv.org/abs/1705.08741
#	- Minibatch persistency: https://arxiv.org/abs/1806.07353
#	- Data Echoing: https://arxiv.org/abs/1907.05550
#	- Bagging: 

# https://pymotw.com/3/asyncio/executors.html

class BatchBuilderBase(metaclass=abc.ABCMeta):
	""" An batch generator with support for asyncio and MP xform/augmentation,
		data echoing etc.
	"""
	def __init__(
		self,
		dataset,			#utis.data.Dataset
		batchsize=16,
		buffersize=128,		#size of our shuffle buffer
		num_workers = 4,	#TODO: not implemented yet
		seed=1234, 
		shuffle = True,		#we almost always want to shuffle
	):
		self.dataset = dataset
		self.batchsize = batchsize
		self.buffersize = buffersize
		self.num_workers = num_workers
		self.shuffle = shuffle
		self.size = len(dataset)
		self.indices = np.arange(self.size)

	def __len__(self):
		return self.size

	def reset(self):
		pass

	def xformbatch(self, batch):
		return batch

	def finalize(self):
		pass	

	@abc.abstractmethod
	def epoch(self, kLogging=False):
		""" our generator which will emit batches 1 at a time for an epoch """
		pass

	@abc.abstractmethod
	def rand_indices(self, num):	
		pass

	@property
	def shuffle_buf(self):
		""" Shuffle buffer - see https://arxiv.org/abs/1907.05550 """
		return self.indices
#end if BatchBuilderBase

def getWeights(dataset):
	""" returns a list of weights suitable for WeightedRandomSampler or torch.multinomial.
	See https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/torchsampler/imbalanced.py 
	"""
	labels = getattr(dataset, 'labels', [item.label for item in dataset])
	labelcnt = Counter(labels)
	weights = [1.0 / labelcnt[item.label] for item in dataset]
	#print(len(weights), np.unique(weights))

	return weights

class BatchBuilder(BatchBuilderBase):
	""" An batch generator with support for asyncio and MP xform/augmentation """
	def __init__(
		self,
		dataset,			#utis.data.Dataset
		batchsize=16,
		buffersize=128,		#size of our shuffle buffer
		num_workers = 4,	#TODO: not implemented yet
		seed=1234, 
		shuffle = True,		#we almost always want to shuffle
	):
		super().__init__(dataset, batchsize, buffersize, num_workers, seed, shuffle)
		#TODO: use 'buffersize' - currently we are using a shuffle buffer the size of the input

		#https://numpy.org/doc/stable/reference/random/index.html
		ss = np.random.SeedSequence(seed)
		child_seeds = ss.spawn(num_workers)		#TODO: for MP code
		self.rng = np.random.Generator(np.random.PCG64(ss))	#PCG64|MT19937
		self.cur_batch = None 

	def xformbatch(self, batch):
		return batch

	def rand_indices(self, num):	
		indices = self.rng.integers(low=0, high=self.size, size=num)
		return indices

	def doshuffle(self):
		if self.shuffle:
			self.rng.shuffle(self.indices)

	def epoch(self, kLogging=False):
		""" our generator which will emit batches 1 at a time for an epoch """
		self.doshuffle()
		
		if kLogging:	
			print(f"indices: {batchbuilder.indices} {batchbuilder.indices.dtype}")

		numentries = len(self.dataset)
		batchsize = self.batchsize
		for start in range(0, numentries, batchsize):
			end = min(start + batchsize, numentries)
			batch = self.indices[start:end]
			self.cur_batch = self.xformbatch(batch)
			yield self.cur_batch 	#GENERATOR
#end.. class BatchBuilder			

class BatchIterator():
	def __init__(
		self,
		batchbuilder
	):
		assert(isinstance(batchbuilder, BatchBuilderBase))
		self.builder = batchbuilder
		self.batch_num = None

	def __iter__(self):
		""" iterator support - i.e. iter(<batchbuilder>) """
		self.batch_num = 0
		return self

	def __next__(self):
		""" Torch style batch iterator 
			Result: (images, labels)
		"""
		builder = self.builder
		numentries = builder.size
		batchsize = builder.batchsize

		batch_num = self.batch_num
		offset = batch_num * batchsize
		self.batch_num += 1
		if offset >= numentries: 
			raise StopIteration 
		end = min(offset + batchsize, numentries)
		# Bagging here - sample with replacement
		indices = builder.indices[offset:end]
		indices = builder.rand_indices(len(indices))
		builder.cur_batch = indices 		#for Minibatch persistency and Data Echoing
		images, labels = getBatchAsync(builder.dataset, indices)
		return images, labels


class Bagging(BatchBuilder):
	""" Use Bagging sampling to generate batches """
	#TODO: Try without-replacement sampling https://papers.nips.cc/paper/6245-without-replacement-sampling-for-stochastic-gradient-methods
	def __init__(
		self,
		dataset,			#utils.data.Dataset
		batchsize=16,
		buffersize=128,		#size of our shuffle buffer
		num_workers = 4,	#TODO: not implemented yet
		seed=1234, 
		shuffle = False,
	):
		super().__init__(dataset, batchsize, buffersize, num_workers, seed, shuffle)
		self.cur_batch = -1

	#iterator support - i.e. iter(<Bagging>)	
	def __iter__(self):
		""" """
		self.batch_num = 0
		return self		#TODO: implement a standalone iterator object (to support multiple iter)

	def __next__(self):
		""" Torch style batch iterator 
			Result: (images, labels)
		"""
		numentries = self.size
		batchsize = self.batchsize
		batch_num = self.batch_num
		offset = batch_num * batchsize
		self.batch_num += 1
		if offset >= numentries: 
			raise StopIteration 
		end = min(offset + batchsize, numentries)
		# Bagging here - sample with replacement
		indices = self.indices[offset:end]
		indices = self.rand_indices(len(indices))
		self.cur_batch = indices 		#for Minibatch persistency and Data Echoing
		images, labels = getBatchAsync(self.dataset, indices)
		return images, labels

	def epoch(self, kLogging=False):
		""" Generator for sample-with-replacement -> Bagging 
			Return: the sample indices for maximum client side flexibility. Typically it will be used
					to call getBatchAsync() to do the heavy lifting.
		"""
		numentries = self.size
		batchsize = self.batchsize

		for start in range(0, numentries, batchsize):
#			print(f"epoch[{start}]", end='')
			end = min(start + batchsize, numentries)
			# Bagging here - sample with replacement
			indices = self.indices[start:end]
			#sample-with-replacement -> Bagging
			indices = self.rng.integers(numentries, size=len(indices))
			self.cur_batch = indices 		#for Minibatch persistency and Data Echoing

			yield indices 	#GENERATOR
#end.. class Bagging			

def verify2result(myresults1, myresults2):
	r1 = sorted(myresults1, key=lambda e: e.index)	
	r2 = sorted(myresults2, key=lambda e: e.index)	

	verified = True
	for i in range(len(r1)):
		item01 = r1[i]
		item02 = r2[i]
		#1: compare our indices
		if item01.index != item02.index:
			verified = False
		data1 = item01.data
		data2 = item02.data
		#2: compare our ndarrays(images)
		if np.array_equal(data1.coeffs, data2.coeffs) == False:
			print(f"[{i}]: {data1.coeffs}, {data2.coeffs}")
			verified = False
		#3: compare our labels
		if data1.label != data2.label:
			verified = False

	assert(verified)
	return verified

#
# asyncio: https://docs.python.org/3/library/asyncio-task.html#task-object
#
GetDesc = namedtuple("GetDesc", "index data")

async def get1(dataset, index):
	return GetDesc(index, dataset[index])

async def getBatch(dataset, indices, logging=False):
	batchsize = len(indices)
	imglist = []	#for collecting the results from async complete callback
	labellist = []
	
	def oneDone(task):
		#https://stackoverflow.com/questions/44345139/python-asyncio-add-done-callback-with-async-def
		myresult = task.result()
		imglist.append(myresult.data.coeffs)
		labellist.append(np.int64(myresult.data.label))	# Torch seems to want labels as torch.long which is int64

	#1: sort indices to get a sequential access order - optimize IO
	batch = np.sort(indices)

	if (logging):
		print(f"batch = {batch}")

	tasks = []
	#1: issue concurrent load for the whole batch
	for i in batch:
		t = asyncio.create_task(get1(dataset, i))
		t.add_done_callback(oneDone)
		tasks.append(t)

	#2: parallel wait on all concurrent tasks	
 	#https://stackoverflow.com/questions/42231161/asyncio-gather-vs-asyncio-wait
	done, pending = await asyncio.wait(tasks)	#, return_when=asyncio.FIRST_COMPLETED
	assert(len(pending) == 0)

	#optionally retrieve each result direct from the tasks to verify add_done_callback()
	if kVerifyResults:
		myresults2 = [task.result() for task in done]
		verify2result(myresults, myresults2)

	#TODO: change get1() to directly output to a ndarray
	return np.asarray(imglist), np.asarray(labellist)

def getBatchAsync(dbchunk, batch, logging=False):
	""" get 'batch' which is an array of indices 
		Ret: ndarray((batchsize, width, height), dtype=float32)
	"""
	results = asyncio.run(getBatch(dbchunk, batch, logging))
	return results

#
# Unit test routines:
#
from shnetutil.pipeline import loadMNIST

def test_epochgen(mnist_train, bsize):
	""" use .epoch() generator on the BatchBuilder """
	trainbatchbuilder = Bagging(mnist_train, bsize)
	labels1 = []
	for i in range(epochs):
		labelcnt = Counter()
		trainiter = trainbatchbuilder.epoch(False)
		#trainiter = iter(train_loader)
		for b, mybatch in enumerate(trainiter):
			#'mybatch' is an array of indices defining the minibatch samples
			#print(mybatch[10:])
			images, labels = getBatchAsync(mnist_train, mybatch)
			#images, label = batch_
			print(f"[{i,b}]{mybatch.shape}, {images.shape}")
			labelcnt.update(labels)
			labels1.append(labels)
		print(labelcnt)	
	return labels1
		
def test_selfIter(mnist_train, bsize):
	""" use iter() on the BatchBuilder itself """
	trainbatchbuilder = Bagging(mnist_train, bsize)
	labels2 = []
	for i in range(epochs):
		trainiter = iter(trainbatchbuilder)
		labelcnt = Counter()

		for b, mybatch in enumerate(trainiter):
			images, labels = mybatch
			print(f"[{i,b}]{type(mybatch)}, {images.shape}")
			labelcnt.update(labels)
			labels2.append(labels)
		print(labelcnt)
	return labels2	

def test_iterObj(mnist_train, bsize):
	""" standalone iterator .BatchIterator """
	trainbatchbuilder = Bagging(mnist_train, bsize)
	train_loader = BatchIterator(trainbatchbuilder)

	labels1 = []
	for i in range(epochs):
		labelcnt = Counter()

		for b, mybatch in enumerate(train_loader):
			images, labels = mybatch
			print(f"[{i,b}]{type(mybatch)}, {images.shape}")
			labelcnt.update(labels)
			labels1.append(labels)
		print(labelcnt)	
	return labels1
#
# BatchCache:
#
class BatchCache(BatchBuilderBase):
	def __init__(
		self,
		batchsize:int,
		batchbuilder: BatchBuilderBase,
		xform = None,
	):	
		assert((batchbuilder == None) or (not batchbuilder.shuffle))
		self.batchsize = batchsize
		self.batchbuilder = batchbuilder
		#self.dataset = batchbuilder.dataset
		self.xform = xform
		self.capture = True
		self.reset()

	def __len__(self):
		return len(self.cache)
		#return len(self.batchbuilder)

	def __getitem__(self, index:int):
		""" index cached batches """
		return self.cache[index]

	def __deepcopy__(self, memo):
		#print(f"BatchCache.__deepcopy__")
		return self

	def getitem(self, index:int):
		""" Treating all batches as a 1-d array of self.size """
		b, offset = divmod(index, self.batchsize)
		return self.cache[b][offset]		

	def reset(self):
		self.cache = []
		self.size = 0

	def setcapture(self, enable:bool):
		self.capture = enable	

	def finalize(self):
		""" Finished first epoch, caching is done """
		self.setcapture(False) 	#next epoch will be replay of capture
		print(f"BatchCache.finalize={self.size}")
		#torch.save(self.cache, "batchcacheX")	#1.52gb for 10k test

	def rand_indices(self, num):	
		pass

	def insert(self, batchdata: tuple):
		shape = batchdata.shape
		self.size += shape[0] 		#size of batch
		#print(f"insert {shape}")
		self.cache.append(batchdata.detach().cpu())

	def loadcache(self, kLogging=False):
		tic1 = time.time()
		batchbuilder = self.batchbuilder
		epoch = batchbuilder.epoch(False)

		for b, mybatch in enumerate(epoch):
			imglist, labels = getBatchAsync(batchbuilder.dataset, mybatch)
			imglist = self.xform(np.asarray(imglist))
			self.insert(imglist)
		if kLogging:
			time_spent(tic1, f"loadcache", count=1)	#5.61/e,
		
	def epoch(self, kLogging=False):
		if self.capture:
			self.loadcache(kLogging=kLogging)
		for batch in self.cache:
			yield batch
#end of BatchCache			

class CachedDataset(dataset_base.DataSet):
	""" An adaptor for BatchCache to make it into a DataSet """
	def __init__(self, batchcache: BatchCache, colorspace = "grayscale"):
		assert(issubclass(type(batchcache), BatchCache))
		super().__init__("cachedDataset", colorspace)
		self.batchcache = batchcache
		self.batchsize = batchcache.batchsize
		self.cached = batchcache.cache

		#1: verify the total count of all cached batches
		count = 0
		for batch in self.cached:
			shape = batch.shape
			count += shape[0]
		assert(count == batchcache.size)	
		self.size = count
		self.getter = self.doubleIndexGetter
		self.finalized = False

	def finalize(self):
		self.finalized = True
		self.cached = cplx.cat(self.cached, dim = 0)
		self.getter = self.singleIndexGetter
		return

	def __getitem__(self, index:int):
		return self.getter(index)
		
	def doubleIndexGetter(self, index: int):
		b, offset = divmod(index, self.batchsize)
		return self.cached[b][offset]

	def singleIndexGetter(self, index: int):
		return self.cached[index]

	def __len__(self):
		return self.size
#end of CachedDataset

def verifyBatchCache(batchbuilder, xform):
	dbchunk = batchbuilder.dataset
	batchcache = BatchCache(batchbuilder, xform)
	batchcache.loadcache(True)

	epoch = batchbuilder.epoch()

	tic2 = time.time()
	for b, mybatch in enumerate(epoch):
		imglist, labels = getBatchAsync(dbchunk, mybatch)
		imglist = xform(np.asarray(imglist))
		#print(f"b[{b}] {len(imglist)}, {type(imglist)}, {type(batchcache.cache[b][0])}")
		#print(f"b[{b}] {len(labels)}, {type(labels)}, {type(batchcache.cache[b][1])}")
		if b == 0:
			#print(labels, batchcache.cache[b][1])
			img00 = imglist[0,:,:]
			img10 = batchcache.cache[b][0][0,:,:]
			print(f"imglist[0,:,:].shape {img00.shape}, batchcache.cache[b][0][0,:,:].shape {img10.shape}")
			#print(img00[:1].real, img10[:1].real)
		#assert(np.array_equal(batchcache.cache[b][1], labels))
		assert(np.array_equal(batchcache.cache[b].numpy(), imglist.numpy()))
	time_spent(tic2, f"verify xform", count=1)	#5.61/e


if __name__ == '__main__':
	fashiondir = projconfig.getFashionMNISTFolder()

	#dataset = MNIST('mnist', train=True, download=True, transform=MNIST_TRANSFORM)
	mnist_train = loadMNIST.getdb(fashiondir, istrain=False, kTensor = False)
	print(f"mnist_train {len(mnist_train)} from {fashiondir}")

	bsize = 1000
	epochs = 2

	#1: use .epoch() generator interface
	labels1 = test_epochgen(mnist_train, bsize)

	#2: use iter() on the BatchBuilder itself
	labels2 = test_selfIter(mnist_train, bsize)

	#2: use standalone iterator
	labels3 = test_iterObj(mnist_train, bsize)

	for i in range(len(labels1)):
		l1 = labels1[i]
		l2 = labels2[i]
		l3 = labels3[i]
		assert(np.equal(l1, l2).all())
		assert(np.equal(l1, l3).all())
	print(f"passed assert(np.equal(l1, l2).all())")	
	print(f"passed assert(np.equal(l1, l3).all())")	
		