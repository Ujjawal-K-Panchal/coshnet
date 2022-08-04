# -*- coding: utf-8 -*-
"""
Title: Training pipeline utils - 
	
Created on Wed Sept 1 16:01:29 2020

@author: Manny Ko & Ujjawal.K.Panchal
"""
import abc, copy, time
from abc import abstractmethod
from collections import namedtuple
from typing import Callable, List, Tuple, Union, Optional
from numpy import around

import torch
import torch.nn as nn

from pyutils.classutils import method_exists
from pyutils.testutil import time_spent

from shnetutil.dataset import dataset_base, datasetutils
from shnetutil.modelling import modelfactory
from shnetutil.pipeline import augmentation, batch, dbaugmentations
from shnetutil.utils import torchutils, trace


# tuple to config the Model for each training stage - used in ModelPipeline()
Stage= namedtuple("Stage", "name base conv2d linear init probtype modeltype fc_comp")

def hasmethod(myclass: object, methodname: str):
	""" predicate for 'myclass' has a method with 'methodname' """
	mymethod = getattr(myclass, methodname, None)
	return (mymethod is not None) and callable(mymethod)

def getModelConfigStr(model):
	ourstr = ''
	if hasmethod(model, "getModelConfigStr"):
		ourstr = model.getModelConfigStr()
	else:
		ourstr = type(model)
	return ourstr


DataPipeline = namedtuple("DataPipeline", "dataset pipeline")

class TrainingParams():
	""" A class wrapper for a set of key-val training parameters.
		Its main purpose is to enable 'isinstance(TrainingParams)'
	Note: TrainingParams() must be able to be deepcopy(). Any object that takes a lot of memory should not be
		  placed in it.
	"""
	def __init__(self, 
		modelfactory: Union[modelfactory.ModelFactory, modelfactory.ModelPipeline],
		recipe: modelfactory.Recipe_base,
		train: 	DataPipeline,
		test: 	DataPipeline,
		validate: DataPipeline,
		batchsize: int = 128,
		validate_batchsize: int = 512,	#bsize for validate and test runs
		lr: float = 0.001,
		trace_test: bool = True, 	#enable tracing during model testing (should default to False)
		lr_schedule: bool = False,
		denoise: bool = False,
		ablation: Optional[str] = None,
		snapshot=None,
		datasetname: Optional[str] = "fashion",
		trset: Optional[str] = "test",	#for older snapshots that does not have it saved
		seed: Optional[int] = 1,
		**kwargs
	):
		self._params = {
			'modelfactory': modelfactory,
			'recipe':	recipe,
			'train': 	train,
			'test': 	test,
			'validate': validate,
			'batchsize': 	batchsize,
			'validate_batchsize': validate_batchsize,
			'lr': 			lr,
			'trace_test':	trace_test,
			'lr_schedule':	lr_schedule,
			'denoise':	denoise,
			'ablation': ablation,
			'snapshot': snapshot,
			'datasetname': datasetname,
			'trset': trset,
			'seed': seed,
		}
		self.check_params()
		self.amend(**kwargs)
		self.setup()

	def check_params(self):
		""" check to see if user passed a good set of parameters """
		train_xform = self.params['train'].pipeline
		test_xform  = self.params['test'].pipeline
		assert(issubclass(type(train_xform), augmentation.Base))
		assert(issubclass(type(test_xform), augmentation.Base))
		#assert(isinstance(self.params['recipe'], modelfactory.Recipe_base))
		return True	

	def __repr__(self):
		return f"TrainingParams({self._params})"

	def __getitem__(self, key: str):
		return self._params.get(key, None)
		
	@property
	def params(self):
		return self._params

	@property
	def progressbar(self):
		return self._params.get('progressbar', None)

	def setup(self):
		""" Callable for derived class to setup our training params using code """
		pass	

	def dup(self):	
		return TrainingParams(**copy.deepcopy(self._params))

	def swap_traintest(self):
		""" swap the values for the two keys 'test', 'train' """
		train_params = self.dup()
		params = train_params.params
		params['test'], params['train'] = params['train'], params['test']
		return train_params

	def amend(self, **kwargs):
		#print(f"amend: {kwargs}")
		self._params.update(**kwargs)
		return self

class OneRun():
	""" Tuple to define (train_params, phases) """
	def __init__(self,
		train_params: TrainingParams, 
		overrides: dict=None,
		runname='',
		indep=True,		#whether each run is independent
	):
		assert(isinstance(train_params, TrainingParams))
		#print(f"OneRun {train_params=}")
		self.name = runname
		self.indep = indep
		self.train_params = train_params if not indep else copy.deepcopy(train_params)
		#self.train_params = train_params
		self.overrides = overrides
		if overrides:
			self.train_params.amend(**overrides)

	def start(
		self,
		model: torch.nn.Module,
		tracectx: trace.TraceContext,
		loi: list = [],
		seed=1,
	):
		""" Start the training loop """
		trace.disable_console(tracectx)
		if isinstance(model, trace.TraceMixin):
			model.register_all_hooks(loi, model.log_hook)	#TODO: remove model.log_hook
		torchutils.initSeeds(seed)

	def doOverrides(self):
		""" Apply the overrides for this run to the global TrainingParams """
		params = self.train_params.params
		#print(f" doOverrides({run.name}: {run.overrides}")
		for k, v in self.overrides.items():
			params.update(v)

	def __repr__(self):
		return f"(train_params={self.train_params}"

	def __iter__(self):
		""" support unpacking """
		return iter((self.train_params, self.overrides))
#end OneRun

def loadAugCache(
	ourTransform: augmentation.Base, 
	dataset, 
	batchsize:int = 512, 
	kRemove:bool = True, 
	kLogging:bool = True
) -> dbaugmentations.CaptureAugmentation:
	#1: insert BatchCache as final stage in the augmentation pipeline
	#ourTransform  = ourAugmentations(dataset.kMean, dataset.kStd)
	capturecache = ourTransform[-1]
	if isinstance(capturecache, dbaugmentations.CaptureAugmentation):
		capturecache.reset()
		capturecache.setcapture(True)

		tic1 = time.time()
		#2.0: create a standard BatchBuilder
		batchbuilder = batch.BatchBuilder(dataset=dataset, batchsize=batchsize, shuffle=False)
		epoch = batchbuilder.epoch()
		#2.1: iterate all the batches and apply our xform to enable them to be captured in 'capturecache'
		for b, mybatch in enumerate(epoch):
			imglist, labels = batch.getBatchAsync(dataset, mybatch)
			imglist = ourTransform(imglist)
		capturecache.finalize()

		augcache = capturecache.cache
		assert(issubclass(type(augcache), batch.BatchCache))
		assert(type(capturecache) == dbaugmentations.CaptureAugmentation)

		if kLogging:
			time_spent(tic1, f"loadAugCache", count=1)
		if kRemove:
			dbaugmentations.removeCaching(ourTransform)
	else:
		capturecache = ourTransform
	return capturecache

prettylist2g = lambda l : '%s' % '|'.join("%.2g" % x for x in l)

def formatAccuracies(accuracies: list) -> str:
	result = ''
	for accuracy in accuracies:
		result += f"{accuracy*100.:.1f}|"
	return result

class ValidateModelNotifyBase():
	""" Notifier base class for ValidateModel """
	def __init__(self,
		train_params: TrainingParams = None,
		optimizer=None
	):
		self.train_params = train_params
		self.optimizer = optimizer
		self.snapshotBest = False
		self.bestFileName = ''

	def notify(self, score, epoch, model, optim):
		pass

	@property
	def isSnapshotBest(self):
		""" predicate for snapshot best model enabled """
		result = self.snapshotBest and (self.bestFileName != '')
		return result

class ValidateModel_base(abc.ABC):
	def __init__(self,
		validateset,
		validateproc: Callable,
		interval: int,
		train_params: TrainingParams,
		notifier: ValidateModelNotifyBase = ValidateModelNotifyBase,
	):
		assert(isinstance(train_params, TrainingParams))
		self.validateset = validateset
		self.validateproc = validateproc
		self.interval = interval
		self.train_params = train_params
		self.notifier = notifier
		self.model = None
		self.reset()
		self.resetScores()

	def onceInit(self):
		pass	

	def reset(self):	
		self.counter = 0

	def resetScores(self):
		self.scores = []

	def recordResult(self, score=None, model=None, optim=None):
		self.scores.append(score)

	def finalize(self, model, device='cuda', bar=None, tracectx = None, klog=False):
		pass	

	@abstractmethod
	def doit(self, model, device, bar=None, tracectx = None):
		pass

	@property
	def enabled(self):
		return self.interval > 0

	@property
	def isSnapshotBest(self):
		""" predicate for snapshot best model enabled """
		result = self.notifier.isSnapshotBest if self.notifier else False
		return result

	@property
	def bestSnapshotName(self):
		return self.notifier.bestFileName if self.isSnapshotBest else ""
		
#end of ValidateModel_base		

class ValidateModelNotify(ValidateModelNotifyBase):
	""" A notifier that implements the best snapshot. """
	def __init__(self, 
		train_params: TrainingParams,
		optimizer=None,
		snapshotBest=True	#enable tracking + saving the best snapshot found so far
	):
		super().__init__(train_params, optimizer)
		self.snapshot = train_params['snapshot'] 
		self.snapshotBest = snapshotBest
		self.bestFileName = ''
		print(f"ValidateModelNotify {self.snapshot}")

	def notify(self, score, epoch, model, optim=None):
		recipe = self.train_params['model_recipe']
		datasetname, trset = self.train_params['datasetname'], self.train_params['trset']

		if self.snapshotBest and self.snapshot:
			self.bestFileName = f"{self.snapshot}-best"

			torchutils.save1model(self.bestFileName, model, epoch, 
				optimizer=optim, recipe=recipe, klog=False, 
				datasetname=datasetname, trset=trset,
			)

class ValidateModel(ValidateModel_base):
	def __init__(self, 
		validateset,
		validateproc: Callable, 	#test1model() - i.e. a callable function
		interval: int,
		train_params: TrainingParams,
		notifier: ValidateModelNotifyBase = ValidateModelNotifyBase(),
	):
		super().__init__(validateset, validateproc, interval, train_params)
		self.notifier = notifier

	def onceInit(self):
		if self.enabled:
			print(f"ValidateModel.onceInit(): ", end='')
			validate = self.train_params['validate']
			validate_batchsize = self.train_params['validate_batchsize']
			xform = validate.pipeline
			if isinstance(xform, augmentation.Sequential):
				capturecache = loadAugCache(validate.pipeline, validate.dataset, validate_batchsize)
				#replace the validate augmentation pipeline with the capturecache
				self.train_params._params['validate'] = DataPipeline(validate.dataset, capturecache)

	def reset(self):	
		super().reset()
		self.bestE = -1
		self.best = 0
		self.accuracies = []		
		self.bestHistory = []

	def doit(self, model, device, bar=None, tracectx = None, optim=None):
		self.counter += 1
		if (self.interval == 0):
			return

		if (self.counter % self.interval) == 0:
			params = self.train_params.params	#get the dict()
			validateset = params['validate']
			assert(type(validateset) == DataPipeline)
			#xform = params['validate_xform']
			params['progressbar'] = bar

			print(f"\nValidateModel({len(validateset.dataset)}): ", end="")

			xform = validateset.pipeline
			xform.rewind()	#rewind the replay buffer

			model.eval()
			score = self.validateproc(validateset, model, device, self.train_params, tracectx)
			self.recordResult(score, model, optim)
			
			#continue training
			model.train()	

	def recordResult(self, score=None, model=None, optim=None):
		super().recordResult(score, model)
		epoch = self.counter

		cm, precision, recall, loss = score
		tp = cm.diagonal()
		accuracy = tp.sum() / cm.sum()
		self.accuracies.append(accuracy)

		if accuracy > self.best:	#argmax(accuracy)
			self.best = accuracy
			self.bestE = epoch
			self.notifier.notify(score, epoch, model, optim)
			self.bestHistory.append(epoch)

	def bestRecord(self) -> tuple:		
		return self.bestE, self.best, self.bestHistory

	def finalize(self, 
		model, device='gpu', 
		bar=None, tracectx = None, klog=False
	):
		if len(self.scores) == 0:
			return
		print(f"ValidateModel.finalize:")

		for epoch, score in enumerate(self.scores): 	#TODO: use idxB = np.argsort()
			cm, precision, recall, loss = score
			accuracy = self.accuracies[epoch]
			tracectx.logstr(f"A:{accuracy*100.:.1f}% precision: {around(precision, decimals=4)}")

		print(formatAccuracies(self.accuracies))
		print(f"best[{self.bestE}]: A:{self.best*100.:.1f}%")
		print(f"best history {self.bestHistory}")
#end ValidateModel

#No-Op.
class NoOp(ValidateModel_base):
	def __init__(self, *args, **kwargs):
		return
	def doit(self, *args, **kwargs):
		return

train_subsets = {
	'test':			None,	#no subsetting here
	'train':		None,	#no subsetting here
	'train10k':		10000,  #TODO: Justify need.
	'train20k':		20000,  #TODO: Justify need.
}

def removePrefix(trainWant:str):
	if trainWant.startswith('test'):
		return trainWant[4:]
	if trainWant.startswith('train'):
		return trainWant[5:]
	return trainWant

def parseSubsetSize(trainWant:str):
	suffix = removePrefix(trainWant)
	setsize = None

	if suffix:
		is_k = suffix[-1] =='k'
		setsize = int(suffix[0:-1])*1000 if is_k else int(suffix)
	return setsize

def usingTrainSet(trainWant:str):
	""" predicate for using a subset or full training set for training """
	return "train" in trainWant

#end No-Op.	
def getTrainTest(
	trainWant:str, 
	train:dataset_base.DataSet, 
	test:dataset_base.DataSet,
	useCDF=True 	#use inverse-CDF sampling to maintain class balance	
) -> tuple:
	""" Whether to use training set for training or swap train|test """
	if not usingTrainSet(trainWant):
		train, test = test, train
	print(f" {trainWant=}, {usingTrainSet(trainWant)=}, {train.name=}, {test.name=}")

	#support subset of train|test here
	trainsize = parseSubsetSize(trainWant)
	if trainsize:
		train = datasetutils.getBalancedSubset(train, trainsize/len(train), useCDF=useCDF, name=trainWant)

	return train, test	