"""
Title: Test Resnet: For testing ResNet in our exp pipeline for robust comparison.

Created on a day I forgot to add date.

@author: Ujjawal.K.Panchal & Manny Ko.
"""
import argparse
from torchvision.datasets import FashionMNIST
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
from torch.utils.data import random_split,DataLoader
import os
import copy

from pathlib import Path
from pipeline import training
from shnetutil import projconfig
from shnetutil.utils import torchutils, trace
from shnetutil.pipeline import logutils, trainutils, batch, augmentation
from shnetutil.dataset import fashion
from shnetutil.dataset import datasetutils


from pyutils import testutil
plt.ion()





class PseudoValidateBase():
	def __init__(self):
		self.resetScores()
	def recordResult(self, score=None, model=None, optim=None):
		self.scores.append(score)
	def resetScores(self):
		self.scores = []
	def reset(self):	
		self.counter = 0

class ResNetValidateProc(PseudoValidateBase):
	def __init__(
		self,
		validateset, 
		validateproc, 
		interval,
		train_params,
		notifier=None,
	):
		super().__init__()
		self.validateset = validateset
		self.validateproc = validateproc
		self.interval = interval
		self.train_params = train_params
		self.notifier = None
		self.reset()
		self.resetScores()
		return

	def resetScores(self):
		self.scores = []

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
			#params.
			params = self.train_params	#get the dict()
			#assert(type(validateset) == trainutils.DataPipeline)
			#xform = params['validate_xform']
			params['progressbar'] = bar
			bs = params["validate_batchsize"]
			validateSet, validateTransform = params["validate"]
			
			#xform and rewinding.
			xform = validateTransform
			xform.rewind()	#rewind the replay buffer
			model.eval()

			#verbosity.
			print(f"\nValidateModel({len(self.validateset.dataset)}): ", end="")
			
			#recording score.
			score = self.validateproc(self.validateset, xform, model, batchsize = bs, device = device)
			self.recordResult(score, model, optim)
			
			#continue training
			model.train()
		return

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
			#self.notifier.notify(score, epoch, model, optim)
			self.bestHistory.append(epoch)
		return
	
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
			#tracectx.logstr(f"A:{accuracy*100.:.1f}% precision: {around(precision, decimals=4)}")

		print(trainutils.formatAccuracies(self.accuracies))
		print(f"best[{self.bestE}]: A:{self.best*100.:.1f}%")
		print(f"best history {self.bestHistory}")


def make_resnet_model(
	modelname: str = 'resnet18',
	num_classes: int = 10,
	pretrained = False,
	device = torch.device('cuda:0')
):
	if modelname.lower() == 'resnet18':
		model = models.resnet18(pretrained=pretrained) #No pretraining for fair comparison.
	elif modelname.lower() == 'resnet50':
		model = models.resnet50(pretrained=pretrained) #No pretraining for fair comparison.
	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, num_classes)
	model = model.to(device)
	return model

def resnet_trainloop(
	model,
	traindataset,
	ourTransform,
	n_steps = 20,
	threshold = 3.0,
	batchsize = 128,
	tracectx = None,
	validateproc: trainutils.ValidateModel_base = trainutils.NoOp,
	optimizer = None,
	lr = 0.001,
	lr_schedule = None,
	loi: list = [],
	device = None,
	l2_weight_decay: float = 0.0,
	l1_weight: float = 0.0,
	dropout: float = 0.0,
):
	if device == None:
		device = torch.device('cuda:0')
	trainbatchbuilder = batch.Bagging(traindataset, batchsize, shuffle=False)
	trainset = trainbatchbuilder.dataset
	print(f"training set({trainset.name}), size {len(trainset)}")
	print(f"train transforms:{ourTransform}")
	losses = None
	modelname = torchutils.modelName(model)
	print(f">>>>>> {modelname}, batchsize:{batchsize}, lr:{lr}, ", end='')
	optim = optimizer if optimizer else torch.optim.Adam(model.parameters(), lr=lr, weight_decay = l2_weight_decay)
	print(f"optimizer {type(optim)}")

	if lr_schedule:
		milestones, gamma = [5], 0.2
		print(f"MultiStepLR milestones={milestones}, gamma={gamma}")
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=milestones, gamma=gamma)		
	else:
		scheduler = None

	model, losses = training.model_fit(
		model, trainbatchbuilder, optim, n_steps=n_steps,
		scheduler=scheduler,
		threshold=threshold, klw=0, 
		reduction="mean",		#TODO: nail down mean|sum
		xform = ourTransform,
		device = device,
		tracectx = tracectx,
		enableEval = True,
		validateproc = validateproc,
		l1_weight = l1_weight,
	)
	return model, optim


def resnet_testloop(
	testdataset,
	testXform,
	model: torch.nn.Module,
	batchsize = 128,
	testmode = 0,
	device = torchutils.onceInit(kCUDA = torch.cuda.is_available(), seed = 42),
	runtag='',
	klog = False,
	batchbuilder: batch.BatchBuilderBase = None,
):
	
	#print(f"test1model({ourTransform=})")
	tic1 = time.time()

	validate_batchsize = batchsize
	threshold = 3.0
	progressbar = None
	testbatchbuilder = batch.BatchBuilder(testdataset, batchsize, shuffle=False)

	if model is None:
		return

	modelname = torchutils.modelName(model)

	if klog:
		print(f"\n>>>>>>Test({len(testdataset)}): {runtag}{modelname}")
	
	score = training.model_score(
		model, testbatchbuilder,
		threshold=threshold,
		xform = testXform,
		device = device,
		details = False,			#P|A only, no confusion matrix
		tracectx = None,
		bar = progressbar
	)
	if klog:
		testutil.time_spent(tic1, 'test time')
	return score

if __name__ == "__main__":
	#arg values.
	parser = argparse.ArgumentParser(description='CoShREM NN based on cplex')
	training.shared_args(parser)

	parser.add_argument('--dataset', type = str, default = 'fashion')
	parser.add_argument('--seed', type=int, default=0, metavar='N',
							help='random seed value')

	parser.add_argument('--modelname', type = str, metavar="resnet18<n>|resnet50<n>",default = 'resnet18')
	parser.add_argument('--trset_size', type = int, default = None)
	args = parser.parse_args()
	args.ablation = "nosh-real" #override ablation arg.
	seed = args.seed
	rnd = np.random.RandomState(seed)
	resnetmodelname = args.modelname
	datasetname = args.dataset
	trset_size = args.trset_size
	epochs  = args.epochs
	logname = f"{datasetname}ResNet18"
	batchsize = args.batchsize
	num_classes = 10 #NUMBER OF CLASSES IN DATASET.
	colorspace = "grayscale" if datasetname.lower() == "fashion" else "lab"
	validate = 1.0 #validate set size.
	lr = 0.001
	lr_schedule = args.lr_schedule
	device = torchutils.onceInit(kCUDA = torch.cuda.is_available(), seed=seed)
	classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
	           'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
	pyfilename = Path(__file__)


	#get dataset.

	fashion_dataset = training.dataset_select(
			datasetname, 		#fashion
			args.trset,
			args, 
			colorspace=colorspace, 
			validate = validate,#0.5 if trainutils.usingTrainSet(args.trset) else 1.0,
			device = device,
	)
	training_set, test_set, validate_set, trainTransform, testTransform, validateTransform = fashion_dataset
	
	if trset_size:
		training_set = datasetutils.getBalancedSubset(training_set, trset_size/len(training_set), offset=0, name="training_set")

	print(f"length of sets (train, validate, test): ({len(training_set)}, {len(validate_set)}, {len(test_set)})")
	# if datasetname == "fashion":
	# 	fashion_train = FashionMNIST(root=projconfig.getFashionMNISTFolder(), train=True, download=True)
	# 	fashion_test  = FashionMNIST(root=projconfig.getFashionMNISTFolder(), train=False, download=True)
	# if args.trset == "test":
	# 	training_set, test_set = fashion_test, fashion_train
	# else:
	# 	training_set, test_set = fashion_train, fashion_test
	validateset = datasetutils.getBalancedSubset(test_set, validate, offset=0, name="validasetset")
	print(validateset)

	#override augs.

	if datasetname == "fashion":
		mean, std = fashion.kMean, fashion.kStd
		ourFashionTransform = augmentation.Sequential([
		augmentation.Normalize(mean, std),
		augmentation.Pad([(0,0), (2,2), (2,2)]),		  
		augmentation.RepeatDepth(n_times = 3, device = device),
		augmentation.ToTorchDims(),
		])
		trainTransform, testTransform, validateTransform = ourFashionTransform, ourFashionTransform, ourFashionTransform

	#get model.
	model = make_resnet_model(modelname = resnetmodelname, num_classes = 10, pretrained = False, device = device)
	pytorch_total_params = sum(p.numel() for p in model.parameters())
	print(f"{resnetmodelname}: paramters: {pytorch_total_params}")

	#TODO: due to model stage and recipe, we cannot use the existing training loop and hence require to design our own. 
	#hence proceeding to replicate here.

	validate_params = {"validate": trainutils.DataPipeline(validate_set, validateTransform), "validate_batchsize": batchsize}
	validateproc = resnet_testloop
	notifier = None

	validateproc = ResNetValidateProc(
		validateset=validate_set,
		validateproc=validateproc, 
		interval=args.validate,
		train_params=validate_params,
		notifier=None,
	)

	#training loop.
	resnet_trainloop(
		model = model,
		traindataset = training_set,
		ourTransform = trainTransform,
		n_steps = epochs,
		batchsize = batchsize,
		validateproc = validateproc,
		lr = lr,
		lr_schedule = lr_schedule,
		loi = [],
		device = device,
		l2_weight_decay = 0.0,
		l1_weight = 0.0,
		dropout = 0.0,
	)

	#testing loop.
	resnet_testloop(
		test_set,
		testTransform,
		model,
		batchsize = 128,
		testmode = 0,
		device = device,
		runtag='',
		klog = True,
		batchbuilder = None,
	)

	torchutils.shutdown()

