# -*- coding: utf-8 -*-
"""
Title: Training pipeline - 
	
	Designed to run several training runs and still guarantee the same result as 
	training a single model.
	
Created on Mon Jul 21 16:01:29 2020

@author: Ujjawal.K.Panchal & Manny Ko
"""
import abc, copy, time, csv
import argparse
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Dict, Union, Optional
from pydantic import BaseModel, ValidationError, validator 
from collections import namedtuple
from numpy import around
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from cplxmodule.nn.relevance import penalties
from cplxmodule.nn.utils.sparsity import sparsity, named_sparsity

from shnetutil.ard import ard
from shnetutil.dataset import dataset_base, datasetutils, fashion
from shnetutil.pipeline import augmentation, batch, dbaugmentations, modelstats, torchbatch, trainutils
from shnetutil.utils import torchutils, trace

from pyutils import testutil

from pipeline import loadmodel


def savemodels(models, epochs, optimizers=None, tag='', recipe=None):
	if optimizers:
		oiter = iter(optimizers)	#TODO: only supports 1 single optimizer for now
		for name, model in models.items():
			if model is not None:
				save1model(f"{name}{tag}", model, epochs, next(oiter), recipe=recipe)
	else:
		for name, model in models.items():
			if model is not None:
				save1model(f"{name}{tag}", model, epochs, optimizer=None, recipe=recipe)

def loadmodels(
	device, 
	models, 
	folder='snapshots/', 
	subfix='', 
	epochs=None,
	optimizer=None	
):
	snapshots = {}
	for name, model in models.items():
		if model is not None:
			snapshot = torchutils.load1model(device, folder, name+subfix, model, epochs, optimizer=optimizer)
			snapshots.update({name: snapshot})
	return snapshots		

def reclaim(models):
	""" Call del to help GPU reclaim resources earlier """
	for model in models:
		del model


def l1_loss(model, weight: float = 0.0):
	if weight <= 0.0:
		return weight
	else:
		l1_parameters= []
		for parameter in model.parameters():
			l1_parameters.append(parameter.view(-1))
		l1_loss = weight * torch.abs(torch.cat(l1_parameters)).sum()
	return l1_loss

def trainloop(
	device,
	train_params: trainutils.TrainingParams,
	run: trainutils.OneRun,
	threshold = 3.0,
	tracectx = None,
	validateproc:trainutils.ValidateModel_base = trainutils.NoOp,
	model = None,
	optimizer = None,
	loi:list = [],
	l2_weight_decay: float = 0.0,
	l1_weight: float = 0.0,
	dropout: float = 0.0,
):
	assert(isinstance(train_params, trainutils.TrainingParams))
	assert(issubclass(type(validateproc), trainutils.ValidateModel_base))
	assert(isinstance(run, trainutils.OneRun))

	recipe = train_params['recipe']
	traindataset, ourTransform = train_params['train']
	batchsize = train_params['batchsize']
	lr = train_params['lr']
	n_steps  = train_params['epochs']
	lr_scedule = train_params['lr_schedule']
	denoise = train_params['denoise']
	ablation = train_params['ablation']
	seed = train_params['seed']

	#1: 1st get us a new model	
	if model is None:
		modelfactory = train_params['modelfactory']
		#print(f"trainloop {modelfactory=}")
		model = modelfactory.makeModel(dropout = dropout, device = device) 	#, ablation_type=ablation
		print(f"trainloop makeModel:{torchutils.modelName(model)=}")

	run.start(model, tracectx, loi, seed=seed)

	#create out batch builder
	trainbatchbuilder = batch.Bagging(traindataset, batchsize, shuffle=False)
	trainset = trainbatchbuilder.dataset
	print(f"training set({trainset.name}), size {len(trainset)}")

	print(f"train transforms:{ourTransform}")
	losses = None
	modelname = torchutils.modelName(model)

	print(f">>>>>> {modelname}, batchsize:{batchsize}, lr:{lr}, ", end='')
	print(f"Model Configuration:{trainutils.getModelConfigStr(model)}")
	torchutils.dumpModelSize(model, details=False)

	#AdamW is slightly better than Adam in the early epochs (1|4E) based on non-exhaustive tests - mck
	optim = optimizer if optimizer else torch.optim.Adam(model.parameters(), lr=lr, weight_decay = l2_weight_decay)
	print(f"optimizer {type(optim)}")

	if lr_scedule:
		milestones, gamma = [5], 0.2
		print(f"MultiStepLR milestones={milestones}, gamma={gamma}")
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=milestones, gamma=gamma)		
	else:
		scheduler = None

	model, losses = model_fit(
		model, trainbatchbuilder, optim, n_steps=n_steps,
		scheduler=scheduler,
		threshold=threshold, klw=0, 
		reduction="mean",		#TODO: nail down mean|sum
		xform = ourTransform, device = device,
		tracectx = tracectx,
		enableEval = True, validateproc = validateproc,
		l1_weight = l1_weight,
	)
	denoiser = ourTransform.xforms[3]
	if isinstance(denoiser, augmentation.Denoise):
		print(f"denoise {denoiser.total}, {denoiser.zeros}, {denoiser.zeros/denoiser.total}")

	return model, optim

def load_best(
	device, 
	train_params: trainutils.TrainingParams, 
	validateproc: trainutils.ValidateModel_base,
	folder='snapshots/', 	
) -> torch.nn.Module:
	assert(isinstance(train_params, trainutils.TrainingParams))
	assert(issubclass(type(validateproc), trainutils.ValidateModel_base))
	notifier = validateproc.notifier
	model = None
	snapshotname = validateproc.bestSnapshotName

	if validateproc.isSnapshotBest:
		print(f"load_best {snapshotname=} --> ")
		loaded = loadmodel.load_model(device, folder=folder, name=snapshotname)
		model = loaded.model

	return model

def trainRuns(
	device,
	model: torch.nn.Module,
	training_runs: List[trainutils.OneRun],
	train_params: trainutils.TrainingParams,
	tracectx: trace.TraceContext,
	validateproc: trainutils.ValidateModel_base,
	snapshot: str = None,
	loi: list = [],
	testmode: int = 1,
	l2_weight_decay: float = 0.0,
	l1_weight: float = 0.0,
	dropout: float = 0.0,
	gen_best_sheet: Optional[str] = None
):
	assert(issubclass(type(validateproc), trainutils.ValidateModel_base))

	validateproc.onceInit()
	bestHistories = []

	if gen_best_sheet:
		print(f"Best Sheet Location: {gen_best_sheet}")

	""" Given a list of training_runs[] each run train the given model """
	for run in training_runs:
		train_params, overrides = run.train_params, run.overrides
		run.doOverrides()

		trace.disable_console(tracectx)

		if run.indep:	#independent runs?
			model = None 	#request fresh model else continue to use previous model

		model, optimizer = trainloop(
			device,
			train_params,
			run,
			tracectx = tracectx,
			validateproc = validateproc,
			model = model,
			optimizer = None,
			loi = loi,
			l2_weight_decay = l2_weight_decay,
			l1_weight = l1_weight,
			dropout = dropout,
		)

		if (snapshot != None):
			tag = snapshot
			recipe = train_params['model_recipe']
			epochs = train_params['epochs']
			trset  = train_params['trset']
			datasetname = train_params['datasetname']

			torchutils.save1model(f"{run.name}{tag}",
				model, epochs, optimizer=optimizer, recipe=recipe, 
				datasetname=datasetname, trset=trset
			)
		#2: test the trained model
		trace.enable_console(tracectx)
	 	#2.1: disable tracing. Usually tracing is only used during training - TODO: use train_params
		if isinstance(model, trace.TraceMixin) and (not train_params.params['trace_test']):
			model.trace = None

		#model = None		#this disables test1model when just want to train or get 'best'
		#2.2:
		with tracectx:
			if testmode > 0:
				test1model(
					train_params['test'],
					model,
					device,
					train_params, 
					tracectx = tracectx,
					runtag=f"{run.name}, ",
					klog = True,
				)
			#torchutils.dumpModelSize(model)
				
			if testmode == 2:
				loadedmodel = load_best(device, train_params, validateproc)
				#loadedmodel if loadedmodel else model
				
				test1model(
					train_params['test'],
					loadedmodel,
					device,
					train_params, 
					tracectx = tracectx,
					runtag=f"{run.name}, ",
					klog = True,
				)

		bestHistories.append(validateproc.bestRecord())
				
		if gen_best_sheet and len(validateproc.accuracies) > 1:
			print(validateproc.accuracies)
			with open(gen_best_sheet, 'w', newline = '') as csvfile:
				writer = csv.writer(csvfile, delimiter = ' ')
				writer.writerow(["Epoch", "Test Accuracy"])
				if False:
					print("Epoch", "Test Accuracy")
					for i in range(1, train_params['epochs'] + 1):
						best_accuracy = max([a for x, a in enumerate(validateproc.accuracies) if x + 1 <= i])
						writer.writerow([i, best_accuracy * 100])
						print(i, best_accuracy * 100)
						print

		#3: see if each run is independent or chained	
			if run.indep:
				validateproc.reset()
				validateproc.resetScores()
				#TODO: create a fresh model instance

			reclaim([model])		#help to recalim GPU resources

			print(bestHistories)


def test1model(
	testpipeline: trainutils.DataPipeline,
	model: torch.nn.Module,
	device,
	train_params: trainutils.TrainingParams,
	tracectx,
	runtag='',
	klog = False,
	batchbuilder: batch.BatchBuilderBase = None,
) -> modelstats.Model_Score:
	#print(f"test1model({ourTransform=})")
	tic1 = time.time()
	assert(type(testpipeline) == trainutils.DataPipeline)
	assert(isinstance(train_params, trainutils.TrainingParams))

	params = train_params.params
	testpipeline = testpipeline if (testpipeline != None) else params['test']
	testdataset, testXform = testpipeline 

	validate_batchsize = params['validate_batchsize']
	threshold = params['threshold']
	progressbar = train_params.progressbar
	testbatchbuilder = batchbuilder if batchbuilder else batch.BatchBuilder(testdataset, validate_batchsize, shuffle=False)

	if model is None:
		return

	modelname = torchutils.modelName(model)
	if klog:
		print(f"\n>>>>>>Test({len(testdataset)}): {runtag}{modelname}")
		#print(f"{ourTransform}")
	#breakpoint()

	score = model_score(
		model, testbatchbuilder,
		threshold=threshold,
		xform = testXform,
		device = device,
		details = False,			#P|A only, no confusion matrix
		tracectx = tracectx,
		bar = progressbar
	)
	if klog:
		testutil.time_spent(tic1, 'test time')

	return score

def testmodels(
	models,
	device,
	train_params,
	tracectx
):
	"""
		models: is a dict keyed by stages (inherited from Ivan's ARD) - e.g. {'None': xx, ...}
			    It is used at the end of the training loop for all stages (each with its own model - e.g. ARD|Masked)
	"""
	assert(isinstance(train_params, trainutils.TrainingParams))
	params = train_params.params
	testdataset = params['test']
	threshold = params['threshold']

	# run tests
	for key, model in models.items():
		if model is None:
			continue

		test1model(
			testdataset,
			model,
			device,
			train_params,
			tracectx,
			runtag=f"stage:{key}, "
		)
		print([*named_sparsity(model, hard=True, threshold=threshold)])


def print_layer_hist(model, layer_name):
	for i, (name, params) in enumerate(model.named_parameters()):
			if name.split(".")[0] == layer_name:
				print(f"{name}'s histogram: {torchutils.histogram(params)}")
	return


def model_fit(model, 
	batchbuilder, 
	optim, n_steps=100, 
	scheduler=None,
	threshold=1.0,
	klw=0.0,			#scale factor for additional penalty (e.g. sparsity) 
	reduction="mean", 
	verbose=True, 
	xform = None, 
	device = "cpu",
	tracectx = None,
	enableEval=False,
	validateproc=trainutils.NoOp(),
	hist_layer = None,
	l1_weight = 0.0
):
	validateproc = validateproc if (validateproc != None) else trainutils.NoOp

	#nested variables
	loss = None
	loss_1 = None
	kl_d = None

	if hist_layer != None:
		print("before training histograms:")
		print_layer_hist(model, hist_layer)

	def model_closure():
		""" reference 'model', 'optim', data', 'target', 'kl_w' """
		nonlocal loss, loss_1, kl_d		#output to nested scope

		optim.zero_grad()
		#2: run forward pass 
		pred = model(data)

		loss_1 = modelstats.softmax_nll(pred, target, reduction=reduction)

		kl_d = sum(penalties(model, reduction=reduction))
		#rint(f"klw {klw:.4f}, kl_d {kl_d:3f}")
		loss = loss_1 + klw * kl_d + l1_loss(model, l1_weight)

		#3: compute gradients using backward pass
		loss.backward()
		return loss

	losses = []
	dbchunk = batchbuilder.dataset

	if not model:
		return None, losses

	with tqdm.tqdm(range(n_steps)) as bar:	#get the progress-bar object
		model.train()

		for ep in bar:
			xform.rewind() 	#rewind the replay buffer (if active)

			epoch = batchbuilder.epoch(False)	#get the generator for our batches for this epoch
			likelihood_coeff = (batchbuilder.size / batchbuilder.batchsize)

			for mybatch in epoch:
				#data := shearlet coefficients from normalized images.
				data, target = torchbatch.getBatchAsync(device, dbchunk, mybatch, xform, logging=False)
				#use Closure to support optimization algorithms that needs to call our model several times
				#https://pytorch.org/docs/stable/optim.html
				optim.step(model_closure)

				losses.append(float(loss))

				f_sparsity = ard.get_sparsity(model, hard=True, threshold=threshold, verbose=verbose)
				bar.set_postfix_str(
					f"{f_sparsity:.1%} {float(loss_1):.3e} {float(kl_d):.3e}"
				)
			if scheduler:
				scheduler.step()	
			# end for
			validateproc.doit(model, device, bar, tracectx, optim)
	# end with
	validateproc.finalize(model, device, bar, tracectx)

	if hist_layer != None:
		print("after training histograms:")
		print_layer_hist(model, hist_layer)	
	return model.eval(), losses


def model_score(
	model, 
	batchbuilder, 
	threshold=1.0, 
	xform=None, 
	device="cpu", 
	details=False, 
	tracectx = None,
	bar=None
) -> modelstats.Model_Score:
	import re

	mstats = modelstats.model_score(model, batchbuilder, threshold, xform, device, details)
	cm, precision, recall, loss = mstats

	# format the arrays and remove clutter
	p_str = re.sub("[',]", "", str([f"{p:4.0%}" for p in precision]))
	r_str = re.sub("[',]", "", str([f"{p:4.0%}" for p in recall]))
	# print(
	# 	f"(S) {f_sparsity:.1%} ({float(kl_d):.2e}) "
	# 	f"(A) {tp.sum() / cm.sum():.1%} ({n_ll.item():.2e})"
	# 	f"\n(P) {p_str}"  # \approx (y = i \mid \hat{y} = i)
	# 	f"\n(R) {r_str}"  # \approx (\hat{y} = i \mid y = i)
	# )
	tp = cm.diagonal()
	print(f"(A) {tp.sum() / cm.sum():.1%} ({loss.item():.2e})")
	if bar:
		bar.set_postfix_str(f"(A) {tp.sum() / cm.sum():.1%}")

	if tracectx:
		tracectx.logstr(f"Confusion Matrix:\n{cm}")
		tracectx.logstr(f"(A) {tp.sum() / cm.sum():.1%} ({loss.item():.2e})")
		tracectx.logstr(f"(P) {p_str}")
		tracectx.logstr(f"(R) {r_str}")
	if details:
		print(re.sub(r"(?<=\D)0", ".", str(cm)))

	return mstats

def eval_train(args, model, device, test_loader):
	model.eval()
	test_loss = 0
	correct = 0
	pred_all = []
	real_all = []
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output, losses = model(data)
			test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
			pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()
			 
	test_loss /= len(test_loader.dataset)
	print('\nTraining set: Average loss: {:.4f}, Overall accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))	


if __name__ == '__main__':
	from shnetutil.pipeline import loadMNIST
	import projconfig

	batchsize = 128
	datasets_root = projconfig.getFashionMNISTFolder()

	fashion_train = loadMNIST.getdb(datasets_root, istrain=True, kTensor = False)
	fashion_test = loadMNIST.getdb(datasets_root, istrain=False, kTensor = False)

	train_params = {
		'train':	 fashion_train,		#10k train
		'test':		 fashion_test,
		'batchsize': batchsize,
	}
	epochs = 1

	phases = {
		"none": 	None,
		"complex":	{'epochs': epochs, 'threshold': 0.0},
		"ard": 		{'epochs': epochs, 'threshold': 1e-2},	#10: 91.3, 20: 91.7, 40: 91.5, 
		"masked":	{'epochs': epochs, 'threshold': 0.0},	#10: 91.7, 20: 91.8, 40: 92.2
	}
	#1: create TrainingParams
	myparams = trainutils.TrainingParams(**train_params)
	print(myparams.params)

	#2: swap test|train
	myparams.swap_traintest()
	print(myparams.params)

	#3: test amend()
	myparams.amend(batchsize=256)
	print(myparams.params)

	ourruns = trainutils.OneRun(train_params, phases, device_)
	print(ourruns)

def shared_args(parser:argparse.ArgumentParser, extras:List[Tuple] =[]):
	""" command line options shared by all models and training scripts """
	assert(type(parser) is argparse.ArgumentParser)
	parser.add_argument('--batchsize', type=int, default=128, metavar='N',
						help='input batch size for training (default: 128)')
	parser.add_argument('--epochs', type=int, default=20, metavar='N',
						help='number of epochs to train (default: 1)')
	parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
						help='learning rate (default: 0.001)')
	parser.add_argument('--threshold', type=float, default=1e-2, metavar='Threshold',
						help='learning rate (default: 1e-2)')
	#parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
						#help='Adam momentum (default: 0.9)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='disables CUDA training')
	parser.add_argument('--log-interval', type=int, default=1, metavar='N',
						help='how many batches to wait before logging training status')
	parser.add_argument('--snapshot', type=str, default= None, help='snapshot name')
	parser.add_argument('--trset', type = str, metavar="test<n>|train<n>",
						default = 'test', help = 'dataset used for training and testing')
	parser.add_argument('--validate', type=int, default=5, help='validate interval')
	parser.add_argument('--vbatchsize', type=int, default=512, help='validate batch size')
	parser.add_argument('--testmode', type=int, default=1, metavar='N', help='final test control')

	parser.add_argument('--hist_layer', type = str, default = None, help = 'Whether to show this layer\'s weight and bias histograms before and after training.')
	parser.add_argument('--denoise', type=int, default=0, metavar='0|1', help='denoise Shearlets') #off by default.
	parser.add_argument('--lr_schedule', action='store_true', default=False, help='Enable MultiStepLR')
	
	#regularization.
	parser.add_argument('--wd', type = float, default = 0.0, help = 'l2 regularization weight decay parameter for optimizer.')
	parser.add_argument('--l1', type = float, default = 0.0, help = 'l1 regularization weight term.')
	parser.add_argument('--dropout', type = float, default = 0.0, help = 'dropout value for dropout layer added after each pool layer. No dropout layer if 0.0')
	
	#ablation.
	parser.add_argument('--ablation', type=str, default=None, choices=(None,'nosh-cplx','nosh-real'), 
		help='ablation of test_fashion.\n Available:\
		1. none (No Ablation).\
		2. nosh-cplx (CVnn but no sherlets).')

	#2. add extras.
	for extra_arg in extras:
		parser.add_argument(*extra_arg[0], **extra_arg[1])

	return parser

def onearg(*args, **kwargs):
	return args, kwargs

def dataset_select(
	datasetname:str,
	trset:str,
	args,
	colorspace = None,		 #TODO: integrate recipe completely in all calls to dataset_select.
	validate:float = 0.1,
	device='cpu',
	train_perturbation: bool = False,
	test_perturbation: bool = False,
	coshrem_args: Optional[BaseModel] = None,
) -> datasetutils.TrainingSet:
	assert(type(trset) == str)
	if (datasetname == 'fashion') or (datasetname == 'mnist'):
		training_set, test_set, validateset = fashion.load_fashion(
			trset, validate=validate, datasetname=datasetname
		)
		trainXform, testXform, validateXform = dbaugmentations.fashion_augmentations(
			fashion.kMean,
			fashion.kStd,
			denoise = args.denoise,
			ablation_type = args.ablation,
			device=device,
			train_additional_xforms = [(0, augmentation.GaussianNoise(mean = 0.0, variance = 50.0)), (1, augmentation.GaussianBlur(max_sigma = 0.75))] if train_perturbation else None,
			test_additional_xforms = [(0, augmentation.GaussianNoise(mean = 0.0, variance = 50.0)), (1, augmentation.GaussianBlur(max_sigma = 0.75)) ] if test_perturbation else None,
			coshrem_args = coshrem_args
		)
	else:
		raise Exception("No other datasets supported yet!")
	return datasetutils.TrainingSet(training_set, test_set, validateset, trainXform, testXform, validateXform)


if __name__ == '__main__':
	def ourArgs(extras:List[Tuple] = []):
		#1. argparse settings
		parser = argparse.ArgumentParser(description='t-ouragrs')
		
		#2. add shared args.
		shared_args(parser)

		for extra_arg in extras:
			parser.add_argument(*extra_arg[0], **extra_arg[1])

		args = parser.parse_args()
		return args

	args = ourArgs([
		training.onearg('--train', type=int, default=0, help='continue training'),
	])
	print(args)
