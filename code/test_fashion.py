"""
Title: test_fashion.

Created on Sun Aug 16 17:44:29 2020

@author: Ujjawal.K.Panchal & Manny Ko & Hector Andrade-Loarca.
"""

#Python imports
from pathlib import Path
import argparse, warnings, time
from typing import List, Tuple, Optional, Callable
from functools import partial
import numpy as np

#PyTorch
import torch
#complex packages for our complex neural network.

#our packages.
from pyutils.testutil import time_spent
import pyutils.dirutils as dirutils

from shnetutil import projconfig
from shnetutil.dataset import datasetutils, fashion
from shnetutil.pipeline import logutils, trainutils
from shnetutil.modelling import modelfactory
from shnetutil.utils import torchutils, trace

#our local modules
from modelling import CVnn, modelfactories, fashionRecipes
from pipeline import training, gen_runs

#filter warnings.
warnings.filterwarnings("ignore")
kFashionColorspace = "grayscale"

kTTFactorMethod = "range" #change to 'gcd' for tiny2.
kTTFactorRange = (20, 30)
BestAccuracyFileName = "tiny-CoShNet-Accuracies.csv"
conv2d_dispatch = {
	'complex': 	 CVnn.ConvLayers.kSplitConv,
	'dcf': CVnn.ConvLayers.kSplitDCF,
}


def ourArgs(extras:List[Tuple] = []):
	""" Fashion|MNIST shared args """
	#1. argparse settings
	parser = argparse.ArgumentParser(description='CoShREM NN based on cplex')
	
	#2. add shared args.
	training.shared_args(parser)

	#3. add custom args.
	parser.add_argument('--recipe', type = str, choices = fashionRecipes.recipe_mapper.keys(), 
		default = 'CoShCVNN', help = "type of CoShCVNN model to use. ")

	#4. add extras.
	for extra_arg in extras:
		parser.add_argument(*extra_arg[0], **extra_arg[1])

	#if not '--recipe' in extra_arg[0]: 	#use this to add --recipe when needed
		#print("add recipe to fashion")

	args = parser.parse_args()
	return args

def fashionArgs(extras:List[Tuple] = [], tests:Optional[list]=None):
	""" Fashion dataset specific args """
	args = ourArgs([
		training.onearg('--dataset', type=str, default='fashion', choices=('fashion','mnist'), help='dataset'),
		training.onearg('--test', type=str, default=None, choices=tests, help='run named test'),
		training.onearg('--train_perturb', action='store_true', default=False, help='whether to perturb the training set or not.'),
		training.onearg('--test_perturb', action = 'store_true', default = False, help = 'whether to perturb the test set or not.'),
		training.onearg('--trset_size', type = int, default = None, help = 'any particular training set size you want (max = size of trset).'),
		training.onearg('--tt_init', type = str, default = None, help = 'the type of tt init you want.\
			(note setting this parameter will override your recipe to make a tiny recipe).'),
	])
	return args

def ourRecipe(
	args,
	colorspace:str = kFashionColorspace,
	tt_factor_method: str = kTTFactorMethod, #QUERY: Unused var. Can be removed?
	tt_factor_range: tuple = kTTFactorRange, #QUERY: Unused var. Can be removed?
) -> modelfactory.Recipe_base: #TODO: Adapt colorspace aware recipe here if/when to merge with recipe.
	coshrem_args, recipe0 = fashionRecipes.recipe_mapper[args.recipe]

	#TODO: This following hack must be removed.
	if args.tt_init:
		ttdesc = fashionRecipes.ttdesc_mapper.get(args.recipe, fashionRecipes.kTT_fc1_desc1)
		ttdesc.tt_init = args.tt_init
		recipe0 = CVnn.makeTiny(fashionRecipes.kCoShNetRecipe,ttdesc)
	return coshrem_args, recipe0

def gen_epochs(epochs:tuple):
	""" a demo run generator that generates a range of test epochs using generateRuns() """
	for epoch in epochs:
		overrides = {
			'epochs':	{'epochs': epoch},
		}
		yield overrides, epoch

# testrun(train_params=train_params, epochs=epochs)
kTestRuns = {	#use partial to bind each generating routine's specific args
	'seeds':	partial(gen_runs.generateRandomSeeds, number_of_runs=10),
	'lrs':		partial(gen_runs.lr_rateRuns, number_of_runs=6, lr_0=.001, step=.001),
	'bsizes':	partial(gen_runs.batchsizeRuns, number_of_runs=6, bsize=32, step=32),
	'epochs':	partial(gen_runs.generateRuns, generator=gen_epochs(range(2,10,2))),
}

def main(
	pyfilename: Path,
	logname="fashionCoShREM",
	kSubSample=False
):
	dirutils.mkdir('logs')
	seed = 1	#1|99|111|999|3407 10K10E: 87.1|88.7|88.5|88.2|89.1
	device = torchutils.onceInit(kCUDA = True, seed=seed)
	#argparse settings
	args = fashionArgs(tests=kTestRuns.keys())
	datasetname = args.dataset
	trset_size = args.trset_size
	mylogger = logutils.getLogger(logname)
	logutils.setup_logger(mylogger, file_name=f'logs/{logname}.log', kConsole=True)

	#2:config our recipe based from 'args'
	coshrem_args, recipe= ourRecipe(args)
	
	#2.1: select our dataset = 'fashion'
	fashion_dataset = training.dataset_select(
		datasetname, 		#fashion|mnist
		args.trset,
		args,
		colorspace=recipe.colorspace,
		validate = 1.0 if trainutils.usingTrainSet(args.trset) else .2,
		device = device,
		train_perturbation = args.train_perturb,
		test_perturbation = args.test_perturb,
		coshrem_args = 	coshrem_args,
	)

	training_set, test_set, validate_set, trainTransform, testTransform, validateTransform = fashion_dataset
	print(f"Test transform:\n{testTransform}\n===\nValidate transform:\n{validateTransform}\n===.")
	
	if kSubSample:	#Subsample our training set (1k-5k etc.):
		training_set = datasetutils.getBalancedSubset(training_set, 0.3, offset=0)

	print(f"** validate_set {len(validate_set)}")

	#training set subsizing.
	if trset_size:
		training_set = datasetutils.getBalancedSubset(
										training_set,
										trset_size/len(training_set),
										offset=0,
										name="training_set"
									)

	#test_set = validate_set
	
	#2.2 create our trace context
	tracectx = trace.TraceTorch(
		mylogger=mylogger,	#mylogger
		kCapture=False, picklename='logs/' + pyfilename.stem + '.pkl',
		capture=4,		#number of log() calls to capture
		kCheckSum=False, #enable checksum or capture raw numpy/tensor
		complex_type = "trabelsi" #trabelsi neural network.
	)
	tracectx._enable = False
	trace.disable_console(tracectx)
	
	# Models and training setttings.
	modelstage = modelfactories.ModelFactory(recipe, tracectx, args.ablation)
	epochs = args.epochs
	batchsize = args.batchsize
	validate_batchsize = 512
	validate = args.validate

	#if False and (args.trset == 'test'):	#only cache 10k
	#	capturecache = trainutils.loadAugCache(trainTransform, training_set, batchsize)
	#	trainTransform = capturecache

	#default set of training parameters
	train_params = trainutils.TrainingParams(
		modelfactory = modelstage,
		recipe = recipe,	
		train  = trainutils.DataPipeline(training_set, trainTransform),
		test   = trainutils.DataPipeline(test_set, testTransform),
		validate = trainutils.DataPipeline(validate_set, validateTransform),
		validate_batchsize = validate_batchsize,
		batchsize = args.batchsize,
		threshold = args.threshold,
		epochs = args.epochs,
		lr = args.lr,
		loss_reduction = "mean",
		hist_layer = args.hist_layer,
		model_recipe = recipe,
		lr_schedule = args.lr_schedule,
		denoise=args.denoise,
		ablation=args.ablation,
		snapshot=args.snapshot,
		datasetname=args.dataset, 	#fashion|mnist
		trset=args.trset,
	)
	
	#parameters for each phase of the training - complex|ard|masked etc.
	overrides0 = {
		"cvnn":	{'epochs': epochs, 'batchsize': 128},	 	#10k20E: 89.2 (60k) tiny2
		#"cvnn":	{'epochs': epochs, 'batchsize': 192}, 	#10k20E: 89.3 (60k) tiny2
	}
	overrides1 = {
		"cvnn":	{'epochs': epochs, 'batchsize': 32},
	}
	#we can run several training runs, one after another.  Each run will be reproducible.
	#
	training_runs = None
	normal_run = not args.test

	if not normal_run:
		testrun = kTestRuns[args.test]
		training_runs, trials = testrun(train_params=train_params, epochs=epochs)

	if normal_run:	
		training_runs = [
			# amend the training params for a different training run
			trainutils.OneRun(train_params, overrides=overrides0, runname="", indep=True),					
#			trainutils.OneRun(train_params, overrides=overrides1, runname="run1", indep=True ),					
		]
	#layers we are interested in capturing/inspecting:
	loi = [
#		'act_conv1', 
#		'act_conv2',
	]
	validateproc = training.test1model
	notifier = trainutils.ValidateModelNotify(train_params, optimizer=None)

	validateproc = trainutils.ValidateModel(
		validateset=validate_set, 
		validateproc=validateproc, 
		interval=validate, 
		train_params=train_params,
		notifier=notifier,
	)
	#model = modelstage.makeModel(device)
	model = None 	#let ModelFactory in trainloop() do it

	training.trainRuns(
		device,
		model,
		training_runs,
		train_params,
		tracectx,
		validateproc,
		args.snapshot,
		loi=loi,
		testmode=args.testmode,
		l2_weight_decay = args.wd,
		l1_weight = args.l1,
		dropout = args.dropout,
		gen_best_sheet = BestAccuracyFileName
	)

	if not normal_run:
		print(f"{trials=}")

	torchutils.shutdown()


if __name__ == '__main__':
	main(pyfilename=Path(__file__), logname="fashionCoShREM")

