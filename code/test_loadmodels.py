"""
Title: test_loadmodels.

Created on Sun Aug 16 17:44:29 2020

@author: Ujjawal.K.Panchal & Manny Ko & Hector Andrade-Loarca.
"""

#Python imports
import argparse, tqdm, pprint, warnings
#PyTorch
import torch, numpy as np
#cplxmodule

#our packages
from shnetutil.dataset import fashion
from shnetutil.utils import torchutils, trace
from shnetutil.pipeline import loadMNIST, logutils, trainutils
from shnetutil.modelling import modelfactory

from pipeline import training, loadmodel
from modelling import CVnn

import test_fashion

kNewFactory=False

#filter warnings.
warnings.filterwarnings("ignore")


def ourArgs():
	args = test_fashion.ourArgs([
		training.onearg('--dataset', type=str, default='fashion', choices=('fashion','cifar10'), help='dataset'),
		training.onearg('--train', type=int, default=0, help='continue training'),
		training.onearg('--all', action='store_true', default=False, help='load all snapshots'),
	])
	return args

def gatherSnapshot(snapshot_folder='snapshots/'):
	totest = Path(snapshot_folder).glob("*.pth")
	return totest

def main():
	args = ourArgs()

	mylogger = logutils.getLogger("test_loadmodels")	
	logutils.setup_logger(mylogger, file_name='logs/test_loadmodels.log', kConsole=True)

	device = torchutils.onceInit(kCUDA = True)

	dataset = args.dataset

	#1: get standard recipes based on user command line arguments
	#epochs = args.epochs
	snapshot = args.snapshot if args.snapshot else '10K1E'

	#3: trace context
	tracectx = trace.TraceTorch(
		mylogger=mylogger,	#mylogger
		kCapture=False, picklename='logs/test_loadmodels.pkl', 
		capture=10,		#number of log() calls to capture
		kCheckSum=False #enable checksum or capture raw numpy/tensor
	)

	loaded, datasetname, model, optim = loadmodel.load_model(device, 
		folder='snapshots/', name=snapshot, 
		datasetname=args.dataset, 
		tracectx=tracectx,
		lr=args.lr,
	)
	if loaded:
		recipe = loaded['recipe'] 	#use recipe from snapshot
		#trset = args.trset  	#TODO: remove this once the loaded['trset'] works 100%
		trset = loaded.get('trset', args.trset)
	
		our_dataset = training.dataset_select(datasetname, trset, args, colorspace = recipe.colorspace, device=device)

		training_set, test_set, validate_set, trainTransform, testTransform, validateTransform = our_dataset
		real_test_set = test_set

		#test_set = validate_set

			
		epochs = args.train
		print(f"continue to train for {epochs} epochs..")

		#2: default set of training parameters
		train_params = trainutils.TrainingParams(
			modelfactory = None,
			recipe = recipe,
			train  = trainutils.DataPipeline(training_set, trainTransform),
			test   = trainutils.DataPipeline(test_set, testTransform),
			validate = trainutils.DataPipeline(validate_set, validateTransform),
			batchsize = args.batchsize,
			threshold = args.threshold,
			lr = args.lr,
			epochs = epochs,
			trset = trset
		)
		if (epochs > 0):
			#parameters overrides for each phase of the training - batchsize, test set etc.
			overrides = {
				'epochs': 	epochs,
				'batchsize': 256, 
			}
			#we can run several training runs, one after another.  Each run will be reproducible.
			training_runs = [
				trainutils.OneRun(train_params, runname="baserun", overrides=overrides),
				#2nd run with different overrides:
				#OneRun(train_params, runname="run2", overrides={'epochs': epochs, 'test': real_test_set, }),
			]
			validate = args.validate
			notifier = trainutils.ValidateModelNotify(train_params, optimizer=None)
		
			validateproc = trainutils.ValidateModel(
				test_set, training.test1model, 
				validate, 
				train_params,
				notifier=notifier,
			)
			for run in training_runs:
				trace.disable_console(tracectx)

				model, optimizer = training.trainloop(
					device,
					run.train_params,
					tracectx = tracectx,
					validateproc = validateproc,
					model = model,
					optimizer=optim,
				)
				#2: test the trained model
				trace.enable_console(tracectx)

		with tracectx:
			training.test1model(
				train_params['test'],
				model,
				device,
				train_params, 
				tracectx = tracectx,
				runtag="test loaded model:",
				klog=True
			)
				
	torchutils.shutdown()	

def test_trsets():
	""" a list of snapshots trained with trset="train" """
	snapshots = [
		'50k5E-best',
		'cifar50k20E-best',
		'cvnncifar50k20E',
		'cvnncifar50k2E',
		'cifar50k2E-best',
	]
	mylogger = logutils.getLogger("test_loadmodels")	
	logutils.setup_logger(mylogger, file_name='logs/test_loadmodels.log', kConsole=True)

	device = torchutils.onceInit(kCUDA = True)

	#3: trace context
	tracectx = trace.TraceTorch(
		mylogger=mylogger,	#mylogger
		kCapture=False, picklename='logs/test_loadmodels.pkl', 
		capture=10,		#number of log() calls to capture
		kCheckSum=False #enable checksum or capture raw numpy/tensor
	)

	for snapshot in snapshots:
		loaded, datasetname, model, optim = loadmodel.load_model(device, 
			folder='snapshots/', name=snapshot, 
			datasetname='fashion',	#this is a default will be extracted from snapshot unless it is a very old one 
			tracectx=tracectx,
			lr=.001,
		)
		if loaded:
			recipe = loaded['recipe'] 	#use recipe from snapshot
			trset = loaded.get('trset', "test")
			print(f"{loaded['trset']=}")	
			assert(trset=="train")

if __name__ == '__main__':
	main()
	#test_trsets()

