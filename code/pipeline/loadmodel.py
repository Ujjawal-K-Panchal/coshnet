"""
Title: test_loadmodels.

Created on Sun Aug 16 17:44:29 2020

@author: Manny Ko &  Ujjawal.K.Panchal & Hector Andrade-Loarca.
"""

#Python imports
import argparse, warnings
from collections import namedtuple
from typing import Callable, List, Tuple, Union, Optional

#PyTorch
import torch, numpy as np

#our packages
from shnetutil.dataset import fashion
from shnetutil.utils import torchutils, trace
from shnetutil.pipeline import logutils
from shnetutil.modelling import modelfactory

from pipeline import training
from modelling import CVnn, modelfactories


kUseRecipe=True

#filter warnings.
warnings.filterwarnings("ignore")

LoadModel_Result = namedtuple("LoadModel_Result", "snapshot, datasetname, model, optim", defaults=(None,)*4)

def infer_dataset(loaded: dict, datasetname:str):
	datasetname = loaded.get('dataset', datasetname)
	return datasetname

def load_restore(
	device,
	loaded:dict,
	recipe: modelfactory.Recipe_base,
	lr: float, 
	ablation_type: Optional[str] = None,
	dropout: float = 0.0
):
	modelfactory = modelfactories.ModelFactory(recipe, ablation_type = ablation_type)
	model = modelfactory.makeModel(device = device, dropout = dropout)
	#print(f"model {model.getModelConfigStr()}")
	torchutils.restore_state(model, loaded)

	optim = torch.optim.Adam(model.parameters(), lr=lr)
	torchutils.restore_optimizer(optim, loaded)

	return model, optim

def load_model(
	device, 
	folder='snapshots/', 
	name:str='complex10K1E', 
	datasetname = 'fashion',
	ablation_type = None,
	lr: float = 0.001, 
	tracectx: trace.TraceContext = None,
	kLogging=True,
	dropout: float = 0.0
) -> LoadModel_Result:

	loaded = torchutils.load1model(
		device, 
		folder='snapshots/', name=name, 
		model=None,		# if kUseRecipe else model 
		optimizer=None
	)
	datasetname, model, optim = None, None, None
	
	if loaded:
		datasetname = infer_dataset(loaded, datasetname)	
		trset = loaded.get('trset', "test")

		if kLogging:
			print(f"loaded: {loaded.keys()}, epoch: {loaded.get('epoch', None)}")
			if tracectx:
				tracectx.logstr(f"loaded: '{name}', epoch: {loaded.get('epoch', None)}")
				tracectx.logstr(f"loaded['recipe']: {loaded['recipe'].modeltype}")
			print(f"dataset '{datasetname=}', {trset=}")

		#4.0 create the factory using the loaded recipe
		recipe = loaded['recipe']
		assert(recipe is not None)
		model, optim  = load_restore(device, loaded, recipe, lr, ablation_type = ablation_type, dropout = dropout)

	return LoadModel_Result(loaded, datasetname, model, optim)

def ourArgs(extras:List[Tuple] = []):
	parser = argparse.ArgumentParser(description='CoShREM NN based on cplex')
	
	#2. add shared args.
	training.shared_args(parser)

	#4. add extras.
	for extra_arg in extras:
		parser.add_argument(*extra_arg[0], **extra_arg[1])

	args = parser.parse_args()
	return args


def unittest():
	args = ourArgs()

	mylogger = logutils.getLogger("test_loadmodels")	
	logutils.setup_logger(mylogger, file_name='logs/test_loadmodels.log', kConsole=True)

	device = torchutils.onceInit(kCUDA = True)

	#1: get standard recipes based on user command line arguments
	datasetname = args.dataset
	epochs = args.epochs
	lr = args.lr
	snapshot = args.snapshot if args.snapshot else 'complex10K1E'

	#3: trace context
	tracectx = trace.TraceTorch(
		mylogger=mylogger,	#mylogger
		kCapture=False, picklename='logs/test_loadmodels.pkl', 
		capture=0,		#number of log() calls to capture
		kCheckSum=False #enable checksum or capture raw numpy/tensor
	)

	loaded, datasetname, model, optim = load_model(device, 
		folder='snapshots/', name=snapshot, 
		datasetname=datasetname, 
		tracectx=None,
		lr=lr,
	)
	if loaded: 
		trset = loaded.get('trset', "test")
	
		our_dataset = training.dataset_select(datasetname, trset, args, recipe = loaded['recipe'])

		training_set, test_set, validateset, ourTrainTransform, ourTestTransform = our_dataset
		real_test_set = test_set
		test_set = validateset	

if __name__ == '__main__':
	unittest()
