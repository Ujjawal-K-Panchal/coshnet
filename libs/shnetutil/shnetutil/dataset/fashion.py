# -*- coding: utf-8 -*-
"""

Title: Fashion dataset related support.
    
Created on Fri Aug 20 17:44:29 2021

@author: Manny Ko & Ujjawal.K.Panchal

"""
from collections import namedtuple
from typing import List, Tuple, Union, Optional

from .. import projconfig
from . import datasetutils
from ..pipeline import loadMNIST, augmentation, dbaugmentations, trainutils


kImageSize = 28
kMean = 0.5042909979820251	#TODO: document this
kStd  = 1.1458107233047485	#TODO: document this
#calculating with
cplxmean = 26.08305905336539
normdshmean = 0.17420224977235

def load_fashion(
	trset: str,			#train|test for training
	kTensor: bool = False,	
	validate: float = 0.2,		#fraction of test set for validation during training
	kLogging: bool = False,
	datasetname: str = "fashion",
) -> tuple:
	# don't reverse the roles of the MNIST train-test split
	dispatch = {
		"fashion": projconfig.getFashionMNISTFolder(),
		"mnist":   projconfig.getMNISTFolder(),
	}
	datasets_root = dispatch[datasetname]
	print(datasets_root)
	fashion_train = loadMNIST.getdb(datasets_root, istrain=True, kTensor = kTensor)
	fashion_test = loadMNIST.getdb(datasets_root, istrain=False, kTensor = kTensor)
	training_set, test_set = trainutils.getTrainTest(trset, fashion_train, fashion_test, useCDF=True)

	#1: subset our training set?
	#training_set = datasetutils.getBalancedSubset(training_set, 0.2)
	#1.1: subset our validate set?
	validateset = datasetutils.getBalancedSubset(test_set, validate, offset=0, kLogging=kLogging, name="validasetset")		#6k (assuming 60Kxx) validate set for frequent validate

	return training_set, test_set, validateset

def fashion_set(args, validate=0.1)-> tuple:
	fashion_dataset = fashion.load_fashion(
		args.trset, validate=0.1, 
	)
	return fashion_dataset


if __name__ == '__main__':
	fashion_train, fashion_test, validateset, *_ = load_fashion('train', validate=0.2)
	print(fashion_train, fashion_test, validateset)