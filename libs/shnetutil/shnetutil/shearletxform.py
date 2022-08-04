# -*- coding: utf-8 -*-
"""

Title: Shearlet based CNN vs. simple MLP in Fashion MNIST.
    
Created on Mon Mar 16 17:44:29 2020

@author: Ujjawal.K.Panchal & Manny Ko

#
# refactored from on-disk-coeffs/only-10k-examples/Main.py

"""
import os, gzip, struct
import argparse, pickle
import random, time
import numpy as np
from joblib import Parallel, delayed
#importing torch libraries.
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import pyshearlab

#import our packages
from pyutils import dirutils, folderiter


def time_spent(tic1, tag='', count=1):
	toc1 = time.process_time() 
	print(f"time spend on {tag} method = {(toc1 - tic1)/count:.2f}s")
	return

def onceInit(shearlet_spec, kCUDA=False):
	"""
		shearlet_spec: 	
			'rows': 		256, 
			'cols': 		256, 
			'scales_per_octave': 2, 
			'shear_level': 	3, 
			'octaves': 		1, 
			'alpha': 		.8,		# Anisotropy level
	"""
	#1: init pytorch to run on the device of choice
	device = torch.device('cuda' if kCUDA else 'cpu')

	sh_sys = initShearlets(shearlet_spec, kLogging=True)
	return sh_sys, shearlet_spec

def initShearlets(shearlet_spec, kLogging=True):
	"""	Getting Shearlet Coefficients and saving them to the disc.	"""
	print(f"shearlet_spec {shearlet_spec}")
	#getting the 2 scale shearlet system.
	#SLgetShearletSystem2D(useGPU, rows, cols, nScales, shearLevels=None, full=0, directionalFilter=None, quadratureMirrorFilter=None):
	sh_sys = pyshearlab.SLgetShearletSystem2D(**shearlet_spec)	#useGPU=1, rows=92, cols=92, nScales=2)
	if kLogging:
		print(f"sh_sys.keys() {sh_sys.keys()}")
	return sh_sys

def shearletxform(
	sh_sys,
	d 			#tuple for the image
):
	""" apply shearlet xform to torch tensor 'd' """
	input, label = d
	X = pyshearlab.SLsheardec2D(input.numpy(), sh_sys)
	#(0 = rows, 1 = columns, 2 = no. of channels).
	# so X (92{0}{rows},92{1}{columns},17{2}{channels}) =(transpose)=> X (17{2}{channels},92{0}{rows},92{1}{columns}).
	#this transpose is done because it is requirement of pytorch input n/w. to have the first dimension of input as channels.
	X = np.transpose(X, (2, 0, 1))
	return X

def shearletxform_np(
	sh_sys,
	d 			#tuple for the image
):
	""" apply shearlet xform to ndarray 'd' """
	#TODO: assert on type(d) being ndarray
	input, label = d
	X = pyshearlab.SLsheardec2D(input, sh_sys)
	#(0 = rows, 1 = columns, 2 = no. of channels).
	# so X (92{0}{rows},92{1}{columns},17{2}{channels}) =(transpose)=> X (17{2}{channels},92{0}{rows},92{1}{columns}).
	#this transpose is done because it is requirement of pytorch input n/w. to have the first dimension of input as channels.
	X = np.transpose(X, (2, 0, 1))
	return X

def shearlet_inv(sh_sys, shcoeffs):
	""" inverse Shearlet xform """

	#(0 = channels, 1 = rows, 2 = cols.).
	# so X (17{0}{channels},92{1}{rows},17{2}{columns}) =(transpose)=> X (17{1}{rows},92{2}{columns},92{2}{channels}).
	#so as to convert from torch req. format as mentioned while transforming to correct format for visualization.
	recon_img = pyshearlab.SLshearrec2D(np.transpose(shcoeffs, (1, 2, 0)), sh_sys)
	return recon_img

def xformdataset_serial(
	sh_sys, 
	dataset, 
	shfilename='output/shearlets.pic', 
	kPickle=False,
	n_jobs=1	
):
	""" apply the Shearlet xform defined by 'sh_sys' to 'dataset' """
	print(f"xformdataset_serial '{shfilename}'")
	#dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers = 0)

	shearlets = []
	count = 1
	tic1 = time.process_time()
	print(f" SLsheardec2D using {n_jobs} cores")

	for i, d in enumerate(dataset):
		input, label = d
		#os.chdir(classes[label.item()])

		#(0 = rows, 1 = columns, 2 = no. of channels).
		# so X (92{0}{rows},92{1}{columns},17{2}{channels}) =(transpose)=> X (17{2}{channels},92{0}{rows},92{1}{columns}).
		#done because it is requirement of pytorch input n/w. to have the first dimension of input as channels.
		X = pyshearlab.SLsheardec2D(input.numpy(), sh_sys)
		X = np.transpose(X, (2, 0, 1))
		#np.save(str(i), X)
		if (i%100 == 0):
			print(i)
		shearlets.append(X)

	time_spent(tic1, "SLsheardec2D()", count=1)
	#time spend on SLsheardec2D() method = 17471.88ms

	if kPickle and (os.path.isfile(outfolder + shfilename)):
		with open(outfolder + shfilename, 'rb') as f:
			prev = pickle.load(f)
			assert(np.array_equal(prev, shearlets))
	print(type(shearlets), len(shearlets))
	return shearlets

def xformdataset(
	sh_sys, 
	dataset, 
	n_jobs=4,
	shfilename='output/shearlets.pic', 
	kPickle=False,
):
	""" apply the Shearlet xform defined by 'sh_sys' to 'dataset' """
	print(f"xformdataset '{shfilename}'")
	#dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers = 0)

	shearlets = []
	count = 1
	tic1 = time.process_time()

	print(f" SLsheardec2D using {n_jobs} cores")
	kSerial = (n_jobs == 1)		#serial|parallel Shearlet Xform

	#TODO: a better MP code is in mpxform.py: it is 13.4s vs 30s for 'test' by using a 
	# smarter MP parallelization instead of just using 'joblib.Parallel'

	for i, d in enumerate(dataset):
		input, label = d
		#ourlabel = label
		input = input.numpy()
		print(f" input.shape {input.shape}")
		break

	shearlets = Parallel(n_jobs=n_jobs, verbose=0, pre_dispatch='1.0*n_jobs')(
		delayed(shearletxform)(sh_sys, d) for d in dataset
	)
	time_spent(tic1, "SLsheardec2D()", count=1)
	#4:   3345ms(1.0),  3281ms(1.1), 
	#6:   3234ms(1.0),  3250ms(1.1), 
	#8:   4496ms(1.0),  3348ms(1.1), 
	#18: 19368ms(1.0), 19968ms(1.5), 
	if kPickle:
		outfolder = os.path.dirname(shfilename)
		dirutils.mkdir(outfolder)
		with open(shfilename, 'wb') as f:
			pickle.dump(shearlets, f)

	print(type(shearlets), len(shearlets))

	return shearlets

if __name__ == '__main__':
	from shnetutil import loadFashion
	parser = argparse.ArgumentParser(description='shearletxform.py')
	parser.add_argument('-dataset', type=str, default='test', help='test or train')
	args = parser.parse_args()

	root = '../data/'
	outfolder = '../output/'
	dirutils.mkdir(outfolder)

	kUseTrain = (args.dataset == 'train')
	print(f"using dataset: {'train' if kUseTrain else 'test'}")

	if kUseTrain:
		shfilename = outfolder + 'shearlets-full.pic'
		mergedfile = outfolder + 'train-set.dat'
	else:
		shfilename = outfolder + 'shearlets-10k.pic'
		mergedfile = outfolder + 'test-set.dat'

	#1: request the test set (10k)
	device, dataset = loadFashion.onceInit(root, istrain=kUseTrain, kCUDA=False)
	print(dataset, type(dataset), {type(dataset[0])})

	sh_sys, shearlet_spec = onceInit(loadFashion.kShearlet_spec, kCUDA=False)

	shearlets = xformdataset(sh_sys, dataset, n_jobs=4)
	print(f"shearlets: {len(shearlets)}, {shearlets[0].shape}")
