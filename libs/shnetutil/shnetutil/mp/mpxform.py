# -*- coding: utf-8 -*-
"""

Title: Use MP to perform our data xform/preconditioning.
	
Created on Mon Mar 16 17:44:29 2020

@author: Manny Ko & Ujjawal.K.Panchal

"""

import functools, pickle
import logging
import multiprocessing
from queue import Empty, Full
import numpy as np
import signal
import os, sys, time

#our packages
from ..dataset import dataset_base
from .. import pysh_xform, shearletxform, torchutils
#our modules

#
# https://pymotw.com/3/multiprocessing/communication.html
# https://wiki.python.org/moin/ParallelProcessing
# https://pymotw.com/2/multiprocessing/communication.html

MAX_SLEEP_SECS = 0.02

def time_spent(tic1, tag='', count=1):
	toc1 = time.process_time() 
	print(f"time spend on {tag} method = {(toc1 - tic1)/count:.2f}s")
	return


def xformdataset_serial(
	shfactory,
	device,	
	dataset,
	work		#(start, end)
):
	""" apply the Shearlet xform defined by 'sh_sys' to 'dataset' """
	print(f"xformdataset_serial {work}..")

	sh_spec, xform_factory = shfactory
	sh_xform = xform_factory(sh_spec)
	sh_xform.start(device)

	sh_sys = sh_xform.shearletSystem
	shearlet_spec = sh_xform.sh_spec

	shearlets = []
	count = 1
	tic1 = time.process_time()
	print(f" {sh_xform} using {1} cores")
	
	start, end = work
	for i in range(0, end-start):
		item = dataset[i]
		#print(f"item {item}")
		img, label = item

		coeffs = sh_xform.xform(item)
		shearlets.append(coeffs)
		#labels.append(label)

	time_spent(tic1, "", count=1)
	print(type(shearlets), len(shearlets))

	return shearlets


def xformdataset(
	dataset,
	shfactory,
	shfilenames, 
	worker,
	workerargs,
	n_jobs=10,
	kLogging=False
):
	""" apply the Shearlet xform defined by 'sh_spec' to 'dataset' """
	sh_spec = shfactory[0]
	print(f"mpxform.xformdataset '{shfilenames}', {sh_spec}", flush=True)

	t = time.time()
	numentries = len(dataset)
	jobs = []
	kNumJobs = n_jobs	#3: 18.5s, 4: 16.1s, 5: 24.8s
	chunk = int((numentries + kNumJobs-1)//kNumJobs)
	pipe_list = []

	queue = multiprocessing.Queue()

	for c in range(kNumJobs):
		queue.put((int(c*chunk), min(int((c+1)*chunk), numentries)))

	#https://stackoverflow.com/questions/10415028/how-can-i-recover-the-return-value-of-a-function-passed-to-multiprocessing-proce/36457960
	for i in range(kNumJobs):
		recv_end, send_end = multiprocessing.Pipe(False)

		p = multiprocessing.Process(target=worker, 
			args=(i, queue, send_end, shfactory, workerargs)
		)
		jobs.append(p)
		pipe_list.append(recv_end)
		p.start()

	# Wait for the worker to finish
	queue.close()
	queue.join_thread()

	for j in jobs:
		j.join()
		if kLogging:
			print('{:>15}.exitcode = {}'.format(j.name, j.exitcode))

	shfiles = [x.recv() for x in pipe_list]
	shfiles = sorted(shfiles, key=lambda entry: entry[0])
	print(f"shfiles(piped): {shfiles}")
	print(f"Elapsed time: xformdataset() {time.time()-t:2f}s")

	if shfilenames:
		with open(shfilenames, "wb") as f:
			pickle.dump(shfiles, f)
		return shfiles


if __name__ == '__main__':
	# for our unit test only (use the established Fashion dataset)
	import shnetutil.dataset.fashion

	def prepFashion(	
		jobid,
		shfactory,	#(sh_spec, xform_obj(subclass of ShXform()))
		workerargs,
	):
		""" the once init functor for each worker in mpxform for Fashion """
		print("prepFashion")
		root = workerargs['root'] 
		istrain = workerargs['istrain'] 
		trset = "train" if istrain else "test"

		device = torchutils.onceInit(kCUDA=kCUDA)

		#1: load our Fashion dataset
		train, test, validate = fashion.load_fashion(trset=trset, kTensor=False, validate=0.2)
		print(type(train), {type(train[0])}, {len(train)})

		return device, train

	root = '../data/'
	outfolder = '../output/'
	#shfilename = outfolder + 'shearlets-10k.pic'
	mergedfile = outfolder + 'test-set.dat'
	#TODO: remove the following when done testing
	mergedfile = outfolder + 'test-set.dat'

	kUseTrain = False

	#1: request the test set (10k)
	#device, dataset = loadFashion.onceInit(root, istrain=kUseTrain, kCUDA=False)
	dataset = loadFashion.getdb(root, istrain=kUseTrain)
	print(type(dataset), {type(dataset[0])}, len(dataset))

	sh_spec = loadFashion.kShearlet_spec
	shfilenames = outfolder+'shfiles.dat'

	#1: package up our xform factory
	shfactory = (sh_spec, pysh_xform.PyShXform)
	workerargs = {
		'istrain'	 : kUseTrain,
		'outfolder'	 : outfolder,
		'root'		 : root,	 
		'onceInit'	 : prepFashion		#per-worker once only init  
	}	
	shfiles = xformdataset(
		dataset,		#a CustomDataset
		shfactory,		#(sh_spec, pysh_xform.PyShXform) 
		shfilenames,	#outfolder+'shfiles.dat'
		worker,			#worker function
		workerargs,		#args for the worker
		n_jobs=10,
	)

	for entry in shfiles:
		num, file, *_ = entry
		print(f" shfilename '{file}", end='')
		with open(file, "rb") as f:
			coeffs = np.load(f, allow_pickle=False)		
			print(coeffs.shape)
