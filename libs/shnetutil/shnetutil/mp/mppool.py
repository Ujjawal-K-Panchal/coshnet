# -*- coding: utf-8 -*-
"""

Title: Shearlet based CNN vs. simple MLP in Fashion MNIST.
	
Created on Mon Mar 16 17:44:29 2020

@author: Manny Ko & Ujjawal.K.Panchal 

"""
#import logging
import abc
from typing import List, Tuple, Union, Optional
import multiprocessing
#import asyncio 
from queue import Empty, Full
#import signal
import os, sys, time

#our packages
#our modules

kUnitTest=False

#
# Python Pool.map() in theory solves the same problem as MPPool(). The reason we did not adopt:
#   1. no concept of per-core/worker onceInit()
#	2. work has to be marshalled across processes:
#	   work = (["A", 5], ["B", 2], ["C", 1], ["D", 3]) 		#our giant dataset
#	   p = Pool(2)
#	   p.map(worker, work)		#'worker' and 'work' are marshalled 
#
# https://pymotw.com/2/multiprocessing/communication.html

def get_chunk(numentries:int, numjobs:int):
	chunk = int((numentries + numjobs-1)/numjobs)
	return chunk

class MPPool(metaclass=abc.ABCMeta):
	""" A pool of processes to run async workers with load balancing 
		- it behaves like persistent-threads where 
		  a) each core is occupied by one 'worker'
		  b) each worker pull work from a shared queue and are otherwise blocked.
		  c) each worker sends its result using a Pipe (lighter than a Queue)
		  d) doit() method perform load-balancing by dividing the work into work-items, using
		     the concept of 'chunkfactor' - which is the the factor we mulitply n_jobs with to
		     divide the input into chunks. The chunks are pushed onto the queue.
	"""
	kSTOP_VALUE = None 		#our poison-pill to stop workers when popped from queue

	def __init__(self, poolsize=4):
		self.poolsize = poolsize	#number of process in our tool
		self.pool = []
		self.jobs = []
		self.queue = None
		self.pipe_list = None
		self.send_list = None

	def __del__(self):
		print("MPPool.__del__")
		#if finalize() was called with kJoin all the processes should have terminated
		self.stopWorkers()

		for j in self.jobs:
			if j.is_alive():
				print(f"join {j}")	
				j.join()		

	@property
	def qInit(self):
		""" predicate for onceInit has not been called """
		return self.pipe_list is None

	@abc.abstractmethod
	def makeWorker(self, 
		queue:multiprocessing.Queue, 
		send_end:multiprocessing.Pipe, 
		workerargs:dict
	) -> multiprocessing.Process:
		""" implemened by client code """
		pass		

	def onceInit(self, workerargs:dict):
		""" per-pool once only init """
		assert(type(workerargs) == dict)
		queue = multiprocessing.Queue()
		self.queue = queue

		if self.qInit:
			jobs = []
			pipe_list = []
			send_list = []
			for i in range(self.poolsize):
				recv_end, send_end = multiprocessing.Pipe(False)
				pipe_list.append(recv_end)
				send_list.append(send_end)
				#TODO: see if streams are better https://docs.python.org/3/library/asyncio-stream.html#register-an-open-socket-to-wait-for-data-using-streams
			
				p = self.makeWorker(queue, send_end, workerargs)
				assert(isinstance(p, multiprocessing.Process))

				jobs.append(p)
				p.start()		#start the worker - mostly will block waiting for work

			self.jobs = jobs
			self.pipe_list = pipe_list
			self.send_list = send_list

	def stopWorkers(self):
		for p in self.jobs:
			if p.is_alive():
				self.queue.put(MPPool.kSTOP_VALUE)		#poison-pill to tell workers to break out

	def scheduleWork(self, numentries:int, workerargs:dict, kPersist=False):	
		print(f" pid[{os.getpid()}] using {self.poolsize} cores to process {numentries}", flush=True)

		queue = self.queue
		self.numentries = numentries

		chunkfactor = workerargs.get('chunkfactor', 3)
		chunk = get_chunk(numentries, self.poolsize*chunkfactor)

		for c in range(0, numentries, chunk):
			start = c
			end = min(numentries, c+chunk)
			queue.put((start, end))
			print(f"chunk={(start, end)}, ", end='')
		print()

		#3: 'poison-pill' => no more work tell workers to quit
		if not kPersist:
			self.stopWorkers()

	def doit(self, workerargs:dict, numentries:int, kPersist:bool=False):
		self.onceInit(workerargs)
		self.scheduleWork(numentries, workerargs, kPersist)
		
		#4: Wait for the worker to finish and collect results from each worker
		self.results = self.finalize(kPersist)

	def finalize(self, kPersist:bool):
		""" Wait for the worker to finish and collect results from each worker """
		results = [x.recv() for x in self.pipe_list]

		self.queue.close()
		self.queue.join_thread()

		self.verify(results)

		#sort results to reconstruct original order
		tmplist = []
		for result in results:
			instid, output = result

			#2: loop through per-worker output list
			tmplist.extend(output)
		results = sorted(tmplist, key=lambda e: e[0][0])	

		if not kPersist:
			#TODO: try not to close the queue for persistent case, just need to test
			for p in self.jobs:
				p.join()
	
		return results

	def verify(self, results):
		""" placeholder for client code to do final verification - called from finalize() """
		pass

if (__name__ == '__main__') and kUnitTest:
	...

