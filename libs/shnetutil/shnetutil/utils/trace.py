# -*- coding: utf-8 -*-
"""
Title: Context-Manager to support tracing PyTorch execution

@author: Manny Ko & Ujjawal.K.Panchal
"""
import sys, pickle, hashlib, logging
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Callable, List, Tuple, Optional, Union
from inspect import getmembers

import matplotlib.pyplot as plt
import numpy as np
import torch

from pyutils import dirutils
from ..pipeline import logutils

#for visualization.
from ..cplx import utils as cplxutils
from ..cplx import visual as cplxvisual
from ..cplx import dispatcher

from . import torchutils

#TODO: move enumLayers to a more logical place - perhaps as part of CVnn_base
def enumLayers(
	model: torch.nn.Module,
	filter: Callable = lambda aname, layer: issubclass(type(layer), torch.nn.Module)
):
	""" Retrieve a list of nn.Modules inside 'model'
	"""
	#layerlist = set(layerlist) if layerlist else set()
	if getattr(model, 'seqdict', None):		#using nn.Sequential?
		for k, layer in model.seqdict.items():
			yield k, layer
	else:
		for member in getmembers(model):
			aname, attrib = member
			if filter(aname, attrib):
				yield aname, attrib		

class TraceContext():
	""" this is a Context-Manager to support tracing PyTorch execution by capturing
	"""
	def __init__(self, 
		mylogger: logging.Logger = None, 
		kCapture: bool = False, 
		picklename: str = None, 
		capture: int = 10
	): 
		self.data = {}
		self.enableCapture = kCapture
		self.logger = mylogger
		self.name = 'TraceContext' if mylogger is None else mylogger.name
		self.picklename = picklename
		self.capturecnt = capture	# number of log() calls to capture
		self._enable = (mylogger is not None) or kCapture
		self.count = 0

	def __enter__(self):
		self.count = 0
		return self

	def __exit__(self, exc_type, exc_value, exc_traceback):	
		#print(f" exit() self.data={self.data}")
		self.save()

	def __eq__(self, other):
		""" compare our trace buffer against 'other' """
		results = compareTrace(self.data, other.data)
		return all(results.values())

	@property
	def enable(self):
		return self._enable
		
	@enable.setter
	def enable(self, flag):	
		self._enable = flag

	def load(self, pickle_name):
		""" restore the trace buffer from the pickle file """
		with open(pickle_name, "rb") as handle:	
			self.data = pickle.load(handle)

	def save(self, pickle_name=None):
		""" save our trace buffer to a pickle file """
		if self.enableCapture:
			if pickle_name is None:
				logdir = 'logs/'
				dirutils.mkdir(logdir)
				defaultname = logdir + self.name + '.pkl'
				pickle_name = self.picklename if self.picklename is not None else defaultname
			with open(pickle_name, "wb") as handle:
				pickle.dump(self.data, handle)

	def countexpired(self):			
		return (self.count >= self.capturecnt) and (self.capturecnt != -1)

	def log(self, obj, tag=''):
		""" use logger to log 'obj' to log file and console. Also capture it in the
			trace buffer (dict)
		"""	
		if (not self.enable) or self.countexpired():
			return
		if type(obj) is torch.Tensor:
			obj = getnumpy(obj)
		if type(obj) is np.ndarray:
			np.set_printoptions(threshold=sys.maxsize)

		if self.logger:	
			self.logger.info(f"{tag} {obj}")

		if self.enableCapture:
			self.data.update({tag: obj})
		self.count += 1	

	def logstr(self, str):		
		if self.logger:	
			self.logger.info(str)


def compare1(val1, val2):
	""" compare two entries in the trace buffer """
	if (type(val1) is np.ndarray):
		if val1.shape != val2.shape:
			return False
		result = np.allclose(val1, val2)
	else:
		if (type(val1) == tuple):
			return compareTuple(val1, val2)
		else:	
			result = (val1 == val2)
	return result	

def compareTuple(tuple1, tuple2):
	result = True
	for i in range(len(tuple1)):
		if (type(tuple1[i]) == torch.Tensor):
			result &= torch.all(torch.isclose(tuple1[i], tuple2[i])).item()
		else:
			result &= compare1(tuple1[i], tuple2[i])
	return result

def decompose(x: np.float32): 
	"""decomposes a float32 into negative, exponent, and significand"""
	negative = x < 0
	n = np.abs(x).view(np.int32) # discard sign (MSB now 0),
								 # view bit string as int32
	exponent = (n >> 23) - 127 # drop significand, correct exponent offset
							   # 23 and 127 are specific to float32
	significand = n & np.int32(2**23 - 1) # second factor provides mask
										  # to extract significand
	return (negative, exponent, significand)
	
def dumpmantissa(fval):
	_, _, significand = decompose(fval)
	print(f"0x{significand:06x}", end=',')
	
def npCompareDetailed(val1, val2, dumpthreshold=1e-5):
	NANinval1 = None in val1
	NANinval2 = None in val2

	print(f"Is None present in val1?: {NANinval1}")
	print(f"Is None present in val2?: {NANinval2}")

	e8result = result = np.allclose(val1, val2)
	e5result = np.allclose(val1, val2, atol = 1e-05)
	e3result = np.allclose(val1, val2, atol = 1e-03)
	e1result = np.allclose(val1, val2, atol = 1e-01)

	print(f"Equality for abs. tol. = 10^-8: {e8result}")
	print(f"Equality for abs. tol. = 10^-5: {e5result}")
	print(f"Equality for abs. tol. = 10^-3: {e3result}")
	print(f"Equality for abs. tol. = 10^-1: {e1result}")

	print(f"Shape equality: {val1.shape == val2.shape}")

	print("How many values close to each other:")

	n1 = val1.flatten()
	n2 = val2.flatten()

	e8count =  e5count = e3count = e1count = 0 
	print_count = 0
	ended = False
	for i in range(n1.shape[0]):
		e8count += np.isclose(n1[i], n2[i])
		e5count += np.isclose(n1[i], n2[i], atol = 1e-5)
		e3count += np.isclose(n1[i], n2[i], atol = 1e-3)
		e1count += np.isclose(n1[i], n2[i], atol = 1e-1)
		
		if not np.isclose(n1[i], n2[i], atol = dumpthreshold):
			print(f"({n1[i]:5f},{n2[i]:5f})", end = ';')
			dumpmantissa(n1[i])
			dumpmantissa(n2[i])
			print_count +=1
			ended = False
		if print_count % 2 == 0 and ended == False:
			print('')
			ended = True

	print(f"with abs. tol. = 10^-8: {e8count} / {n1.shape[0]}")
	print(f"with abs. tol. = 10^-5: {e5count} / {n1.shape[0]}")
	print(f"with abs. tol. = 10^-3: {e3count} / {n1.shape[0]}")
	print(f"with abs. tol. = 10^-1: {e1count} / {n1.shape[0]}")
	return result


def compareTrace(trace1, trace2, details={}):
	""" compare 2 trace buffers and return the results in a dict() """
	result = {}
	for key, val1 in trace1.items():
		val2 = trace2.get(key, None)
		if key in details:
			print(f"===(Compare:'{key}')===")
			result.update({key: npCompareDetailed(val1, val2)})
			print("===(Compare complete:'pool(F.relu(conv3(x))')===")
		else:
			result.update({key: compare1(val1, val2)})
	return result

class NullTraceMixin():
	def _init__(self):
		self.logger = None

	""" a No-op replacement for TraceMixin """
	def settrace(self, trace: TraceContext, **kwargs):
		pass	
	def qHookNeeded(self):
		return False
	def log_initweights(self):
		pass
	def log(self, obj, tag):
		pass
	def logstr(self, str):
		pass
	def log_grads(self, weight, str):
		pass

class TraceMixin():
	""" See CVnn.py's imported networks for example on how to use this mixin to give trace/loggging/capture to a NN """

	def __init__(self, enableObjs = False, enableGrads = False, enableStrs = False, enableImgs = False):
		""" this constructor usually is not called unless derived class explicitly do it """
		"""
		"""
		self.settrace(trace=None, 
			traceobjs=enableObjs, tracestrs=enableStrs, tracegrads=enableGrads, traceImgs=enableImgs)
		self._responseTracers = []
		self.remaining_pics = 5
		self.layerLogMap = {"linear": self.lin_logger, "conv": self.conv_logger}
		self._modelName = ""
		return

	def settrace(self, 
		trace: TraceContext, 
		traceobjs = True, tracestrs = True, tracegrads = False, traceImgs = False,
		loi: list = [],
	):
		if trace:
			assert(issubclass(type(trace), TraceContext))
		self.trace = trace	
		self.enableObjs = traceobjs
		self.enableStrs = tracestrs
		self.enableGrads = tracegrads
		self.enableImgs = traceImgs
		self.setloi(loi)
		return

	def setloi(self, loi: list):
		""" Set the list of layers we are interested in capturing/inspecting """
		self.loi = loi
		#print(f"loi {loi}")

	def qHookNeeded(self):
		""" Predicate for whether forward_hook is needed """
		return self.trace and self.trace.enableCapture

	def register_hook(self, module: torch.nn.Module, hook: Callable):
		""" Conditionally register 'hook' if it will be used """
		assert(issubclass(type(module), torch.nn.Module))
		if self.qHookNeeded():
			print(f"register_hook: {torchutils.modelName(module)}")
			module.register_forward_hook(hook)	

	def register_all_hooks(self, 
		loi: List[str], 
		hook: Callable,
		klogging = False
	):
		assert(type(loi) == list)
		self.setloi(loi)	
		if klogging: print(f"register_all_hooks loi {self.loi}")
		layers = enumLayers(self)
		for name, layer in layers:
			if name in self.loi:
				self.register_hook(layer, hook)
			else:
				if klogging: print(f"skipped: {name}")
					
	def lin_logger(self, x, string, layer, logInput = False):
		if logInput:
			self.log(x[0, 0:5], f"input to {string}")

		x = layer(x)
		
		self.log(x[0, 0:5], string)
		return x


	def pic_logger(self, x, title, path, training = False):
		"""
		take pictures. depends on remaining_pics.
		"""
		if not training and self.remaining_pics > 0:
			self.log_cplximg(x, title = title, path = path)
			self.remaining_pics -= 1
		return


	def conv_logger(self, x, string, layer, logInput = False):
		#1. log input is required.
		if logInput:
			self.pic_logger(x, f"input(x) for {string}", f"logs/input2{string.replace(' ', '_')}.png", layer.training)
			self.log(x.view(-1)[:5], string)

		#2. forward input through layer.
		x = layer(x)

		#3. log output of layer.
		self.pic_logger(x, string, f"logs/inpu2{string.replace(' ','_')}.png", layer.training)
		self.log(x.view(-1)[:5], string)
		return x

	def log_implement(self, x, layer, string, layertype = "linear", logInput = False):
		"""
		function to implement a lyer with logging.
		(Now deprecated. Should be replaced by self.module.register_forward_hook(self.log())  or some variant.)
		---
		args:
			1. x (python object input to layer): the input x on which layer is to be implemented.
			2. layer (torch.nn.Module/Similar.): the layer which will be implemented.
			3. string (str): the string to be associated to the log.
			4. layertype: (str: "linear" | "conv"). due to difference in logging.
			5. logInput: (bool: True|False) if to log the input.
		"""
		x = self.layerLogMap[layertype](x,f"{self._modelName} {string}", layer, logInput) 
		return x

	def log_initweights(self):
		params = list(self.parameters())	#TODO: do we need to convert to list()?
		initPs = params[0].data.flatten()

		for i in range(1, len(params)):
			initPs = torch.cat((initPs, params[i].data.flatten()))
		self.trace.log(initPs, "init parameters")

	def log_cplximg(self, *args, **kwargs):
		if self.trace and self.enableImgs:
			self.trace.takeComplexPic(
				*args, **kwargs
			)

	def log(self, obj, tag):
		if self.trace and self.enableObjs:
			self.trace.log(obj, tag)

	def logstr(self, str):
		""" log a pure string """
		if self.trace and self.enableStrs:
			self.trace.logstr(str)

	def log_grad(self, weight, str):
		if self.trace and self.enableGrads:
			self.log(weight.grad, str)

	def log_hook(self, module, input, output):
		self.log(
				dispatcher.get_histogram(output, bins = 10),
				dispatcher.getModelConfigStr(module) + f"_run_{str(self.forward_count)}"
		)
		return

	def log_input_hook(self, module, input, output):
		self.log(
				dispatcher.get_histogram(input[0], bins = 10),
				f"input_run_{self.forward_count}"
		)
		return

	def makeResponseTrace(self, tracectx, capturelimit=10, name="default"):
		response = ResponseTracer(tracectx, capturelimit, name)			
		self._responseTracers.append(response)
		return response

	@property
	def responseTracers(self):
		return self._responseTracers

def processResponseTracer(model):
	""" response tracing completion. """
	for response in getattr(model, 'responseTracers', None):
		response.doAll()


def enable_console(tracectx, formatter=logutils.kDefFormatter, level=logging.INFO):
	if tracectx:
		logutils.enable_console(tracectx.logger, formatter, level)				

def disable_console(tracectx):
	if tracectx:
		logutils.disable_console(tracectx.logger)				

def getnumpy(obj):
	if (type(obj) == torch.Tensor) or (type(obj) == torch.nn.parameter.Parameter):
		obj = obj.cpu().detach().numpy()
	else:
		#print(f"getnumpy type(obj) {type(obj)}")
		obj = np.asarray(obj)	
	return obj	

def chksum_numpy(sha, obj):
	#1: get the ndarray into a single continuous C-struct	
	datastream = BytesIO()
	np.save(datastream, obj, allow_pickle = True)	#TDOD: reduce numpy version dependency here
	sha.update(datastream.getvalue())

def chksum_torch(sha, obj):
	obj = getnumpy(obj)
	chksum_numpy(sha, elem)

def chksum_tuple(sha, obj):
	for elem in obj:
		elem = getnumpy(elem)
		chksum_numpy(sha, elem)

def dochecksum(obj):	
	sha = hashlib.sha256()
	dispatch = {
		torch.Tensor: 	chksum_torch,
		tuple: 			chksum_tuple,
		list: 			chksum_tuple,
		np.ndarray:		chksum_numpy,
	}
	dispatch[type(obj)](sha, obj)	#TODO: handle type not in 'dispatch'

	checksum = sha.hexdigest()
	return checksum

class TraceTorch(TraceContext):
	""" Specialization of TraceContext to log PyTorch tensors as hashes """
	def __init__(self, 
		mylogger=None,		#logger object from 'logging' (or logutils.py) 
		kCapture=False,		#enable capturing 
		picklename=None,	#.pkl file for the captured data 
		capture=10, 		#number of .log() calls to capture/checksum
		kCheckSum=False,		#checksum the objects getting logged instead of its raw form.
		cplxpicsaver = cplxvisual.saveComplexImage, #save image.
		complex_type = "numpy", #Takes values Trabelsi. Type of the network for which it is used.
		image_consolidation = np.mean
	): 
		super().__init__(mylogger, kCapture, picklename, capture)
		self.checksum = kCheckSum

		self.cplxpicsaver = cplxpicsaver
		self.complex_type = complex_type
		self.image_consolidation = image_consolidation
		self.set_cplx_preprocess_()
		return

	def set_cplx_preprocess_(self):
		preprocess = None

		if self.complex_type.lower() == "numpy":
			def make_npcplx(npcplx):
				return npcplx

		elif self.complex_type.lower() == "trabelsi":
			def make_npcplx(x):
				npcplx = cplxutils.cplx2numpy(x)
				return npcplx
		else:
			raise Error("TraceTorch: Unsupported complex_type.")

		def preprocess(x):
			npcplx = make_npcplx(x)
			npcplx = np.transpose(npcplx, (1,2,0))
			npcplx = self.image_consolidation(npcplx, axis = 2)
			return npcplx
		
		self.cplx_preprocess = preprocess
		return

	def takeComplexPic(
		self, x,
		title = "complex Image", 
		path = "./cplximg.png",
		figsize = (6,6)
		):
		np_cplx = self.cplx_preprocess(x)

		self.cplxpicsaver(np_cplx, path = path, title = title, figsize = figsize)
		return

	def __eq__(self, other):
		""" compare our trace buffer against 'other' """
		results = compareTrace(self.data, other.data)
		return all(results.values())

	def log(self, obj, tag=''):
		""" use logger to log 'obj' to log file and console. Also capture it in the
			trace buffer (dict)
		"""
		if (self.count < self.capturecnt):	#to save the call to dochecksum() if we are going to ignore it
			if self.checksum:
				obj = dochecksum(obj)
			super().log(obj, tag)

class Fisher(ABC):
	"""
	A Fisherman catches a particular type of fish (gradient) and 
	notes it's details in his Notebook.

	An array of fisherman can catch different types of fish and
	log their details in a common notebook. 
	"""
	def __init__(self, 
		fishName: str, 
		tracer: TraceContext 
	):
		self.fishName = fishName
		self.fishNote = tracer
		assert(isinstance(tracer, TraceContext))

	@abstractmethod
	def catch(self, fish):
		pass	

	def finalize(self):
		pass

class GradFisher(Fisher):
	"""
	A Fisherman catches a particular type of fish (gradient) and 
	notes it's details in his Notebook.

	An array of fisherman can catch different types of fish and
	log their details in a common notebook. 
	"""
	def __init__(self, 
		fishName: str, 
		tracer: TraceContext, 
		count:int =10
	):
		super().__init__(fishName, tracer)
		self.iceBucket = []
		self.count = count
		self.hooks = []

	def __del__(self):	
		self.emptyBucket()
		self.release()

	def catch(self, fish):
		#TODO: Proper checksum.
		#till then, log only first .count grads.
		if len(self.iceBucket) < self.count:
			grad = getnumpy(fish.reshape(1,-1)[0, :5])
			self.iceBucket.append(grad)

	def finalize(self):
		self.fishNote.log(self.iceBucket, self.fishName)
		self.emptyBucket()

	def emptyBucket(self):
		for h in self.hooks:
			h.remove()
		self.iceBucket = []

	def release(self):
		self.fishName = "Forced Empty"
		if (len(self.iceBucket) != 0):
			self.finalize()
		self.fishNote = None #release reference to tracer to be safe.

	def install_hook(self, weights):
		self.hooks = install_hook(weights, self.catch)

#gradient tracing code.
def requires_grad(weights):
	result = True
	if type(weights) is tuple:
		for w in weights:
			result &= hasattr(w, 'requires_grad')
		return result			
	else:
		return hasattr(weights, 'requires_grad')	
	
def install_hook(weights, hook):
	hooks = []
	if type(weights) is tuple:
		for w in weights:
			h = w.register_hook(hook)
			hooks.append(h)
	else:		
		h = weights.register_hook(hook)
		hooks.append(h)
	return hooks

def gradhook(model, layers, tracer, Catcher: Fisher = GradFisher, count=5):	
	layers = torchutils.getModulesByName(model, layers)
	catchers = []
	for layer in layers:
		name, module = layer
		weights = model.getweights(module)
		#print(f" {name}.requires_grad: {requires_grad(weights)}")
		if requires_grad(weights):
			gradcatcher = Catcher(name + " grad", tracer, count=count)
			gradcatcher.install_hook(weights)
			catchers.append(gradcatcher)
	return catchers

def freegradHooks(catchers):
	#free gradient tracers.
	if not catchers:
		return
	for gt in catchers:
		gt.finalize()
		gt.release()

def breakdownWeights(weights):
	weightList = []
	if not torch.is_tensor(weights):
		weightList = [getnumpy(w).flatten() for w in weights]
	else:
		ws = getnumpy(weights).flatten()
		weightList.append(ws)
	return weightList

def logWeights(model, layers, tracer, bins = 5, id = "initweights_"):
	layers = torchutils.getModulesByName(model, layers)
	for name, module in layers:
		weightList = breakdownWeights(model.getweights(module))
		for i, weights in enumerate(weightList):
			wstats = (np.mean(weights), np.min(weights), np.max(weights))

			#plot histogram.
			plt.hist(weights, bins = bins, range=[wstats[1], wstats[2]])
			plt.xlabel("Bins")
			plt.ylabel("Frequency")
			plt.title(f"Layer: {name}")
			plt.savefig(f"{id}{name}_{i}.png")
			plt.clf()
			#log weights.
			tracer.log(wstats, f"{id}{name}_{i}'s mean, min, max:")

class ResponseTracer():
	def __init__(self, tracectx, batchlimit = 10, name = "default"):
		self.batchlimit = batchlimit
		self.data = []
		self.trace = tracectx
		self.averaged = False
		self.name = name

	def addBatchResp(self, batch):
		if (len(self.data) < self.batchlimit):
			self.data.append(breakdownWeights(batch))

	def avgData(self):
		self.avgData = np.sum(self.data) / len(self.data)
		self.averaged = True
		return  self.avgData#averaged across batches.
	
	def makeHistogram(self, bins = 10, fn = "logs/ResponseHist"):
		assert (self.averaged)
		plt.hist(self.avgData, bins)
		plt.xlabel("Bins")
		plt.ylabel("Frequency")
		plt.title(f"Response Name: {self.name}")
		plt.savefig(f"{fn}_{self.name}.png")
		return
	
	def wStats(self):
		assert (self.averaged)
		return np.mean(self.avgData), np.min(self.avgData), np.max(self.avgData)
		
	def logResponse(self):
		wstats = self.wStats()
		self.trace.log(wstats, f"{self.name} response (mean, min, max): ")

	def doAll(self):
		self.avgData()
		self.makeHistogram()
		self.logResponse()

class IndexRecorder(TraceContext):
	"""
	---
	Record interesting *indices and tensors* from a 
	*prediction run* for a particular *model*.
	---
	1. logSet: (batchIndex, itemIndex, factLabel, predTensor):
   		  	1. batchIndex: Index of the batch that sample belongs.
   		  	2. itemIndex: item index in that particular batch.
   		  	3. factLabel: actual label of the item.
   		  	4. predTensor: predicted non-softmaxed tensor.
	"""
	def __init__(
			self,
			model_id = "default", 
			set_id = "train",
			id = "",
			kCapture = True
			):
		super().__init__(kCapture=kCapture, picklename=f"logs/{model_id}_{set_id}_{id}.pkl")
		self.data = []
		
	def log(self, batchIndex, itemIndex, factLabel, predTensor):
		if self.enable:
			self.data.append((batchIndex, itemIndex, factLabel, predTensor))



