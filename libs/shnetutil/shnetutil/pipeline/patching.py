# -*- coding: utf-8 -*-
"""
Title: Shearlet based CNN vs. simple MLP in Fashion MNIST.
	
Created on Sun Aug 4 17:44:29 2020

@author: Manny Ko & Ujjawal.K.Panchal


Problem: Closure requires local variables. These local variables are not picklable. Hence, we need to use dill. 
pathos uses dill. initializer arg. of mp pool as shown in line 110 allows sharing resources across procs. 
_ProcessPool of pathos consists of the same argument as multiprocessing.Pool. #TODO: remove all problems from comments.

When initializer is tested in a seperate script, (see pathos_test.py), it works. However, when the initializer is called
from this file from line 129, it doesn't run ("initializer called" never gets printed).

"""
#small python imports.
from typing import List, Callable, Iterable, Tuple, Optional
from functools import partial
from itertools import product

#heavier python imports.
import os, time
import numpy as np
from multiprocessing import Queue, Process, RawArray, Pool

#methods.
def find_patch(xy: Tuple[int, int], image: np.ndarray,  patch_size: int):
	x, y = xy
	patch = image[x:x + patch_size, y:y+patch_size,...]
	return patch

#doubt: why the outside tuple int?????? how does that work????
def find_patch_mp(patch_func: Callable[[Tuple[int, int]], int], 
				  in_queue: Queue, out_queue: Queue) -> None:
	while True:
		xy = in_queue.get()
		if xy is None:
			break
		x = patch_func(xy)
		out_queue.put(obj = x)

	return None


#MP POOL PATCHING.
axis_steps = lambda dimvals, patch_size, stride: (int((dimvals[0] - patch_size) / stride) + 1, int((dimvals[1] - patch_size) / stride) + 1)
get_xy_list = lambda x_steps, y_steps, stride: product(range(0, x_steps * stride, stride), range(0, y_steps * stride, stride))

var_dict = {}
def share_resources_init(buf, imgs_shape, dtype):
	var_dict['imgs_np'] = buf
	var_dict['imgs_np_shape'] = imgs_shape
	var_dict['imgs_np_dtype'] = dtype
	#print("initializer called.")
	return

def find_patches4img(index: int, patch_func: Callable, patch_size: int, stride: int):
	#get image from shared dset.
	img = np.frombuffer(var_dict['imgs_np'], dtype = var_dict['imgs_np_dtype']).reshape(var_dict['imgs_np_shape'])[index]
	
	#make xy list.
	x_steps, y_steps = axis_steps((img.shape[0], img.shape[1]), patch_size, stride)
	xy_list = get_xy_list(x_steps, y_steps, stride)

	#get and return patch_list.
	patch_list = [patch_func(xy, img, patch_size) for xy in xy_list]
	return (index, patch_list) 

def make_params4img(index: int, patch_func: Callable,  patch_size: int, stride: int):
	return (index, patch_func, patch_size, stride)

def get_mp_patches_byPool(dset: Iterable, patch_func: Callable, patch_size: int, 
						  stride: int, num_workers: Optional[int] = None, verbose = 0, param_strategy = 0):
	"""
	find and return patches from a given set of images.
	
	---
	Args:

		1. dset: iterable set of data points (elements stacked along 0th dim.).
		2. patch_func: a function used for patching images given starting x,y coordinates and patch size.
		3. patch_size: Int which determines size of patch.
		4. stride: amount of steps to go forward.
		5. num_workers (default = num cpus): Number of worker processes.
		6. verbose (int)(default = 0): in {0,1,2}.
		7. param_strategy (default = 0) parameter passing strategy. 0 = using partial.
	---
	TODO: Verbose Processing.

	"""
	#decide on number of workers.
	if num_workers is None:
		num_workers = os.cpu_count()


	#1. make sharable dset.
	imgs = np.array([img for img in dset]) #full dset.
	buf = RawArray(typecode_or_type = np.ctypeslib.as_ctypes_type(imgs.dtype),
				    size_or_initializer = imgs.size)
	imgs_np = np.frombuffer(buf, dtype = imgs.dtype).reshape(imgs.shape)
	np.copyto(dst = imgs_np, src = imgs)


	#Parameter passing strategy.
	if param_strategy == 0:
		#process operation wide static parameters.
		make_params = partial(make_params4img, patch_func = patch_func, stride = stride, patch_size = patch_size)
			#start pool.
		with Pool(num_workers, initializer = share_resources_init, initargs = (buf, imgs.shape, imgs.dtype)) as pool:
			
			#1. make_args.
			args = [make_params(index) for index in range(len(dset))]
			
			#2. Do async multiprocessing.
			op = pool.starmap_async(find_patches4img, args)

			#3. Sort the output.
			result = np.array([r for index, r in sorted(op.get())])

	else:
		raise Exception("Error<!>: Invalid parameter strategy. param_strategy \\in [0,1]")

	#return sorted results.
	return result

#MP POOL PATCHING END.		

def get_mp_patchesbyImgProcs(dset: Iterable, patch_func: Callable[[Tuple[int, int]], int], 
			   patch_size: int, stride: int, num_workers: Optional[int] = None, verbose: int = 0):
	"""
	find and return patches from a given set of images.
	
	---
	Args:

		1. dset: iterable set of data points (elements stacked along 0th dim.).
		2. patch_func: a function used for patching images given starting x,y coordinates and patch size.
		3. patch_size: Int which determines size of patch.
		4. stride: amount of steps to go forward.
		5. num_workers (default = num cpus): Number of worker processes.
		6. verbose (int)(default = 0): in {0,1,2}.
	"""
	#ensure number of processes.
	if num_workers is None:
		num_workers = os.cpu_count()
		if verbose:
			print(f"using default config, using : {num_workers} processes.")

	#queues for putting tasks and results.
	in_queue = Queue(maxsize = len(dset))
	out_queue = Queue(maxsize=-1)
	if verbose:
		print(f"using in_queue of size : {len(dset)} processes.")


	 #Sources ofr sharing numpy arrays across processes: (Not async since order matters).
        # 1. https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html
        # 2. https://stackoverflow.com/questions/33247262/the-corresponding-ctypes-type-of-a-numpy-dtype
        # 3. https://stackoverflow.com/questions/7894791/use-numpy-array-in-shared-memory-for-multiprocessing
	patches = []
	for index,img in enumerate(dset):
		t1 = time.time()
		#1. make sharable buffer, stylize buffer to shape and type, copy image to sharedholder. 
		
		buf = RawArray(typecode_or_type = np.ctypeslib.as_ctypes_type(dtype = img.dtype),
					   size_or_initializer = img.size)
		img_np = np.frombuffer(buffer = buf, dtype = img.dtype).reshape(img.shape)
		np.copyto(dst = img_np, src = img)
		
		if verbose:
			print(f"shared memory image loaded. (type, shape): ({type(img_np)},{img_np.shape}) ")


		#2. step parameters calculation.

		#2.1. number of x-axis steps.
		x_steps = int((img.shape[0] - patch_size) / stride) + 1
		if verbose:
			print(f"using xsteps : {x_steps}.")

		
		#2.2. number of y-axis steps.
		y_steps = int((img.shape[1] - patch_size) / stride) + 1
		if verbose:
			print(f"using ysteps : {y_steps}.")
		
		#2.3.  step size (same for both axes).
		step_size = stride
		if verbose:
			print(f"using step_size : {step_size}.")

		#3. Extract patches using multiprocess.

		#3.1. Make Processes.
		processes = [
						Process(
								target = find_patch_mp,
								args = (partial(patch_func, image = img_np, patch_size = patch_size), in_queue, out_queue)
								)
						for _ in range(num_workers)
					]
		if verbose:
			print(f"made process list of length : {len(processes)}.")
		#3.2. Start Processes as daemons.
		for i,p in enumerate(processes):
			p.daemon = True
			p.start()
			if verbose > 1:
				print(f"started process: {i}.")

		#3.3. Put (x,y) in the queue.
		for xy in product(range(0, x_steps * step_size, step_size),
						  range(0, y_steps * step_size, step_size)):
			in_queue.put(xy)
			if verbose > 1:
				print(f"put position: {xy} in in_queue.")

		#3.4. Terminate signal to running processes.
		for _ in range(num_workers):
			in_queue.put(None)
			if verbose > 1:
				print(f"putting None in in_queue.")

		img_patches = np.array([out_queue.get() for _ in range(x_steps * y_steps)])
		patches.append(img_patches)
		
		#3.5. Join Processes.
		for i,p in enumerate(processes):
			p.join(timeout = 1)
			if verbose > 1:
				print(f"joining process {i}.")
			
		if verbose:
			print(f"image {index} to {len(img_patches)} patches took {time.time() - t1} seconds")
	return np.array(patches)


#DEFAULT DEFINITIONS.
get_mp_patches = get_mp_patches_byPool


if __name__ == "__main__":
	
	#test imports.
	from PIL import Image
	import matplotlib.pyplot as plt

	#test folders: WARNING. Not pushed (DEL after tesing).
	test_img_folder = "test-imgs/"
	patch_folder = "patches/"

	if not os.path.exists(test_img_folder):
		raise Exception(f"test_img_folder: '{test_img_folder}' does not exist. First make it and fill it with some images.")

	if not os.path.exists(patch_folder):
		print(f"patch_folder: '{patch_fodler}' does not exist. Making it.")
		os.mkdir(patch_folder)

	#Getting images.
	imgs = []
	for img in os.listdir(test_img_folder):
		im = np.array(Image.open(f"{test_img_folder}" +img))
		imgs.append(im)
		#print(f"image shape: {im.shape}")


	TEST_BY_POOL = True
	TEST_PER_IMAGE_PROCS = True

	if TEST_BY_POOL:
		t1 = time.time()	
		imgPatchList = get_mp_patches_byPool(imgs, find_patch, 100, 80)
		t2 = time.time()
		print(f"dataset mp Pool patching took: {t2 - t1} seconds.")

	if TEST_PER_IMAGE_PROCS:
		t1 = time.time()
		#Getting patches.
		imgPatchList = get_mp_patchesbyImgProcs(imgs, find_patch, 100, 80, verbose = 0)
		t2 = time.time()
		print(f"Per image mp patching took: {t2 - t1} seconds.")
	
	#Saving patches.
	for i, imgPatches in enumerate(imgPatchList):
		for j, patch in enumerate(imgPatches):
			save_loc = f"{patch_folder}{i}-{j}.png"
			Image.fromarray(patch).save(save_loc)

			