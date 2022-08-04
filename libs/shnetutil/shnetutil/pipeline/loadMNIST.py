# -*- coding: utf-8 -*-
"""

Title: Shearlet based CNN vs. simple MLP in Fashion MNIST.

Created on Mon Mar 16 17:44:29 2020

@author: Ujjawal.K.Panchal & Manny Ko

"""
import os, gzip, struct
from collections import namedtuple
from pathlib import Path
import requests
import numpy as np
from pyutils import dirutils
from ..dataset import dataset_base

# Week1:
#   1. study Samples/testwget.py on ways to download from an URL.
#   2. download the famous MNIST dataset from: http://yann.lecun.com/exdb/mnist/
#   3. study the simple MNIST format and implement loadMNIST():
#	   - it is compressed using gzip. So we have to uncompressed it first
#	   - the uncompressed files are binary. Treat it carefully.
#	   - one way to decode the file is to use 'struct' but there are other ways
#
# Week2:
#   1.  built a CustomDataset from the 10k test images and labels.
#       You have to implement the 3 methods in a sensible way.
#

# ================================================================== #
#                5. Input pipline for custom dataset                 #
# ================================================================== #
ImageDesc = namedtuple("images_header", "magic count rows cols")
LabelDesc = namedtuple("labels_header", "magic count")

def imageHeader(f):
	""" read and decode the header for the image file """
	HDRFMT = ">iiii"
	buf = f.read(struct.calcsize(HDRFMT))		#0x0803, # of images, # or rows, # of cols
	#0x0803, # of images, # or rows, # of cols
	header = struct.unpack(HDRFMT, buf)
	images_header = ImageDesc(*header)
	return images_header

def labelHeader(f):
	""" read and decode the header for the labels file """
	buf = f.read(8)		#0x0803, # of images
	#0x0803, # of labels
	header = struct.unpack(">ii", buf)
	images_header = LabelDesc(*header)
	return images_header

def loadMNIST(imgfile, labelfile):
	""" load MNIST data set using 'int.from_bytes' """
	images = None
	labels = None

	with gzip.open(imgfile, "rb") as f:
		imgheader = imageHeader(f)
		#2: read the rest of the bytes which are the 28x28 pixels
		pixels = f.read()
	images = (imgheader, pixels)

	with gzip.open(labelfile, "rb") as f:
		labelheader = labelHeader(f)
		labels = f.read()
	labels = (labelheader, labels)

	return images, labels

def filesize(filename):
	statinfo = os.stat(filename)
	return statinfo.st_size

# ================================================================== #
#                5. Input pipline for custom dataset                 #
# ================================================================== #
#MNIST:
MNIST_site_url="http://yann.lecun.com/exdb/mnist/"
#FASHION:
#https://github.com/zalandoresearch/fashion-mnist
FASHION_site_url="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

dataset_names = [
	"train-images-idx3-ubyte.gz",
	"train-labels-idx1-ubyte.gz",
	"t10k-images-idx3-ubyte.gz",
	"t10k-labels-idx1-ubyte.gz"
]
image_size = 28
num_labels = 10


Classes = [
	'T-shirt-or-top',
	'Trouser',
	'Pullover',
	'Dress',
	'Coat',
	'Sandal',
	'Shirt',
	'Sneaker',
	'Bag',
	'Ankle boot'
]

def get_site_url(kFashion=True):
	site_url = FASHION_site_url if kFashion else MNIST_site_url 
	return site_url

def get_dataset_name(kFashion=True):
	names = ('FashionTrain', 'FashionTest') if kFashion else ('MNISTtrain', 'MNISTtest')
	return names
#
# You should build your custom dataset as below.
class FashionDataset(dataset_base.DataSet):		#torch.utils.data.Dataset
	kImageSize = 28		#remove these constants from here. They are already in dataset.fashion
	kMean = 0.5042909979820251	#TODO: document this
	kStd  = 1.1458107233047485	#TODO: document this
	#calculating with
	cplxmean = 26.08305905336539
	normdshmean = 0.17420224977235

	def __init__(self, imagefile, labelfile, name=None):
		# 1. Preprocess the data to convert the images and labels to efficient forms.
		name = Path(imagefile).parts[-1] if name is None else name
		super().__init__(name)

		self.images, self.labels = loadMNIST(imagefile, labelfile)
		rows = self.images[0].rows
		cols = self.images[0].cols
		self.stride = rows * cols
		self.size = int(len(self.images[1] )/self.stride)
		images = np.frombuffer(self.images[1], dtype=np.uint8)
		self.pixels = images.reshape((len(self), rows, cols))
		self._numlabels = np.max(self.get_labels()) + 1
		self.verify()

	def __getitem__(self, index) -> dataset_base.ImageDesc:
		# TODO
		# 1. Return a data pair (e.g. image and label).
		#print(f"type(np.frombuffer {type(self.pixels)} {self.pixels.shape}")
		#return self.pixels[offset:offset+self.stride].reshape(28, 28)
		#offset = index * self.stride
		return dataset_base.ImageDesc(self.pixels[index], self.labels[1][index])

	def __len__(self):
		# You should change 0 to the total size of your dataset.
		return self.size

	def get_labels(self):
		labels = np.frombuffer(self.labels[1], dtype=np.uint8)
		return labels

	@property
	def num_classes(self):
		return self._numlabels

	def get_full(self, kOneHot=False):
		""" return the full dataset in ndarray, labels one-hot encoded """
		pixels = self.pixels.reshape((-1, self.stride)).astype(np.float32)
		# Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...] - one-hot encoding
		y = self.get_labels()

		return pixels, y

	def verify(self):
		assert(self.images[0].magic == 0x803)
		assert(self.images[0].rows == FashionDataset.kImageSize)
		assert(self.images[0].cols == FashionDataset.kImageSize)

		assert(self.labels[0].magic == 0x801)
		assert(self.labels[0].count == self.images[0].count)
		#assert(len(self.labels[1]) == 10000)

def dumpInfo(onepair, imagefile, labelfile):
	images, labels = onepair

	imgheader, pixels = images
	magic, count, *_ = imgheader
	#print(f"{imagefile} magic={hex(magic)}, count={hex(count)}, size={len(pixels)}")

	labelheader, labels = labels
	magic, count = labelheader
	#print(f"{labelfile} magic={hex(magic)}, count={hex(count)}, size={len(labels)}")

def downloadMNIST(site_url="http://yann.lecun.com/exdb/mnist/", outfolder='input/'):
	dirutils.mkdir(outfolder)
	datasets_root = Path(outfolder)
	result = True

	for dataset_name in dataset_names:
		url = site_url + dataset_name

		filename = datasets_root / Path(dataset_name)

		if (not os.path.isfile(filename)):
			try:
				print(f"downloading: {url}..")
				response = requests.get(url)
			except:
				result = False
				print(f"URL error: {url}")

			if response.ok:
				with open(filename, "wb") as f:
					#f.write(response.text)
					f.write(response.content)
				print(f" file written {filename}, {filesize(filename)}")
			else:
				result = False
				print(f"request error {response.status_code}")

	trainimagefile = datasets_root / Path(dataset_names[0])
	trainlabelfile = datasets_root / Path(dataset_names[1])
	testimagefile  = datasets_root / Path(dataset_names[2])
	testlabelfile  = datasets_root / Path(dataset_names[3])

	#print(f" {trainimagefile}, {filesize(trainimagefile)}")
	#print(f" {trainlabelfile}, {filesize(trainlabelfile)}")
	#print(f" {testimagefile}, {filesize(testimagefile)}")
	#print(f" {testlabelfile}, {filesize(testlabelfile)}")

	if False:
		trainimages, trainlabels = loadMNIST(trainimagefile, trainlabelfile)
		testimages, testlabels = loadMNIST(testimagefile, testlabelfile)

		dumpInfo((trainimages, trainlabels), trainimagefile, trainlabelfile)
		dumpInfo((testimages, testlabels), testimagefile, testlabelfile)

	return dataset_names

def getdb(
	outfolder, 
	istrain, 		#train or test
	kFashion=True,	#Fashion or MNIST 
	kTensor=False	#tensor or numpy
):
	datasets_root = Path(outfolder)
	site_url = get_site_url(kFashion)
	names = get_dataset_name(kFashion)	#names for the Dataset class
	dataset_names = downloadMNIST(site_url, outfolder)

	if istrain:
		dataset = FashionDataset(datasets_root/dataset_names[0], datasets_root/dataset_names[1], names[0])
	else:
		dataset = FashionDataset(datasets_root/dataset_names[2], datasets_root/dataset_names[3], names[1])
	return dataset


if __name__=='__main__':
	outfolder = 'data/'
	dirutils.mkdir(outfolder)
	datasets_root = Path(outfolder)

	train_set = getdb(outfolder, istrain=True, kTensor=False)
	test_test = getdb(outfolder, istrain=False, kTensor=False)
	print(type(train_set), len(train_set))
	print(type(test_test), len(test_test))

	#dataset_names = downloadMNIST(FASHION_site_url, outfolder)

	#train_set = CustomDataset(datasets_root/dataset_names[0], datasets_root/dataset_names[1])
	#test_test = CustomDataset(datasets_root/dataset_names[2], datasets_root/dataset_names[3])

#
#Output:
# input\t10k-images-idx3-ubyte.gz, 1648877
# input\t10k-labels-idx1-ubyte.gz, 4542
# input\t10k-images-idx3-ubyte.gz magic=0x803, count=0x2710, size=7840000
# input\t10k-labels-idx1-ubyte.gz magic=0x801, count=0x2710, size=10000
