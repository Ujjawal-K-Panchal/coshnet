# -*- coding: utf-8 -*-
"""

Title: Complex Visualization.
	
Created on Mon Mar 16 17:44:29 2020

@author: Ujjawal.K.Panchal & Hector Andrade-Loarca
"""
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import numpy as np
from . import utils

def getComplexImage(z: np.ndarray):
	hue_start = 90
	rmin = 0
	rmax = 1
	# get amplidude of z and limit to [rmin, rmax]
	amp = np.abs(z)
	amp = np.where(amp < rmin, rmin, amp)
	amp = np.where(amp > rmax, rmax, amp)
	ph = np.angle(z, deg=1) + hue_start
	
	# HSV are values in range [0,1]
	h = (ph % 360) / 360
	s = 0.85 * np.ones_like(h)
	v = (amp -rmin) / (rmax - rmin)
	return hsv_to_rgb(np.dstack((h,s,v)))


def complexImageShow(z: np.ndarray, figsize: tuple = (6,6)):
	#z = z[::-1]
	plt.figure(figsize = figsize)
	plt.axis("off")
	return plt.imshow(getComplexImage(z))

def showimage(image, figsize = (6,6), show=False):
	plt.figure(figsize = figsize)
	plt.axis("off")
	plt.imshow(image, cmap = "gray")
	if show:
		plt.show()

def saveComplexImage(z, path, figsize = (6,6), title = "Complex Image"):
	complexImageShow(z, figsize = figsize)
	plt.title(title)
	plt.savefig(path)
	return 

