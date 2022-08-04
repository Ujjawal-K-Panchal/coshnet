# -*- coding: utf-8 -*-
"""

Title: Shearlet based CNN vs. simple MLP in Fashion MNIST.
	
Created on Mon June 30 17:44:29 2020

@author: Manny Ko & Ujjawal.K.Panchal 

"""
import sys, os, time
from pathlib import Path

def time_spent(tic1, tag='', count=1):
	toc1 = time.time() 
	print(f"time spend on {tag} method = {(toc1 - tic1)/count:.2f}s")
	return toc1

def parent_folder(curfile):
	""" tests/ often are subfolder of a module folder. We need to access data/, logs/ etc """
	ourpath = Path(curfile)
	#filepath = "D:/Dev/SigProc/onsen/network_dev/COVID/pipeline"
	return str(Path(*ourpath.parts[0:-2]))
	
def setup_tests(curfile):
	""" pass '__file__' to setup importing from parent folder """
	parent = parent_folder(curfile)  	#"D:/Dev/SigProc/onsen/network_dev/COVID/pipeline"
	sys.path.append(parent)
