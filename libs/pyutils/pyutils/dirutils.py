# Copyright (C) Manchor Ko - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# * Proprietary and confidential
# * Written by Manchor Ko man961@yahoo.com, August 2019
#
import os, sys
from pathlib import Path
  

#check to see if 'dirname' is already a valid folder    
def direxist(dirname):
	return os.path.isdir(dirname) and os.path.exists(dirname)

#create 'dirname' folder if it does not exist - single level only
def mk1dir(dirname):
	if not direxist(dirname):
		ourpath = Path(dirname)
		os.mkdir(dirname)
	#os.makedirs(os.path.dirname(model_path), exist_ok=True)	

def mkdir(dirname):
	""" create 'dirname' folder if it does not exist """
	if not direxist(dirname):
		ourpath = Path(dirname)
		for nest in range(len(ourpath.parts)):
			folders = ourpath.parts[0:nest+1]
			subpath = Path(*folders)
			mk1dir(subpath)

#robust routine to form a folder path 'dirname'
def mkdirname(dirname):
	return dirname if dirname[-1] == '/' else dirname + '/'

#pass '__file__' from your main program to extract where your script is running from
def workingdir(myfilepath):
	dir_path = os.path.dirname(os.path.realpath(myfilepath))
	return mkdirname(dir_path)
	
