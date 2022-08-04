# Copyright (C) Manchor Ko - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# * Proprietary and confidential
# * Written by Manchor Ko man961@yahoo.com, August 2019
#
import os, sys


# default folder filer is a pass-through:
def deffolder_filter(subdir): 
	subd = os.path.basename(subdir)
	return True if subd == '' else subd[0] != '.' and subd[0] != '..'

# default file filter - ignore '.xxx'
deffile_filter = lambda file: True if file == '' else file[0] != '.'
# selects .png files
pngfile_filter = lambda file: deffile_filter(file) and (os.path.splitext(file)[-1].lower() == '.png')
# selects .jpg files
jpgfile_filter = lambda file: deffile_filter(file) and (os.path.splitext(file)[-1].lower() == '.jpg')
# selects .npy files
npyfile_filter = lambda file: deffile_filter(file) and (os.path.splitext(file)[-1].lower() == '.npy')
# selects .wav files
wavfile_filter = lambda file: deffile_filter(file) and (os.path.splitext(file)[-1].lower() == '.wav')

#robust routine to form a folder path 'dirname'
def addslash(folder):
	return folder if folder[-1] == '/' else folder + '/'

#create 'dirname' folder if it does not exist
def ensure_dir_exists(dir_name):
	if not os.path.exists(dir_name):
		os.makedirs(dir_name)
		
def folder_iter(
	rootdir,
	functor,
	context = None,
	folder_filter = deffolder_filter,
	file_filter   = deffile_filter,
	followlinks	  = True,				#follow symlinks or not
	logging=True
):
	if type(rootdir) is str:
		rootdir = rootdir if rootdir[-1] == '/' else rootdir + '/'
	
	for subdir, dirs, files in os.walk(rootdir, topdown=True, followlinks=followlinks):
		#print("subdir %s" % subdir)
		#1: give our 'folder_filter' a chance to decide
		if not folder_filter(subdir):
			continue

		if logging:
			print("subdir %s" % subdir)
		#print(os.path.dirname(subdir))
		for file in files:
			#2: give our 'file_filter' a chance to decide
			if (file_filter(file)):
				filepath = os.path.join(subdir, file)
				functor(filepath, context)
	return

def functor(filepath, context):
	filelist = context
	print(f"  functor({filepath})")
	filelist.append(filepath)

# default folder filer is a pass-through:
def ourfolder_filter(subdir, pattern='input'):
	print(f"entering '{subdir}'", end='')
	#1: first make sure 'subdir' pass the default dir filter checks:
	result = deffolder_filter(subdir)
	if result:
		result = pattern in subdir
		if (not result):
			print(" skipping...")
		else:
			print(" ->")
		return result
	print("\n")
	return result


if __name__=='__main__':
	rootdir = './'		#where we want to start looking
	filelist = []		#this will be used to store files that pass the filters

	folder_iter(
		rootdir,
		functor,
		context = filelist,
		folder_filter = ourfolder_filter,	#filter out all folders that do not have 'input' in it
		file_filter   = pngfile_filter,		#filter out all files except .png
		logging=False
	)
	print(filelist)
