# Copyright (C) Manchor Ko - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# * Proprietary and confidential
# * Written by Manchor Ko man961@yahoo.com, August 2019
#
from os import path

def filesize(filepath):
	try:
		filesize = path.getsize(filepath)
	except:
		filesize = 0
	return filesize

def comparefiles(file1, file2):
	with open(file1, "rb") as f1:
		with open(file2, "rb") as f2:
			return file1.read() == f2.read()
	return False
