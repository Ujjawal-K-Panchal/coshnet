# -*- coding: utf-8 -*-
"""

Title: Dataset loader and conditioner for COVID dataset
    
Created on Thurs July 6 17:44:29 2020

@author: Sydney Hsu & Manny Ko

"""
import logging

def setuplogger(logfilename='imageloader.log'):
	logger = logging.getLogger(__name__)
	# logger.setLevel(logging.INFO)
	logger.setLevel(logging.WARNING)

	formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

	file_handler = logging.FileHandler(logfilename, encoding="utf-8")
	file_handler.setFormatter(formatter)

	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)

	logger.addHandler(file_handler)
	logger.addHandler(stream_handler)
	return logger