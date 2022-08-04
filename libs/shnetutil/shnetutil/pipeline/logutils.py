# -*- coding: utf-8 -*-
"""
Title: Helper to setup and use 'logging'

@author: Manny Ko & Ujjawal.K.Panchal
"""
import logging, sys

#TODO: https://docs.python.org/3/library/logging.config.html#logging-config-dictschema

kDefFormatter=logging.Formatter('%(levelname)s: <%(module)s>: %(message)s')

def getLogger(name):
	return logging.getLogger(name)

def enable_console(logger, formatter=kDefFormatter, level=logging.INFO):
	if not logger:
		return
	stream_handler = logging.StreamHandler(sys.stdout)	#default to sys.stderr
	stream_handler.setLevel(level)
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)

def disable_console(logger):
	if not logger:
		return
	for stream in logger.handlers:
		if type(stream) is logging.StreamHandler:
			logger.removeHandler(stream)

def setup_logger(logger, level=logging.INFO, file_name='test_batch.log', kConsole=True):
	logger.setLevel(level)

	formatter = kDefFormatter	#TODO: make this a parameter too

	#1: setup FileHandler to also our .info()
	if file_name is not None:
		file_handler = logging.FileHandler(file_name, mode='w')
		file_handler.setLevel(level)
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)

	#2: setup StreamHandler (console) to also receive our .info()
	if kConsole:
		enable_console(logger, formatter, level)
