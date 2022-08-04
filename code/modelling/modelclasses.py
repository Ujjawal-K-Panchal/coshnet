# -*- coding: utf-8 -*-
"""

Title: Model class dispatch.
    
Created on Fri Aug 20 17:44:29 2021

@author: Manny Ko & Ujjawal.K.Panchal

"""
#from enum import Enum, EnumMeta
#from collections.abc import KeysView
from typing import Callable, List, Tuple, Optional, Union

from . import CVnn

""" all the supported model classes """
supported_classes = {
	"CoShCVNN":  CVnn.CVnnFactory,
}
