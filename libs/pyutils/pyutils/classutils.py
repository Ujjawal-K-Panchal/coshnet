# Copyright (C) Manchor Ko - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# * Proprietary and confidential
# * Written by Manchor Ko man961@yahoo.com, Sept 2021
#
from inspect import ismethod

def method_exists(instance, method):
    return hasattr(instance, method) and ismethod(getattr(instance, method))
