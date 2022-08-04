# Copyright (C) Manchor Ko - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# * Proprietary and confidential
# * Written by Manchor Ko man961@yahoo.com, August 2020
#

def first_true(iterable, default=False, pred=None):
	""" lifted our of more_itertools """
	return next(filter(pred, iterable), default)
