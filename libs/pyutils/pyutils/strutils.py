# Copyright (C) Manchor Ko - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# * Proprietary and confidential
# * Written by Manchor Ko man961@yahoo.com, August 2020
#
def numericSuffix(mystr:str) -> int:
	suffix = ''
	for i in range(len(mystr)-1, 0, -1):
		if mystr[i].isdigit():
			suffix += mystr[i]
		else:
			break
	return int(suffix[::-1])
	
