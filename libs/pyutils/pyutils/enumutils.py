# Copyright (C) Manchor Ko - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# * Proprietary and confidential
# * Written by Manchor Ko man961@yahoo.com, August 2021
#
from enum import Enum, EnumMeta
from collections.abc import KeysView

# https://betterprogramming.pub/5-ways-to-get-enums-in-python-3e5d6e610ec1

def qMemberOf(enumeration: Enum, key):
	""" Predicate for 'key' being a member of 'enumeration' """
	assert(isinstance(enumeration, Enum))	#strong type checking
	return key in enumeration

def Enum2Str(enum1):
	""" extract the name associate with 'enum1' """
	return enum1.name 		#'kCoShCVNN'

def IsValidValue(enumeration: Enum, val):
	""" Predicate for 'val' being one of the .value within the enum """
	return val in [en.value for en in enumeration]

class Enumbase(Enum):
	""" Add some helpers for enum.Enum """
	pass

	@classmethod
	def dict(cls) -> KeysView:
		""" Cast ourselves into a dict() """
		return cls.__members__

	@classmethod
	def names(cls) -> KeysView:
		return cls.__members__.keys()

	@classmethod
	def qMemberOf(cls, key):
		""" Set membership """
		return qMemberOf(cls, key)

	@classmethod
	def isValid(cls, val):
		""" Predicate for 'val' being one of the valid values """
		return IsValidValue(cls, val)

	@classmethod
	def values(cls):
		return [en.value for en in cls]	

	@classmethod
	def LookupbyValue(cls, val, default=None):
#		assert(cls.isValid(val))
		for en in cls:
			if en.value == val:
				return en
		return default		
