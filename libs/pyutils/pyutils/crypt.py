# Copyright (C) Manchor Ko - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# * Proprietary and confidential
# * Written by Manchor Ko man961@yahoo.com, August 2019
#
import cryptography
from cryptography.fernet import Fernet
import base64
import os
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

kLogging=True

aws_access_key_id='DxQ-BElg2kdBsJcr0zZCjsCGqQTzQ5KxDhroy7HIvBI='
aws_secret_access_key='qcQ1PgMazttU03LPK6CxGPtbjS98QZ1-zrdaQ8pAO3s='

def generate_key(tofile='Fernet.key'):
	key = Fernet.generate_key()

	if (tofile != None):
		with open(tofile, 'wb') as file:
			file.write(key) # The key is type bytes still
	return key

def load_key(keyfile='Fernet.key'):
	try:
		with open(keyfile, 'rb') as file:
			key = file.read() 	# The key is type bytes still
	except:
		key = None
	return key

def encrypt(msg, key):
	#print(type(msg))
	msg_b = msg if (type(msg) == bytes) else msg.encode() 	#to bytes
	f = Fernet(key)
	encrypted = f.encrypt(msg_b)
	return encrypted

def encryptstream(stream, key):
	buf = stream.getvalue()
	encrypted = encrypt(buf, key)
	return encrypted

def encryptfile(filename, key):
	basename = os.path.basename(filename)
	fname, ext = os.path.splitext(filename)
	encoded_f = fname + '-encoded' + ext

	with open(filename, "rb") as f:
		buf = f.read()
		encrypted = encrypt(buf, key)
		with open(encoded_f, "wb") as of:
			of.write(encrypted)
	return encoded_f

def decrypt(encrypted, key):
	encrypted = encrypted if(type(encrypted) == bytes) else encrypted.encode()
	f = Fernet(key)
	try:
		msg = f.decrypt(encrypted)
		result = msg.decode()
	except:
		result = None
	return result

def decryptstream(stream, key):
	buf = stream.getvalue()
	data = decrypt(buf, key)
	return data

def decryptfile(filename, key):
	with open(filename, "rb") as f:
		buf = f.read()
		data = decrypt(buf, key)
	return data

if __name__=='__main__':
	#1: load or generate an encryption key
	ourkey = load_key()
	if ourkey == None:
		ourkey = generate_key()
	print(ourkey)

	#2: our 'secret'
	print(f"aws_access_key_id: {aws_access_key_id}")
	
	#3: encrypt our secret
	encrypted = encrypt(aws_access_key_id, ourkey)
	print(f"encrypted: {encrypted}, {type(encrypted)}")

	#4: decrypt our secret
	decrypted = decrypt(encrypted, ourkey)
	print(f"decrypted: {decrypted}, {type(decrypted)}")

	testIAM = 'input/testIAM-credentials.csv'

	encoded_f = encryptfile(testIAM, ourkey)

	testIAM_data = decryptfile(encoded_f, ourkey)
	print(testIAM_data)