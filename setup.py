import os, sys, setuptools, subprocess
import platform
import argparse
from collections import namedtuple
from typing import Optional
from pathlib import Path

#static vars.
venvname = "venv4setup2S"
install_reqs_file = "install_requirements.py"
supported_os = ['Linux', 'Windows', 'Macos']


## python supported versions.
min_py_ver = 380
max_py_ver = 400

## supported cuda versions.
min_cuda_ver = 101
max_cuda_ver = 114

def pInsideVENV() -> bool:
	return sys.prefix != sys.base_prefix

def getVENV() -> str:
	if not pInsideVENV():
		return None
	sysprefix = Path(sys.prefix)
	venv = sysprefix.parts[-1]
	return venv

def check_py_version_support(min_version: int = min_py_ver, max_version: int = max_py_ver):
	version = sys.version_info
	version_num = version.major*100 + version.minor*10
	if not (version_num >= min_version and version_num <= max_version):
		raise Exception(f"<!>: python version: {version_num} not supported. Python should be in range: [{min_version}, {max_version}].")
	return version_num

def get_cuda_version() -> int:
	process = subprocess.Popen(
						'nvcc -V',
						stdout = subprocess.PIPE,
						stderr = subprocess.PIPE,
						shell = True
				)
	out, error = process.communicate()
	if error:
		print (f"<!>: CUDA not detected.") 
		return None
	full_version_str = str(out).split("V")[-1]
	splitted = full_version_str.split(".")
	#version_num = int("".join(full_version_str.split(".")[:-1]))
	#print(f"{splitted=} {len(splitted)}")
	major = splitted[0]
	minor = splitted[1]
	return int(major + minor)

def check_cuda_version_support(min_version: int = min_cuda_ver, max_version: int = max_cuda_ver):
	version_num = get_cuda_version()

	if version_num and (not (version_num >= min_cuda_ver and version_num <= max_version)):
		raise Exception(f"<!>: CUDA version: {version_num} not supported. CUDA should be in range: [{min_version}, {max_version}].")
	return version_num

def install_requirements(cudaver: Optional[int] = None):
	#1. update stuff.
	os.system("python -m pip install --upgrade pip")	#in python 3.8 pip is a builtin module
	os.system("pip install wheel")

	#2. if cuda flavor install, install torch with cuda.
	if cudaver:
		os.system(f"pip install torch==1.9.0+cu{cudaver} torchvision==0.10.0+cu{cudaver} torchaudio===0.9.0 \
			-f https://download.pytorch.org/whl/torch_stable.html")
	
	#3. install other requirements (not skips over torch if already present).
	os.system("pip install -r requirements.txt")
	return

if __name__ == "__main__": 
	parser = argparse.ArgumentParser(description='CoShREM NN based on cplex')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
	parser.add_argument('--no-venv', action='store_true', default=False, help='run without a venv')
	args = parser.parse_args()

	no_cuda = args.no_cuda
	no_venv = args.no_venv

	#1. detect OS.
	osname = platform.system()
	
	#2. check that python version is supported.
	pyver = check_py_version_support(min_py_ver, max_py_ver)

	#3. cuda ver check.
	cudaver = None if no_cuda else check_cuda_version_support(min_cuda_ver, max_cuda_ver)

	#4: check
	venv = getVENV()

	#5. print system-stats:
	print("System Stats:")
	print("===")
	print(f"1. OS name: {osname}")
	print(f"2. Python version: {pyver}")
	print(f"3. CUDA version detected: {cudaver}")
	print(f"venv '{getVENV()}' found.")

	if (not no_venv) and (not venv):
		raise Exception(f"<!>: not running inside a virtualenv.")

	install_requirements(cudaver)
