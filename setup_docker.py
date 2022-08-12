import setup

if __name__ == "__main__": 
	no_cuda = False #turn at your whim.
	no_venv = False #turn at your whim.

	#1. detect OS.
	osname = setup.platform.system()
	
	#2. check that python version is supported.
	pyver = setup.check_py_version_support(setup.min_py_ver, setup.max_py_ver)

	#3. cuda ver check.
	cudaver = None if no_cuda else setup.check_cuda_version_support(setup.min_cuda_ver, setup.max_cuda_ver)

	#4: check
	venv = setup.getVENV()

	#5. print system-stats:
	print("System Stats:")
	print("===")
	print(f"1. OS name: {osname}")
	print(f"2. Python version: {pyver}")
	print(f"3. CUDA version detected: {cudaver}")
	print(f"venv '{setup.getVENV()}' found.")

	if (not no_venv) and (not venv):
		raise Exception(f"<!>: not running inside a virtualenv.")

	setup.install_requirements(cudaver)