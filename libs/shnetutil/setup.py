#from distutils.core import setup
import setuptools
#https://caremad.io/posts/2013/07/setup-vs-requirement/

# https://setuptools.readthedocs.io/en/latest/userguide/datafiles.html
# https://docs.python.org/3/distutils/setupscript.html#installing-additional-files

setuptools.setup(name='shnetutil',
	version='1.03',
	description='Complex Shearlet and Torch Utilities',
	author='Manny Ko, Ujjawal K. Panchal',
	author_email='man960@hotmail.com, ujjawalpanchal32@gmail.com',
	#url='https://www.python.org/sigs/distutils-sig/',
	packages=[
		'shnetutil',
		'shnetutil.ard',
		'shnetutil.bases',
		'shnetutil.cplx',
		'shnetutil.dataset',
		'shnetutil.math',
		'shnetutil.modelling',
		'shnetutil.mp',
		'shnetutil.pipeline',
		'shnetutil.pipeline.color',
		'shnetutil.utils',
	],
	include_package_data=True,
	package_data={'': ['bases/*.npy']},

	install_requires=[
		"torch>=1.9",		#cuda 11.1 we are using complexFloat 
		"numpy",
		"scipy",
		"cplxmodule",		#this is included here for completeness. We usually do local installs so that the source is accessible.
        'tensorly>=0.7.0',
        'tensorly-torch>=0.3.0',
        't3nsor>=1.0',
	],
)
