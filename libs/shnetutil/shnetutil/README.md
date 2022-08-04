# Library for ShearNet Utilities

**This Library contains a lot of utilities useful for creating, training, testing, tracing/logging Shearlet Transform and Complex Valued Neural Networks and it's pipeline.**


## Content:

1. [ard](./ard/): Contains modules for ARD. relevant sparsity calculation.

2. [bases](./bases/): Contains DCF (Decomposed Convolution Filters) bases.

3. [cplx](./cplx/): Contains useful modules for operating with/developing different types of *Complex Valued Neural Networks*.

4. [dataset](./dataset/): Our dataset handling utilities.

5. [math](./math/): Contains various stats related stuff used for sampling from dataset while batching.

6. [modelling](./modelling/): Contains various modelling utilities. Eg.: Some base classes called `ModelFactory` for making our type of models. 

4. [mp](./mp/): Contains various utilities for multiprocessing in pipeline. 

5. [pipeline](./pipeline/): Contains various useful utilities for use during construction of the pipeline like: 

	* augmentation (transforms).
	* batching (async. i/o. based).
	* Loading MNIST like Datasets.
	* utilities for logging.
	* Plotting performance metrics (ROC, Prec/Recall).    

6. [utils](./utils/): Various type of utilities:
	* torchutils (utilities for pytorch and CUDA).
	* ourlogger.py (sets up logger that tracks information and stores in [code/logs/](../code/logs/)).
	* trace.py (classes for noting down various types of logs from models and training pipeline).

6. [coshrem_xform](./coshrem_xform.py): Contains tools for conducting *Complex Shearlet Transform*, __in broadcasted form__ on data. This is derived from *[3., 4.]*.

7. [projconfig](./projconfig.py/): Helps get the context directory in which the project will execute.

10. [pysh_xform](./pysh_xform.py): Discrete Shearlet transform PyCoShREM for static data which will be stored to disk.
*(obsolete and non-broadcasted (inefficient) version)*

11. [shearletxform](): using real value based pyshearlab *[7.]*.*(obsolete and non-broadcasted (inefficient) version)*

12. [shxform](./shxform.py): Parent Class for handling CoShREM based xform in [coshrem_xform](./coshrem_xform.py).

13. [torchutils](./torchutils.py): Contains Pytorch related utilities. Some examples:

	* Automatic CPU/GPU selection and garbage collection.
	* Optimizer choosing.
	* Saving/Loading snapshots.
	etc.

14. [trace](./trace.py): All logging/tracing utilities are contained in this file.

## TODO:

-  Modify current **README.md**.
-  Create **README.md**s for *pipeline*.

## References:

1. [Deep Complex Networks](https://openreview.net/forum?id=H1T2hmZAb).
	*  Derived from [cplxmodule](https://github.com/ivannz/cplxmodule).

2. [PyCoShREM](https://github.com/rgcda/PyCoShREM).

3. [Shearlab](http://shearlab.math.lmu.de/).

4. [Sydney Hsu and Manny Ko's  covid_19 collector](../../data/covid19_collector)
	* Derived from [github link](https://github.com/SidneyHsuYC/covid19_collector).

5. [pyshearlab](https://github.com/stefanloock/pyshearlab).