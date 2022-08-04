# code:
This folder contains various Complex Shearlet Network (CoShNet) trained on the Fashion MNIST + MNIST Dataset. 
Two most prominent file being `test_fashion` which is the main script for CoShCVNN 
to build our CVNN and `test_resnet.py` which is used to train our baselines ResNet(18|50).
convolution etc.

## Contents:

1. [logs/](./logs/): This folder contains logs of various runs related to ShearNet on Fashion. These logs are extremely useful to see stuff like:

	* __Response__ of various layers *(wstats ([t1.]) and histograms)*
	* __Input__ to the network (raw input and wstats)
	* __Weight Matrices__ of different layers. *(wstats)*.
	* __Testing Results__ of different types of runs.

	Different naming conventions like [t2.], [t3.] etc. denote the machine on which the tests were conducted. There are a few other self explanatory naming conventions used in the context of the folder.

2. [modelling/](./modelling/): This folder contains model architectures for models of different types and some related useful utilities for the same.

3. [pipeline/](./pipeline/): This folder contains useful utilities for pipeline of the model.

4. [snapshots/](./snapshots/): This folder is supposed to contain saved snapshots of models and their relevant optimizers *(saving and restoring optimizers does in practice affect performance)*. They can be easily identified by their naming convention along with number of epochs. (Note: you may look at `loadmodel.py` on how to load a sample snapshot. Note it won't run unless you have a snapshot with that name.)

5.  [projconfig](./projconfig.py): Utility for getting specific folders for the dataset.

6. [test_fashion](./test_fashion.py): Train and test Fashion with [__CoShCVNN()__](./modelling/CVnn.py).

7. [test_resnet](./test_resnet.py): Trains and tests ResNet(18|50) in our training pipeline.

## Terminology:
[t1.] __wstats__: Mean, Min, Max of an array of values. In exactly that order.

[t2.] __mck__: genesis in Manny Ko's machine.

[t3.] __uj__: genesis in Ujjawal K. Panchal's machine.

[t4.] __ARD__: [Automatic Relevance Determination](https://link.springer.com/chapter/10.1007/978-1-4471-0847-4_15).

## References:
[1.] [Fashion MNIST Dataset from torchvision](https://pytorch.org/docs/stable/torchvision/datasets.html#fashion-mnist)
