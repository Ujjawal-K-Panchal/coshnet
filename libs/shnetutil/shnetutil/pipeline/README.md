# Pipeline

Contains all code relevant to developing an efficient pipeline for Complex Valued ShearNet.
The most prominent module is 'batch.py' which contains our batch builders.

# Content

1. [augmentation](./augmentation.py): Comprises of many types of transforms that can be performed on data. *Shearlet Transform* which is essential to this project is one of them.
2. [batch](./batch.py): Comprises of all utilities for *multiprocessing and asynchronous i/o.* based batching directly from disk (several advantages over having it in RAM, **even in terms of speed in some cases**). This utility is based on a better randomization strategy than pytorch's DataLoader and is also comparitively faster (**in most cases**).
3. [color](./color): Colorspace utilities for future.
4. [dbaugmentations.py](./dbaugmentations.py): dbAugmentations.py contains various data augmentation pipelines/recipes/methodologies for datasets.
5. [loadMNIST](./loadMNIST.py): This file contains utilities to download/process MNIST like datasets.
6. [logutils](./logutils.py): Logging Utilities.
7. [roc](./roc.py): Utilities to plot *ROC (receiver operating characteristic)*  curves during testing.
8. [torchbatch](./torchbatch.py): This file is used to load the batch as a Pytorch tensor on a selected device of choice.
9. [trainutils](./trainutils.py): Utilities for training pipeline schema, runs etc.