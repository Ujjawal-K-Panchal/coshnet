# Research purpose implementation of Tensor Train and Tensor Ring algorithms in Pytorch

[Paper](https://arxiv.org/pdf/1901.10787.pdf)

### System requirements
- Python 3
- CPU or NVIDIA GPU + CUDA

### Dependencies and Getting Started
- ``torch >= 1.0.0``
- ``torchvision >= 0.2.1``
- ``numpy``
- ``sympy``
- ``scipy``

You may install `PyTorch` using any suggested method for your environment [here](https://pytorch.org/get-started/locally/).

Also, after cloning the repo, you can run  ``python setup.py install`` in the command line to install the required packages.


#### Additional setup steps for us:
1. `pip install torchtext`
2. if it says anywhere `from torchtext import <something>`, change it to `from torchtext.legacy import <something>`
3. `pip install spacy`
4. `python -m spacy download en_core_web_sm`
5. Make new directory. `./sentiment/logdir`

### Setting up experiments

To check the experiments settings, see a file ``experiments.sh``.

For example, to run an experiment for TT embedding layer you can run:

```sh
python train.py --embed_dim 256 --dataset imdb --embedding tt \
    --n_epochs 100 --d 3 --ranks 16 --gpu 1
```

### Repository structure

The directory `t3nsor` contains classes and function for TT and TR decompositions, embedding layers and so on.
The directory `sentiment` contains the models and experiment setting files.

## Authors
- Valentin Khrulkov
- Oleksii Hrinchuk
- Leyla Mirvakhabova
- Elena Orlova
- Ivan Oseledets

If you use these algorithms in your research we kindly ask you to cite our work

```
@article{khrulkov2019tensorized,
  title={Tensorized {E}mbedding {L}ayers {F}or {E}fficient {M}odel {C}ompression},
  author={Khrulkov, Valentin and Hrinchuk, Oleksii and Mirvakhabova, Leyla and Orlova, Elena and Oseledets, Ivan},
  journal={arXiv preprint arXiv:1901.10787},
  year={2019}
}
```
