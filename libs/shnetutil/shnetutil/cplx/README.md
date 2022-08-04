Companion code for Ivan's cplxmodule located at 'complexNN/cplxmodule' which provided
a rich set of PyTorch operators to construct CVNN. 

Also contains code that works with other modules located at ./complexNN.

1. [activations](./activations.py): Contains implementation for complex activations
that are not in Ivan's 'cplx' - e.g. MagRelu, Cardoid etc.
2. [threshold.py](./threshold.py): TODO: move denoise, thresholding code here.
3. [utils.py](./utils.py): utilities to work with 'cplx'. Currently it mostly contains support to <phase,mag>. To change format of complex numbers/arrays from different to/from *x 
    3.1. numpy.array
    3.2. cplxmodule.cplx.Cplx
    3.3. torch.Tensor
4. [cplxlayer](./cplxlayer.py): complex layer baseclasses and simple layers: (`CplxLinear`, `CplxConv2d`).
5. [CVNN_base](./CVNN_base.py): All Full Model baseclass schema with useful methods.
6. [DCF](./DCF.pys): DCF Layer from [DCF Paper](https://arxiv.org/pdf/1802.04145.pdf).
7. [dispatcher](./dispatcher.py): Common Dispatcher file for all CVnn modules.
8. [layers](./layers.py): Our specialized layers extended from various novel layers in $\mathbb{R}$ domain to $\mathbb{C}$ domain.
9. [tt_layer](./tt_layer.py): $\mathbb{R}$ valued Tensor Train Linear layers.
10. [utils](./utils.py): various complex domain utilities.