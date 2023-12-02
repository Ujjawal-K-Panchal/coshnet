<div align="center">
	<a href = "https://arxiv.org/abs/2208.06882">
		<img width = "300" src = "./imgs/coshnet-logo.svg"></a></img>
		
<div><p>Authors: Manny Ko, Ujjawal K. Panchal, Héctor Andrade Loarca, Andres Mendez-Vazquez</p></div>
</div>

# CoShNet Architecture

<details open>
<summary>Architecture</summary>
<img width = "850" src = "https://github.com/Ujjawal-K-Panchal/coshnet/blob/main/imgs/CoShNet-TNR.png">

CoShNet is a fully complex hybrid neural network. We use the CoShREM (now call SymFD) <code><a href>http://www.math.uni-bremen.de/cda/software.html</code>
signal transform to produce a stable embedding. The network operates entirely in $\mathbb{C}$ domain to take advantage of unique properties of CVNNs.

<details> <summary>Architecture Brief</summary>

<ol>
	<li> Input is any $32\times32 \in \mathbb{R}$ image. </li>
	<li> Input is CoShREM transformed to produce a $32\times32\times20 \in \mathbb{C}$ output. </li>
	<li> CoShREM output is convolved with the $2$ cplx-conv layers.

Each cplx-conv layer is composted of := 
$\mathbb{C}$-Conv + $\mathbb{C}$-ReLU + $\mathbb{C}$-AvgPool2d.</li>
	<li>The response is flattened and passed through $2$ cplx-linear layers.
	
Each cplx-linear layer is composted of := $\mathbb{C}$-linear layer + $\mathbb{C}$-ReLU.
</li>
	<li>The $\mathbb{R}$, $\mathbb{I}$ components are stacked together (see shape) and passed through $1$ final $\mathbb{R}$-linear layer.</li>
</ol>
</details>
</details>

# Setup
Python 3.8.x and newer are supported:

<details>
	<summary>Automated Setup</summary>
	<ol>
		<li> Create a virtualenv at the root of the repo: <code>python -m venv venv4coshnet</code> </li>
		<li> Activate venv4coshnet:
			<ul>
				<li> Windows: <code>venv4coshnet\Scripts\activate</code> </li>
				<li> Linux/MacOS: <code>source venv4coshnet/bin/activate</code> </li>
			</ul> 
		</li>
		<li> Run setup.py:
			<ul>
				<li> with <code>CUDA</code>: <code>python <a href = "./setup.py">setup.py</a></code> </li>
				<li> without <code>CUDA</code>: <code>python <a href = "./setup.py">setup.py</a> --no-cuda</code> </li>
				<li> use <code>--no-venv</code> to disable venv check (e.g. inside a docker) </li>
			</ul>
		</li>	
	</ol>

</details>

<details>
	<summary>Manual Setup</summary>
	<ul>
		<li> Please follow: <code><a href = "./setup.txt">setup.txt</a></code></li>
	</ul>
</details>

<details>
	<summary>Docker Setup</summary>
	<ul>
		<li> Build image: <code>docker build -t coshnet-docker:latest .</code> (Some systems might require running this in `sudo` mode.)</li>
	</ul>
</details>

# Contents
<div>
<details>
	<summary>Contents List</summary>
	<ol>
		<li> <code><a href = "./code/">code/</a></code>: Contains all code essential to run experiments in this repo. </li>
		<li> <code><a href = "./libs/">libs/</a></code>: Contains all custom-made and collected libs and modules we use for our experiments.
						   (Installed automatically in setup.txt)</li>
		<li> <code><a href = "./data/">data/</a></code>: Folder where datasets are present. Created automatically when running for first time.</li>
		<li> <code><a href = "./setup.txt">setup.txt</a></code>: Steps for setting up repo.</li>
		<li> <code><a href = "./requirements.txt">requirements.txt</a></code>: requirements file.</li>
		<li> <code><a href = "./changelog.md">changelog.md</a></code>: all changes relevant to releases, branch prs,
							       or any other general notes needed for maintenance.</li>
	</ol>
</details>
</div>

# How to Run?
<details>
	<summary>Running in Local</summary>
	<code>cd <a href = "./code/">code/</a></code>. Following are the possible cases:
	<div>
	<ol>
		<li> Running our models:  run: <code>python <a href = "./code/test_fashion.py">test_fashion.py</a> --help</code>
		     to see several arguments you are allowed to tune. (Default run (10k20E) gets 89.2% on <code>RTX 2080 Super</code>).
		     The default will use the 10k test set of Fashion to train for 20 epochs, and the 60k training set to test. </li>
		<li> Running resnet(18|50): run: <code>python <a href = "./code/test_resnet.py">test_resnet.py</a> --help</code>
		     to see several arguments you are allowed to set. (Default run (RN18, 10k20E) gets 88.3% on <code>RTX 2080 Super</code>).</li>
	</ol>
		Note: This code (shown in <code><a href = "./code/test_fashion.py">test_fashion.py</a></code>,<code><a href = "./code/test_resnet.py">test_resnet.py</a></code>) will not run in (<code>jupyter|google colab</code>) notebook(s). This is because our code defaults to using `asyncio` for batch generation for speed. Hence, if you absolutely have to run in a notebook, please create your own batch generation code.
	</div>
</details>

<details>
	<summary>Running in Docker</summary>
	<ul>
		<li>Run Image: <code>docker run coshnet-docker:latest</code> (Some systems might require running this in `sudo` mode.)</li>
	</ul>
	Note: The above is a brief demo for running our codebase in a docker. If you want to do something specific, e.g. deliver an API endpoint through a docker, you will have to edit the <code><a href = "./Dockerfile">Dockerfile</a></code> 
</details>


# Some Results
| Model | Epochs | Parameters | Size Ratio | Top-1 Accuracy (60k)| Top-1 Accuracy (10k) |
|-------|:------:|:----------:|:----------:|:-------------------:|:--------------------:|
| ResNet-18| 100| 11.18M| 229| 91.8%| 88.3%|
| ResNet-50| 100| 23.53M| 481| 90.7%| 87.8%|
| CoShNet(base)|20|1.37M|28|**92.2%**|**89.2%**|
| CoShNet (tiny)|20|**49.99K**|1|91.6%|88.0%|

_Note: 60k = train on train-set (60k observations), test on test-set (10k observations). 10k = vice-versa. K or k = 1000, M = Million._

# Cite
```bibtex
@misc{coshnet2022,
  doi = {10.48550/ARXIV.2208.06882},
  url = {https://arxiv.org/abs/2208.06882},
  author = {Ko, Manny and Panchal, Ujjawal K. and Andrade-Loarca, Héctor and Mendez-Vazquez, Andres},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {CoShNet: A Hybird Complex Valued Neural Network using Shearlets},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
```
# License
<div>
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
</div>
