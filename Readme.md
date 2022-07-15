<div align="center">
	<a href = "https://arxiv.org/">
		<img width = "850" src = "./imgs/coshnet-logo.svg"></a></img>
		
<div><p>Authors: Manny Ko, Ujjawal K. Panchal, Héctor Andrade Loarca, Andres Mendez-Vazquez</p></div>
</div>



# Setup

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

# Contents
<details>
	<summary>Contents List</summary>
	<ol>
		<li> <a href = "./code/">code</a>: Contains all code essential to run experiments in this repo. </li>
		<li> <a href = "./libs/">libs</a>: Contains all custom-made and collected libs and modules we use for our experiments.
						   (Installed automatically in setup.txt)</li>
		<li> <a href = "./data/">data</a>: Folder where datasets are present. Created automatically when running for first time.</li>
		<li> <a href = "./setup.txt">setup</a>: Steps for setting up repo.</li>
		<li> <a href = "./requirements.txt">requirements</a>: requirements file.</li>
		<li> <a href = "./changelog.md">changelog</a>: all changes relevant to releases, branch prs,
							       or any other general notes needed for maintenance.</li>
	</ol>
</details>

# How to Run?
<details>
	<summary>Running Instructions</summary>
	<code>cd <a href = "./code/">code/</a></code>. Following are the possible cases:
	<div>
	<ol>
		<li> Running our models:  run: <code>python <a href = "./code/test_fashion.py">test_fashion.py</a> --help</code>
		     to see several arguments you are allowed to tune.
		     The default will use the 10k test set of Fashion to train and the 60k training set to test. </li>
		<li> Running resnet(18|50): run: <code>python <a href = "./code/test_resnet.py">test_resnet.py</a> --help</code>
		     to see several arguments you are allowed to set. </li>
	</ol>
	</div>
</details>

# Cite

