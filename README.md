# computerGeneratedHolography

---

### Table of Contents
Sections headers that will be used to reference location of destination.

- [Description](#description)
- [How To Use](#how-to-use)
- [License](#license)
- [Author Info](#author-info)

## Description
This Github project was developped during the 6-month internship (*Projet de Fin d'Etudes* in french) to complete the Electronics and Computer Engineering (*Electronique et Informatique Industrielle (EII)* in french) course at INSA Rennes. The internship subject was "Study, adaptation and implementation of source separation methods for the coding of holographic videos".

#### Directory tree

```bash
└───computerGeneratedHolography
    ├───data
    │   ├───2D_dice
    │   └───dices1080p-AP
    ├───implementations_codes
    │   ├───SF-analysis
    │   ├───source-separation
    │   │   ├───mixture_dataset(0147)
    │   │   ├───mixture_dataset(2points)
    │   │   └───output
    │   └───TF-analysis
    ├───machine_learning
    ├───neural_network
    │   ├───classification_problem
    │   │   ├───backup
    │   │   ├───hologram
    │   │   │   └───logs
    │   │   │       └───old
    │   │   └───wigner_distribution
    │   │       └───logs
    │   │           └───old
    │   ├───old_jupyter_notebooks
    │   ├───regression_problem
    │   │   ├───example
    │   │   ├───hologram
    │   │   │   └───logs
    │   │   └───wigner_distribution
    │   │       └───logs
    │   └───wigner_distribution
    ├───output
    │   ├───dataset
    │   │   └───oneClass
    │   ├───machine_learning
    │   ├───main
    │   ├───neural_networks
    │   └───wigner_distribution
    ├───rapports
    └───tests_wigner_distribution
        └───tests
```

#### Folders
- root (computerGeneratedHolography)
Contains the .m files to create holograms and restore them to images. It also contains .m scripts to create the datasets used by the neural networks and a .m script to locate the particules positions from the hologram.

- data
Contains some images and holograms used for testing (not relevant).

- implementations_codes
Contains some tests and analysis made in the time-frequency (MATLAB, python) and space-frequency domains (jupyter notebook). It also contains the tests and implementations made for the source separation in the holograms (python, jupyter notebook).

- machine_learning
Contains implementations of some machine learning algorithms (python, jupyter notebook). In the case, an object (class) was created with certain features of a hologram.

- neural_networks
Contains the implementation of a neural networks (python) to solve a classification and a regression problem. The objective is to train the neural network with holograms and also their neural networks.

- output
Contains the outputs and the results of the algorithms and scripts of this Github project.

- rapports
Contains some reports made, the final report of the internship and the defense presentation.

- test_wigner_distribution
Contains scripts recovered for the calculation of the Wigner distribution (MATLAB, jupyter notebook) and the reference used (zip folder) for this calculation.

#### Technologies
- Python 3.7.6
- MATLAB R2016a

---

## How to use

### Preparing the enviroment (Windows user)

#### 1. Installing Python (https://www.python.org/)

To check if python is installed on your computer, you must open the command prompt and write "python". If you see the python environment, then it is installed, otherwise, to install you can download on the folling link (https://www.python.org/downloads/) the desired version.

#### 2. Installing Jupyter (https://jupyter.org/)

The process to install jupyter lab is shown in the jupyter website (https://jupyter.org/install). I have used "pip" to install with the follow commmand, but there are others options.

```bash
$ pip install notebook
```

#### 3. Installing python libraries

To install the python libraries that will be used in this project just install it from the "requirements.txt" file. Execute in the terminal the command (if you are in the main folder of the project, otherwise find the path where the requirements file is located): 

```bash
$ pip install -r ./requirements.txt
```

#### 4. Installing Octave (https://www.gnu.org/software/octave/)

Click on the folling link and download the file according to your operating system (https://www.gnu.org/software/octave/download.html). I highlight that the .m scripts were written in MATLAB, so I recommend use it. Octave is capable to execute theses codes, but some simple changes must be made.

---

## License

MIT License

Copyright (c) [2020] [Fernando Lucas Araujo Amaral]

Permission is hereby granted, free of charge, to any person obtainign a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Author Info

- LinkedIn - [/fernandolucasaa](https://www.linkedin.com/in/fernandolucasaa/)
- Website - [Fernando Lucas Araujo Amaral](https://fernandolucasaa.github.io/)