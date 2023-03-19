# Implementation of Feedforward Neural Network with Backpropagation
    Course Assignment of CS6910 Deep Learning IIT Madras
## Abstract<br/>
The feedforward neural network is built from scratch using only ```numpy``` library for all matrix/vector operations and without using any automatic differentiation packages.
## Dataset<br/>
The implemented feedforward network is trained and tested on the ```Fashion MNIST``` dataset consisting of a training set of 60,000 examples and a test set of 10,000 examples.
Each example is a 28x28 grayscale image, associated with a label from 10 classes. The best 3 hyperparameter configurations obtained for ```Fashion MNIST``` are also used for 
training and testing on ```MNIST``` dataset.
## Objective<br/>
The objective is to implement a variety of optimizers such as ```stochastic gradient descent```, ```momentum based gradient descent```, ```nesterov accelerated gradient descent```, ```rmsprop```, ```adam``` and ```nadam``` and
doing a hyperparameter search efficiently using ```wandb```.
## Folder Structure<br/>
The files named ```Q<x>.ipynb``` have codes specific for the subproblems given in the [Assignment](https://wandb.ai/cs6910_2023/A1/reports/CS6910-Assignment-1--VmlldzozNTI2MDc5).
The file named ```train.py``` is a ```python``` script that takes the hyperparameters as command line arguments, and trains the dataset for the given hyperparameter configuration 
to obtain test accuracy on the test data. 
## Results<br/>
The best test accuracy on ```Fashion MNIST``` dataset achieved is **87%** while on ```MNIST``` dataset is **97%**. The explanation and results of subproblems 
can be accessed [here](https://wandb.ai/cs22m059/cs6910_dl_assgn_1_q_1/reports/CS6910-Assignment-1--VmlldzozODI2MjUz?accessToken=2h1is0xkc4ulro78mya3drlqh59qcig8nazwyi6fi6gdv1luky81ebgn81lwvybe).
