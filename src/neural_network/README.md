# Neural Network

The code in this folder implements a lot of functionality in order to train a Neural Network.

Table of Contents
=================

  * [Usage](#usage)
    * [Example execution](#code_run)
    * [Initialization](#arch)
  * [Analysis](#analysis)
    * [Training evolution](#train_plots)
  
Usage
====

Creating your own neural network using this repository is really easy. If you have your data
ready you must create a "Neural Network" object and initialize it.

````
import src.neural_network.NeuralNetwork

# Creation of neural network architecture
nn = NeuralNetwork(architecture=[4, 15, 3], activation_function=['tanh', 'tanh'], learning_rate=0.1)
````

Example execution
---

In order to give to the first user a fast understanding of the code he can review the Example.py code available
in this folder. To execute it:

1) Open a terminal.
2) Change your directory to this folder.
3) Execute in the console: ``python Example.py``.
4) Ready.

The example will load the Iris Dataset and train a neural network using some of the tools that this
repository provides.

Architecture
---

The definition of the architecture is given by the input array "architecture" of the constructor. 
All the values have to be integers. The first value correspond to the number of inputs of the neural 
network. You need to be careful because this parameters will define your entire network.

The activation_function parameter can be a single string or a list of strings. If a single value is passed into
the net it will be occupied for all the neurons. At the moment the allowed activation functions are sigmoidal,
tanh and step functions.

Finally, the learning rate parameter gives you the freedom to control the scale of the updates of the
neural network in the learning procedure.



