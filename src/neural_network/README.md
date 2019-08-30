# Neural Network

The code in this folder implements a lot of functionality in order to train a Neural Network.

Table of Contents
=============`=`===

  * [Cloning the repository][#cloning]
  * [Usage](#usage)
    * [Example execution](#example-execution)
    * [Initialization](#arch)
  * [Analysis](#analysis)
    * [Training evolution](#train_plots)
  

Cloning the repository
===

To clone the repository please follow the next steps (you must have the git software in your system).

1) Choose a folder where the repository will be stored and go to it.
2) Run ``git clone https://github.com/rudyn2/cc5114.git``
3) Ready.

Usage
====

Creating your own neural network using this repository is really easy. If you have your data
ready you must create a "Neural Network" object, initialize it and then fit it and train it.

````
import src.neural_network.NeuralNetwork

# Creation of neural network architecture
nn = NeuralNetwork(architecture=[4, 15, 3], activation_function=['tanh', 'tanh'], learning_rate=0.1)

# Fit and train the neural network
nn.fit(X, y)
nn.train(100, verbose=True)
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

Analysis
===

I have used the Example script to perform some experiments with the neural network. First of all, i have trained
the neural network using the Iris Dataset from sklearn. The purpose of this is seeing the evolution of the loss
and accuracy of the net along the training.

![Learning Curve](https://raw.githubusercontent.com/rudyn2/cc5114/master/src/neural_network/example_resources/learning_curve.png)






