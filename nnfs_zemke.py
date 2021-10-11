#!/usr/bin/env python

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


"""
batches of inputs are fed into the NN
here 3 inputs in a batch of two
[[1,2,3], [4,5,6]]
        /\
       /  \    therefore the input layer must
      /    \   consist of two neurons
     /      \
   _/_       \__
  |   |     |   |  each neuron does the dot product
  |___|     |___|  of the input matrix and weight matrix
   \          /
    \        /
    Activation     all outputs of a layer are
     function      passed into an activation function
       \  /        for hidden layers that's often ReLU
       _\/_
      |    |  again input matrix
      |____|  mutliplied by weight matrix
        |
        |  the activation function for output layers
        |  is often different to the hidden layers
        |  (Softmax for example)
        V
      labels

The input to a NN is often called features and referenced as X
by convention.
The output is referred to as labels and denoted as yfeatures.

one bias    per neuron
one weight  per input       per neuron
  n neurons per layer
  x inputs  per neuron
  y inputs  per input batch

There's always y inputs to a neuron and one output.
In a densely connected NN each neuron in the layer receives
the same inputs.
"""


class Activation:
  """
  Activation functions are applied to the output of all layer's neurons.
  That is for each neuron in the layer the dot product + bias.
  Therefore they're run on layer-leven rather than for each neuron.
  """

  @staticmethod
  def ReLU(X):
    """
    Very commonly used in hidden layers.
    """
    return np.maximum(X, 0)

  @staticmethod
  def Step(X):
    """
    Was popular before ReLU.
    """
    return np.vectorize(lambda x: int(x > 0))(X)

  @staticmethod
  def Sigmoid(X):
    raise Exception("TODO")

  @staticmethod
  def Softmax(X):
    """
    Often used for the output layer.
    Softmax is Exponentiation then Normalization.
    Softmax requires the outputs of all neurons in the layer.
    """
    # as always in numpy the operation is done element-wise
    # axis 0 is max in col, 1 is max in row
    # max reduces an elemnts array to a scalar but keepdims retains
    #  that dimension making it a one-element vector
    exp = np.exp(X - np.max(X, axis=1, keepdims=True))
    norm = exp / np.sum(exp, axis=1, keepdims=True)
    return norm


class Neuron:
  
  def __init__(self, n_inputs, weights):
    """
    Neurons are weights per input and a bias.
    After the forward pass an activation function is applied.
    """
    self.weights = weights
    self.bias = 0

  def forward(self, X):
    """
    The forward pass of a neuron is the dot product of
    its weights and the inputs.
    i[0] * weights[0] + i[1] * weights[1] ... + bias
    """
    # numpy often works on an element-basis meaning the
    #  + bias part is applied to each element of the numpy array
    self.y = np.dot(np.array(X), self.weights) + self.bias
    return self.y


class DenseLayer:

  def __init__(self, n_inputs, n_neurons):
    """
    A layer is made up of neurons.
    """
    weights = .1 * np.random.randn(n_inputs, n_neurons).T
    self.neurons = [Neuron(n_inputs, weights[i]) for i in range(n_neurons)]

  def forward(self, X, activation_fn):
    """
    The inputs to each neuron in the layer are the same
    in a densely connected NN. Every neuron from the previous
    layer is connected to each neuron in the next layer.
    The neurons only differ by their assigned weights.
    """
    y = []
    for n in self.neurons:
      fp = n.forward(X)
      y.append(fp)
    self.y = activation_fn(np.array(y).T)
    # expected shape should be (batch_size, n_neurons)
    return self.y


# per convention X is the training data input also called the features
# the NN ends up outputting what's called the labels by convention y
X, y = spiral_data(100, 3)

# number of neurons in the previous layer is the number of inputs
#  per neuron in the next layer
layers = [DenseLayer(2, 3), DenseLayer(3, 3)]

layers[0].forward(X, Activation.ReLU)
print("layer1 y:")
print(layers[0].y)

layers[1].forward(layers[0].y, Activation.Softmax)
print("layer2 y:")
print(layers[1].y)

