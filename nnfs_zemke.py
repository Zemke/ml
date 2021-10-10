#!/usr/bin/env python

import numpy as np

import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


"""
for each node in the graph
which is a neuron in the neural net

batch of inputs
      |
      V
input * weight + bias <-------|
      |                       |
      | activation function   | per layer
      | probably ReLU         | (not recursive)
      V                       |
    output --------------------
      |
      | activation function toward output layer
      | which is maybe probably softmax
      V
 output layer

one bias    per neuron
one weight  per input       per neuron
  n neurons per layer
  x inputs  per neuron
  y inputs  per input batch

there's always y inputs to a neuron
 and one output
in a densely connected NN each neuron in the layer
 receives the same inputs
"""


class Activation:
  """
  Apply activation functions on batches of inputs.
  """

  @staticmethod
  def ReLU(X):
    return np.maximum(X, 0)

  @staticmethod
  def step(X):
    return np.vectorize(lambda x: int(x > 0))(X)

  @staticmethod
  def sigmoid(X):
    raise Exception("TODO")

  @staticmethod
  def softmax(X):
    raise Exception("TODO")


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
    self.output = np.dot(np.array(X), self.weights) + self.bias
    return self.output


class DenseLayer:

  def __init__(self, n_inputs, n_neurons):
    """
    A layer is made up of neurons.
    """
    weights = .1 * np.random.randn(n_inputs, n_neurons).T
    self.neurons = [Neuron(n_inputs, weights[i]) for i in range(n_neurons)]

  def forward(self, X):
    """
    The inputs to each neuron in the layer are the same
    in a densely connected NN. Every neuron from the previous
    layer is connected to each neuron in the next layer.
    The neurons only differ by their assigned weights.
    """
    y = []
    for n in self.neurons:
      fp = n.forward(X)
      # activation function applies to result of forward pass
      y.append(Activation.ReLU(fp))
    self.y = np.array(y).T
    # expected shape should be (batch_size, n_neurons)
    return self.y


# per convention X is the training data input also called the features
# the NN ends up outputting what's called the labels by convention y
# this is an input of 4 in a batch of size 3
X, y = spiral_data(100, 3)
print("X")
print(X)


# number of neurons in the previous layer is the number of inputs
#  per neuron in the next layer
layers = [DenseLayer(2, 5)]

layers[0].forward(X)
print("layer1 y:")
print(layers[0].y)

