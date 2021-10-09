#!/usr/bin/env python

import numpy as np


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
"""


class Activation:

  @staticmethod
  def ReLU(i):
    return max(i, 0)

  @staticmethod
  def step(i):
    return int(i > 0)

  @staticmethod
  def sigmoid(i):
    raise Exception("TODO")

  @staticmethod
  def softmax(i):
    raise Exception("TODO")


class Neuron:
  
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias

  def forward(self, ii):
    # numpy often works on an element-basis
    #  meaning the + bias part is applied to each element
    #  of the numpy array
    # i[0] * weights[0] + i[1] * weights[1] ... + bias
    self.output = np.dot(self.weights, ii) + self.bias
    return self.output

X = [1, 2, 3, 2.5]

# every input per neuron has a weight
weights = [[0.2, 0.8, -0.5, 1.0],  # weights of inputs into neuron 1
           [0.5, -0.91, 0.26, -0.5],  # weights of inputs into neuron 2
           [-0.26, -0.27, 0.17, 0.87]]  # weights of inputs into neuron 3

nn = [Neuron(weights[0], 2.0), Neuron(weights[1], 3.0), Neuron(weights[2], 0.5)]

for n in nn:
  n.forward(X)
  print(n.output)

