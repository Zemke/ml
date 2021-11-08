#!/usr/bin/env python

import math
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

The main difference to Sentdex/NNfSiX is that the implementation
here does the forward pass (dot product + bias) on a per-neuron level.
Sentdex/NNfSiX does the pass for the whole layer.
This is visible in how DenseLayer.forward is implemented which
delegates the dot product calculation to every single neuron in its
layer rather than doing the matrix product itself.

Weights, biases, activation, loss, optimization
Optimize weights and biases output by activations to reduce loss.
"""

class Ops:

  @staticmethod
  def accuracy(X, y):
    predictions = np.argmax(X, axis=1)
    y1 = np.argmax(y, axis=1) if len(y.shape) == 2 else y
    return np.mean(predictions == y1)


class Optimization:

  @staticmethod
  def GradientDescent():
    pass

  @staticmethod
  def StochasticGradientDescent():
    pass


class CategoricalCrossEntropy:

  def forward(self, X, tt):
    """
    An example:
    There are three classes the NN is meant to result into.
    Using one-hot encoding this translates to a vector of 3.
    tt represents that (t as in target).
    yy is the vector actual predictions of the NN.
    If the prediction yy is [.7, .1, .2] then each number is the confidence
    of the NN per categorical variable as per one-hot encoding.
    For example .7 is the confidence that it's class at tt[0].
    Since one-hot encoding creates kind of a binary content vector
    ML categorical cross-entropy simplifies to just the negative natural log
    of the predictions (multiplying by 0 gives nothing, by 1 gives everything).
    """
    clipped = np.clip(X, 1e-7, 1 - 1e-7)  # prevent results of -inf through log(0)
    sample_r = range(len(X))
    if len(tt.shape) == 1:
      confidences = clipped[sample_r, tt]
    elif len(tt.shape) == 2:
      # one-hot encoded
      confidences = clipped[sample_r, np.where(tt == 1)[1]]
    else:
      raise Exception("Invalid target tt shape " + tt.shape)
    losses = -np.log(confidences)
    self.y = losses
    return np.mean(losses)

  def backward(self, X, y_true):
    samples = len(X)
    if len(y_true.shape) == 2:
      y_true = np.argmax(y_true, axis=1)
    self.X_grad = X.copy()
    self.X_grad[range(samples), y_true] -= 1
    self.X_grad = self.X_grad / samples


class Step:

  def forward(self, X):
    return np.vectorize(lambda x: int(x > 0))(X)

  def backward(self, X):
    raise Exception("not yet implemented")


class ReLU:

  def forward(self, X):
    self.X = X
    return np.maximum(X, 0)

  def backward(self, X_grad):
    self.X_grad = X_grad.copy()
    self.X_grad[self.X <= 0] = 0
    return X_grad


class Softmax:

  def forward(self, X):
    exp = np.exp(X - np.max(X, axis=1, keepdims=True))
    norm = exp / np.sum(exp, axis=1, keepdims=True)
    self.y = norm
    return norm

  def backward(self, X):
    self.X_grad = np.empty_like(X)
    for index, (single_y, single_X_grad) in enumerate(zip(self.y, X)):
      single_y = single_y.reshape(-1, 1)
      jacobian_matrix = np.diagflat(single_y) - np.dot(single_y, single_y.T)
      self.X_grad[index] = np.dot(jacobian_matrix, single_X_grad)
    return self.X_grad


class DenseLayer:

  def __init__(self, n_inputs, n_neurons, activation):
    self.weights = .01 * np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))
    self.activation = activation

  def forward(self, X):
    self.X = X

    self.y = self.activation.forward(np.dot(X, self.weights) + self.biases)
    return self.y

  def backward(self, X_grad):
    self.weight_grad = np.dot(self.X.T, X_grad)
    self.bias_grad = np.sum(X_grad, axis=0, keepdims=True)
    self.X_grad = np.dot(X_grad, self.weights.T)
    return X_grad


X, y = spiral_data(samples=200, classes=3)

layers = [DenseLayer(2, 3, ReLU()), DenseLayer(3, 3, Softmax())]
loss_fn = CategoricalCrossEntropy()

print("--- Forward passes ---")

layers[0].forward(X)
print("layer1 y:")
print(layers[0].y)

layers[1].forward(layers[0].y)
print("layer2 y:")
print(layers[1].y)

print("loss", loss_fn.forward(layers[1].y, y))
print("acc", Ops.accuracy(layers[1].y, y))

print("--- Backpropagation ---")

loss_fn.backward(layers[1].y, y)
layers[1].backward(loss_fn.X_grad)
layers[0].activation.backward(layers[1].X_grad)
layers[0].backward(layers[0].activation.X_grad)

print('grads')
print(layers[1].X_grad)
print(layers[1].weight_grad)
print(layers[1].bias_grad)
print(layers[0].X_grad)
print(layers[0].weight_grad)
print(layers[0].bias_grad)

