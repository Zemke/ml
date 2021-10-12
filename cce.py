import math


def cce(yy, one_hot):
  """
  Categorical Cross-Entropy
  Loss function run after the output layer
  Softmax activation function.
  """
  r = 0
  for i in range(len(one_hot)):
    print(one_hot[i], yy[i])
    r += one_hot[i] * math.log(yy[i])
  return -(r)


yy = [0.7, 0.1, 0.2]  # predictions
one_hot = [1, 0, 0]  # targets

print("cce", cce(yy, one_hot))


