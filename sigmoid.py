import numpy as np
import matplotlib.pyplot as plt

# Generating 100 points from -10 to 10.
input = np.linspace(-10, 10, 100)

def sigmoid(X):
  return 1/(1+np.exp(-X))

output = sigmoid(input)

plt.plot(input, output)
plt.xlabel("input")
plt.ylabel("output")
plt.title("Sigmoid")
plt.show()

# Lesson learnt: Values ranging from -10 to 10 are now ranging from 0 to 1.
#  In other words: close to 0 or closer to 0
#  That makes for binary classification.

