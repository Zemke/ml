# Neural Networks in Machine Learning

Neural Networks are made of layers of neurons. Most importantly they have weighted connections and each neuron also has a bias. It is these weights and biases that when appropriately tweaked they enable the Neural Network to learn.

## Biology

Biology | Artificial NN
--- | ---
Neuronal Dendrites | ANN Input
Thickness of Dendrite | Weight of connection (think graphs)
Axon Terminal | ANN Output

From dendrites through the axon out the axon terminals.

## Forward Pass

Inputs are passed forward throught the NN. The first layer is the input layer representing the inputs (also called features). On the other end they come out the output layer as labels. In between lay the hidden layers. This is mostly done in batches of inputs to reduce the load of operations. \
In the end this is just basic Algebra: matrix multiplications after each layer including the weight and input matrices.

### Activation function

Each neuron's output also goes through an activation function. Just like in biology a neuron is only activated when the signal (output here) is strong enough.

A popular activation function is ReLU which is as easy as `f(x) = max(0, x)`.

### Loss function

After the inputs have been forward passed through the NN the output layer's neurons values are evaluated by a loss function.

The loss function compares the NN output with the expected results (also called labels) and the loss for each neuron is greater the greater the difference to the result is. Therefore the goal of the NN is to reduce loss which is done by tweaking weights and biases of the neurons and its connections.

## Backpropagation

The loss is groundwork for determining how the weights need to be tweaked. Remember learning is all about changing the weights of the NN.

Differentiation is the Calculus that helps us getting to know how to tweak the weights with the goal to reduce loss.

# Differential Calculus

Rate of change with respect to another change.

## Multiplication Rule

Move factor outside the derivative.

```
f=(fg) -> fg'+f'g
f=3; g=x^3
f=(3x^3) -> 3*3x^2+x'(3)*x^3
f=(3x^3) -> 3*3*1^2+0*x^3
f=(3x^3) -> 9^2
```

## Sum Rule

The derivative of a sum is the sum of derivatives.

```
f=(x+y) -> x'+y'
```

## Power Rule

```
x^n = nx^(n-1)
```

## Examples

The derivative of `f` with respect to `x`.

```
 f(x) = 2x      + 3y^2
f'(x) = 2*x'(x) + 3*x'(y^2)
      = 1*2*1^0 + 2*3*0^1
      = 2 + 0
      = 2
```

The derivative of `f` with respect to `x`.

```
f(x,y) = 3x^3      - y^2     + 5x      + 2
 f'(x) = 3*x'(x^3) - x'(y^2) + 5*x'(x) + 2
       = 3*3*x^2   - 2*0^1   + 5*1     + 0
       = 9^2       - 0       + 5       + 0
       = 9^2
```

The derivative of `f` with respect to `y`.

```
f(x,y) = 3x^3      - y^2     + 5x      + 2
 f'(y) = 3*y'(x^3) - y'(y^2) + 5*y'(x) + 0
       = 3*3x^2    - 2y      + 0       + 0
       = 3*3*0^2   - 2y      + 0       + 0
       = 0         - 2y      + 0       + 0
       = -2y
```

When deriving `f(x,y)` with respect to `x` then y is a constant and moved out of the derivative. \
This leaves `x'(x)` meaning the derivative of `x` with respect to `x` which is 1.

```
f(x,y) = x*y

 f'(x) = y * x'(x)
       = y * 1
       = y

 f'(y) = x * y'(y)
       = x * 1
       = x
```

The derivative of `f` with respect to `z` and `y` afterwards.

```
f(x,y,z) = 3x^3z        - y^2     + 5z      + 2yz

   f'(z) = 3x^3 * z'(z) - 0       + 5*z'(z) + 2y * z'(z)
         = 3x^3 * 1     - 0       + 5*1     + 2y * 1
         = 3x^3                   + 5*1     + 2y

   f'(y) = 0            - y'(y^2) + 0       + 2z*y'(y)
         =              - 2y                + 2z

   f'(x) = 3z*x'(x^3)   - x'(y^2) + 5*x'(z) + 0
         = 3z*3x^2      - 0       + 5*0     + 0
         = 3z*3x^2
         = 3z*3x^2
         = 9zx^2
```

When deriving with respect to `z`, then `x` and `y` are constants and moved out of the derivative. Also, the constant rule states that they derive to 0. \
When deriving with respect to `y` then `3x^3z` derives to `3*y'(x^3z) = 0` as `z` and `x` are constants and are independent of `y`. \
When deriving with respect to `x` then `z` is kept because it affects `x`. Therefore `f'(x) = 3x^3z = 3z*x'(x^3) = 3z*3x^2`.


## Chain Rule:

`f(g(x)) = f'(g(x)) * g'(x)`

```
3*x'((2x^2))^5 * 2*x'(x^2)
3*5*(2x^2)^4   * 2*2*x
15*(2x^2)^4 * 4x
```

distribute the exponent of 4:
```
15*(2^(1*4)*x^(2*4)) * 4x
15*2^4*x^8           * 4x
```

combine the `x`es:
```
15*2^4*4*x^9
```

combine the constants:
```
15*2^4*4*x^9
960*x^9
```

Chaining goes on forever:

```
`f(h(y,g(x))) = f'(g(x)) * h'(y, g(x)) * g'(x)`
```

## Gradient

A partial derivative is one derivative of a multivariate function. That is a function taking multiple parameters. The vector of all the partial derivatives of a function is called the gradient.

The Neural Network can be understood as a huge function of arithmetic operations with weights and biases as paramters. \
Differentiation helps getting to know how the weights and biases need to change to reduce loss.

Loss is the mountains, differentiation and gradient descent is the mechanism to descent the mountains into the global minimum which is the points of least loss.

# Lingo

## Perceptron

A perceptron is an algorithm for supervised learning of binary classifiers.

Money quote: "The Neural Networks work the same way as the perceptron."<sup>2</sup>

a. Inputs are multiplied with their weights called. \
b. *Add* all multiplied values and call them *Weighted Sum*. \
c. Apply the *Weighted Sum* to the correct *Activation Function*.

*Neural network without hidden layers.*

## Automatic Differentiation

[A practical application: Gradient Descent](https://medium.com/@rhome/automatic-differentiation-26d5a993692b#d4e5)

## Over-fitting

The NN has memorized correct results rather than trained for it.

Can be observed when training accuracy is greater than test accuracy.

## Vanishing gradient problem

When during backpropagation the weights get so vanishingly small that the gradient descent cannot operate anymore. The connection to the neuron kind of dies and cannot be leveraged anymore because the mathematical operations on the value of the weight are so small.

The opposite of this is **exploding gradient problem**.

## Architectures

- ResNet
- EfficientNet
- AlexNet
- VGG
- SqueezeNet
- DenseNet
- Inception

### Convolutional Neural Network

- shared weights
- convolutions are filters on images
- feature maps
- pooling layers
 - extracting the important features from the feature map

## Tensor

Data container, n-dimensional array. Not multidimensional -- n-dimensional! A scalar is therefore a 0-dimensional tensor.

Scalar is of rank 0. That's number of axes. Matrices are of rank 2

That's the theory. Practically tensors are referred to as tensors when they are n>=3 dimensional matrices.

Type | Array | Dimensions
--- | --- | ---
Scalar | 1 | 0
Vector | [1,2,...] | 1
Matrix | [[1,2,3,...],[1,2,3,...],...] | 2
Tensor | [[[1,2,3...], [1,2,3...]], [[1,2,3...], [1,2,3...]]] | n

Tensor of rank 3:

```
[[[  1,   4,   7]
  [  2,   5,   8]
  [  3,   6,   9]]
 [[ 10,  40,  70]
  [ 20,  50,  80]
  [ 30,  60,  90]]
 [[100, 400, 700]
  [200, 500, 800]
  [300, 600, 900]]]
```

[Source](https://towardsdatascience.com/quick-ml-concepts-tensors-eb1330d7760f)

## Optimizer

Algorithms to optimize the NN such as amending weights to reduce the loss. Gradient descent is used by most of them.

## Neural Network Topologies

What kind of NNs are there.

### Perceptron (P)

Single-layer NN. Input and output layer.<sup>1</sup> \
Calculation of weighted input then an activiation function (usually Sigmoid).

This is called a Perceptron model. A recurring term.

**Applications**: Classification, Encode Database, Monitor Access Data. The latter two being Multilayer Perceptron.

Multilayer perceptrons are a feed forward NN.

### Feed Forward (FF)

* Nodes never form a cycle.
* Fully connected.

> Therefore, all the nodes are fully connected. Something else to notice is that there is no visible or invisible connection between the nodes in the same layer. There are no back-loops in the feed-forward network. Hence, to minimize the error in prediction, we generally use the backpropagation algorithm to update the weight values.

**Applications**: Data Compression, Pattern Recognition, Computer Vision, Sonar Target Recognition, Speech Recognition, Handwritten Characters Recognition.

### Deep Convolutional Network (DCN)

* Classifications of images, clustering of images and object recognition
* Unsupervised construction of hierarchical image representations

**Application**: Identify Faces, Street Signs, Tumors, Image Recognition, Video Analysis, NLP, Anomaly Detection, Drug Discovery, Checkers Game, Time Series Forecasting.

### Other

Many more NNs at https://medium.com/towards-artificial-intelligence/main-types-of-neural-networks-and-its-applications-tutorial-734480d7ec8e

# TODO

- Neural Network from Scratch
- Implement Decision Tree
- Implement Random Forest
- Do NLP and Image Processing with a NN
- Play with different NN architectures
- Probability theory
- C, C++
- Data Pipelines

# Sources

https://medium.com/towards-artificial-intelligence/building-neural-networks-from-scratch-with-python-code-and-math-in-detail-i-536fae5d7bbf

<sup>1</sup>I think the input layer isn't counted, therefore two layers but a single-layer NN. \
<sup>2</sup>https://towardsdatascience.com/what-the-hell-is-perceptron-626217814f53 \

# See also

- [NN Concept Animations](https://nnfs.io/neural_network_animations) \
- [Rules-of-thumb for building a Neural Network](https://towardsdatascience.com/17-rules-of-thumb-for-building-a-neural-network-93356f9930af) \
- [Tic Tac Toe supervised Machine Learning](https://github.com/Mathspy/tic-tac-toe-NN) \
- [A Guide to transitioning from software Developer/Engineer to Machine Learning Engineer](https://towardsdatascience.com/a-guide-to-transitioning-from-software-developer-engineer-to-machine-learning-engineer-49c8395dd63a) (Also the linked Stanford lectures in there) \
- [geohot/tinygrad NN from scratch reference implementation](https://github.com/geohot/tinygrad) \
- [NN from Scratch in Python](https://github.com/Sentdex/NNfSiX/tree/master/Python)

