# What is an Artificial Neural Network?

> Simply put, an ANN represents interconnected input and output units in which each connection has an associated weight. During the learning phase, the network learns by adjusting these weights in order to be able to predict the correct class for input data.

https://medium.com/towards-artificial-intelligence/building-neural-networks-from-scratch-with-python-code-and-math-in-detail-i-536fae5d7bbf

# Human Learning

The irony is: in order for Machine Learning to exist, the human must learn Machine Learning. Here are my findings:

# Biology

Biology | Artificial NN
--- | ---
Neuronal Dendrites | ANN Input
Thickness of Dendrite | Weight of connection (think graphs)
Axon Terminal | ANN Output

From dendrites through the axon out the axon terminals.

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

# Lingo

## Layers

Given I've learnt about input, hidden and output layers it's safe to assume that these three exist. Given the information from below (Hidden layers) there's always at leat one input layer and one output layer. The layers between the two -- should they exist in the NN -- are hidden layers. \
Hidden layers are only connected among themselves or with input or output layers but never with the outer world.

## Perceptron

A perceptron is an algorithm for supervised learning of binary classifiers.

Money quote: "The Neural Networks work the same way as the perceptron."<sup>2</sup>

a. Inputs are multiplied with their weights called. \
b. *Add* all multiplied values and call them *Weighted Sum*. \
c. Apply the *Weighted Sum* to the correct *Activation Function*.

*Neural network without hidden layers.*

## Automatic Differentiation

[A practical application: Gradient Descent](https://medium.com/@rhome/automatic-differentiation-26d5a993692b#d4e5)

## Backpropagation

Initially the nodes receive a random weight because we don't know the importance of each connection. \
When the NN has output it is compared to the expected output. Using backpropagations the weights are re-evaluated by how much the output is off from what was expected. \
This can work with a **gradient descent** algorithm.

Weight decides how vital the information is for the output.

Backpropagation also applies **automatic differentiation** which is used to calculate the **gradient descent** efficiently (see below (see below)).

## Loss function

Also referred to as cost function.

Calculates the gradient of the loss function. This sounds fancy and I don't really know what it means. But loss functions come into play somewhere in the realm of backpropagation and optimizing the graph.

Remember, training the NN on a test data set is essentially an operation to improve the assignment of weights in the NN. \
The lower the loss value, the less a value is off from what was expected. So, loss should be as low as possible.

There's some subtleties to this, though, like over-fitting.

## Over-fitting

The NN has memorized correct results rather than trained for it.

Can be observed when training accuracy is greater than prediction accuracy.

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


## Activation Function

Gets output of a node. Also refered to as Transfer Function.

Proceeding to just pasting here as I don't fully understand but might recall later:

[Source](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)

> It is used to determine the output of neural network like yes or no. It maps the resulting values in between 0 to 1 or -1 to 1 etc. (depending upon the function).

### Sigmoid

NNs are mainly for classification, in binary classification there are naturally two types. \
Sometimes the result of a NN the result can be any amount of arbitrary numbers. Sigmoid can group these into two outcomes for a binary classification.

See `sigmoid.py` for an implementation.

### ReLu

```
f(x)=max(0,x)
```

According to [this](https://deepai.org/machine-learning-glossary-and-terms/relu) ReLu replacing Sigmoid for typical use because it saves on computation.

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

[Neural Network Implementation from Scratch](https://medium.com/towards-artificial-intelligence/building-neural-networks-from-scratch-with-python-code-and-math-in-detail-i-536fae5d7bbf)

# Sources

https://medium.com/towards-artificial-intelligence/building-neural-networks-from-scratch-with-python-code-and-math-in-detail-i-536fae5d7bbf

<sup>1</sup>I think the input layer isn't counted, therefore two layers but a single-layer NN. \
<sup>2</sup>https://towardsdatascience.com/what-the-hell-is-perceptron-626217814f53 \
<sup>3</sup>https://medium.com/towards-artificial-intelligence/building-neural-networks-from-scratch-with-python-code-and-math-in-detail-i-536fae5d7bbf

# See also

- [NN Concept Animations](https://nnfs.io/neural_network_animations) \
- [Rules-of-thumb for building a Neural Network](https://towardsdatascience.com/17-rules-of-thumb-for-building-a-neural-network-93356f9930af) \
- [Tic Tac Toe supervised Machine Learning](https://github.com/Mathspy/tic-tac-toe-NN) \
- [A Guide to transitioning from software Developer/Engineer to Machine Learning Engineer](https://towardsdatascience.com/a-guide-to-transitioning-from-software-developer-engineer-to-machine-learning-engineer-49c8395dd63a) (Also the linked Stanford lectures in there) \
- [geohot/tinygrad NN from scratch reference implementation](https://github.com/geohot/tinygrad) \
- [NN from Scratch in Python](https://github.com/Sentdex/NNfSiX/tree/master/Python)

