# What

Looking at a replay or live gameplay of Worms Armageddon, how to get to know which weapons were used?

# Goal

The end result is automatize the tool used in [this](https://www.twitch.tv/videos/814304436) video at the top to track used weapons.

# How

Besides all the rather manual approaches like bare-bones sound and image recognition the obvious idea is Machine Learning.

Reciting the manual approach: Some weapons make different sounds when used. This could be one way to the used weapon. \
Doesn't work for all and it probably gets complicated the deeper you dig.

# Approach

Machine Learning

Start finding the usage of a single weapon. For that I choose **shotgun**. \
It has a simple starting and ending sequence unlike ropes which can be shot in the air which wouldn't actually account for using it. Or worm select is used and then you're circling through your worms which. \
The weapon is also always visible in the worm's hands when used, the sound it makes is distinct in the game. Also it's used quite frequently which should make for more test data.

# What is an Artificial Neural Network?

> Simply put, an ANN represents interconnected input and output units in which each connection has an associated weight. During the learning phase, the network learns by adjusting these weights in order to be able to predict the correct class for input data.

https://medium.com/towards-artificial-intelligence/building-neural-networks-from-scratch-with-python-code-and-math-in-detail-i-536fae5d7bbf

## Training

I have no idea so far. My naive Machine Learning approach goes something like this:

1. Choose an appropriate Neural Network design for the task.
2. Let it look at replays.
3. At the right time tell it, "That was as shotgun."
4. Profit.

### Training data

Given the tool [WAaaS](https://waaas.zemke.io) it gives at what time of the replay which weapon is used.

# Human Learning

The irony is: in order for Machine Learning to exist, the human must learn Machine Learning. Here are my findings:

# Biology

Biology | Artificial NN
--- | ---
Neuronal Dendrites | ANN Input
Thickness of Dendrite | Weight of connection (think graphs)
Axon Terminal | ANN Output

From dendrites through the axon out the axon terminals.

# Lingo

## Layers

Given I've learnt about input, hidden and output layers it's safe to assume that these three exist. Given the information from below (Hidden layers) there's always at leat one input layer and one output layer. The layers between the two -- should they exist in the NN -- are hidden layers. \
Hidden layers are only connected among themselves or with input or output layers but never with the outer world.

## Perceptron

A perceptron is an algorithm for supervised learning of binary clasifiers.

Money quote: "The Neural Networks work the same way as the perceptron."<sup>2</sup>

a. Inputs are multiplied with their weights called.
b. *Add* all multiplied values and call them *Weighted Sum*.
c. Apply the *Weighted Sum* to the correct *Activation Function*.

*Neural network without hidden layers.*

## Backpropagation

Initially the nodes receive a random weight because we don't know the importance of each connection. \
When the NN has output it is compared to the expected output. Using backpropagations the weights are re-evaluated by how much the output is off from what was expected. \
This can work with a **gradient descent** algorithm.

Weight decides how vital the information is for the output.

### Gradient descent

Since in **backpropagation** the idea is to re-evaluate the weights, gradient descent is an optimization algorithm. It's used when training a machine learning model.

### Binary Classifier

Classify into two groups (binary, you know?) by a classficiation rule. \
I assume the classification rule could be an activation function, but also  **TODO**

> Perceptron is usually used to classify the data into two parts. Therefore, it is also known as a Linear Binary Classifier. [Source](https://towardsdatascience.com/what-the-hell-is-perceptron-626217814f53)

## Activation Function

**TODO**

Gets output of a note. Also refered to as Transfer Function.

Proceeding to just pasting here as I don't fully understand but might recall later:

[Source](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)

> It is used to determine the output of neural network like yes or no. It maps the resulting values in between 0 to 1 or -1 to 1 etc. (depending upon the function).

> **Derivative or Differential**: Change in y-axis w.r.t. change in x-axis.It is also known as slope.
> **Monotonic function**: A function which is either entirely non-increasing or non-decreasing.

## Neural Network Topologies

What kind of NNs are there.

### Perceptron (P)

Single-layer NN. input and output layer.<sup>1</sup> \
Calculation of weighted input then an activiation function (usually Sigmoid).

This is called a Perceptron model. A recurring term.

**Applications**: Classification, Encode Database, Monitor Access Data. The latter two being Multilayer Perceptron.

Multilayer perceptrons are a feed forward NN.

### Feed Forward (FF)

* Nodes never form a cycle.
* Fully connected.

> Therefore, all the nodes are fully connected. Something else to notice is that there is no visible or invisible connection between the nodes in the same layer. There are no back-loops in the feed-forward network. Hence, to minimize the error in prediction, we generally use the backpropagation algorithm to update the weight values.

**Applications**: Data Compression, Pattern Recognition, Computer Vision, Sonar Target Recognition, Speech Recognition, Handwritten Characters Recognition.

### Radial Basis Network (RBN)

**TODO** 

Many more NNs at https://medium.com/towards-artificial-intelligence/main-types-of-neural-networks-and-its-applications-tutorial-734480d7ec8e

# TODO

[6. Sigmoid function](https://medium.com/towards-artificial-intelligence/building-neural-networks-from-scratch-with-python-code-and-math-in-detail-i-536fae5d7bbfo)
[Monte Carlo simulation](https://medium.com/towards-artificial-intelligence/building-neural-networks-from-scratch-with-python-code-and-math-in-detail-i-536fae5d7bbf)

# Sources

https://medium.com/towards-artificial-intelligence/building-neural-networks-from-scratch-with-python-code-and-math-in-detail-i-536fae5d7bbf

<sup>1</sup>I think the input layer isn't counted, therefore twok layers but a single-layer NN.
<sup>2</sup>https://towardsdatascience.com/what-the-hell-is-perceptron-626217814f53
<sup>3</sup>https://medium.com/towards-artificial-intelligence/building-neural-networks-from-scratch-with-python-code-and-math-in-detail-i-536fae5d7bbf

# See also

- [Rules-of-thumb for building a Neural Network](https://towardsdatascience.com/17-rules-of-thumb-for-building-a-neural-network-93356f9930af)
- [Tic Tac Toe supervised Machine Learning](https://github.com/Mathspy/tic-tac-toe-NN)

