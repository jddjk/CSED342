#!/usr/bin/python

import random
import collections # you can use collections.Counter if you would like
import math

import numpy as np

from util import *

SEED = 4312

############################################################
# Problem 1: hinge loss
############################################################

def problem_1a():
    """
    return a dictionary that contains the following words as keys:
        so, touching, quite, impressive, not, boring
    """
    # BEGIN_YOUR_ANSWER
    words = ['so', 'touching', 'quite', 'impressive', 'not', 'boring']
    weights = {word: 0 for word in words}

    examples = [
        ({'so': 1, 'touching': 1}, 1),
        ({'impressive': 1, 'quite': 1}, 1),
        ({'so': 1, 'boring': 1}, -1),
        ({'not': 1, 'boring': 1}, -1)
    ]


    for x, y in examples:
        wx = sum(weights[word] * x.get(word, 0) for word in weights)
        hinge_loss = max(1 - wx * y, 0)
        if hinge_loss > 0:
            for word in weights:
                weights[word] -= y* x.get(word)

    return weights
    # END_YOUR_ANSWER

############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_ANSWER
    return collections.Counter(x.split())
    # END_YOUR_ANSWER

############################################################
# Problem 2b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    '''
    weights = {}  # feature => weight

    def sigmoid(n):
        return 1 / (1 + math.exp(-n))

    # BEGIN_YOUR_ANSWER
    for _ in range(numIters):
        for x, y in trainExamples:
            phi = featureExtractor(x)
            wx = dotProduct(weights, phi)
            prediction = sigmoid(wx)
            if y == 1:
                increment(weights, (1 - prediction) * eta, phi)
            elif y == -1:
                increment(weights, -1 * prediction * eta, phi)
            
    return weights
    # END_YOUR_ANSWER

############################################################
# Problem 2c: bigram features

def extractNgramFeatures(x, n):
    """
    Extract n-gram features for a string x
    
    @param string x, int n: 
    @return dict: feature vector representation of x. (key: n consecutive word (string) / value: occurrence)
    
    For example:
    >>> extractNgramFeatures("I am what I am", 2)
    {'I am': 2, 'am what': 1, 'what I': 1}

    Note:
    There should be a space between words and NO spaces at the beginning and end of the key
    -> "I am" (O) " I am" (X) "I am " (X) "Iam" (X)

    Another example
    >>> extractNgramFeatures("I am what I am what I am", 3)
    {'I am what': 2, 'am what I': 2, 'what I am': 2}
    """
    # BEGIN_YOUR_ANSWER
    words = x.split()
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
    return collections.Counter(ngrams)
    # END_YOUR_ANSWER

############################################################
# Problem 3: Multi-layer perceptron & Backpropagation
############################################################

class MLPBinaryClassifier:
    """
    A binary classifier with a 2-layer neural network
        input --(hidden layer)--> hidden --(output layer)--> output
    Each layer consists of an affine transformation and a sigmoid activation.
        layer(x) = sigmoid(x @ W + b)
    """
    def __init__(self):
        self.input_size = 2  # input feature dimension
        self.hidden_size = 16  # hidden layer dimension
        self.output_size = 1  # output dimension

        # Initialize the weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
        self.init_weights()

    def init_weights(self):
        weights = np.load("initial_weights.npz")
        self.W1 = weights["W1"]
        self.W2 = weights["W2"]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        """
        Inputs
            x: input 2-dimensional feature (B, 2), B: batch size
        Outputs
            pred: predicted probability (0 to 1), (B,)
        """
        # BEGIN_YOUR_ANSWER
        
        self.a0 = x
        print('forward')
        print(x)
        self.z1 = np.matmul(self.a0, self.W1) + self.b1
        print('forward-z1')
        print(self.W1)
        print(self.b1)
        print(self.z1)
        self.a1 = self.sigmoid(self.z1)
        print(self.a1)
        self.z2 = np.matmul(self.a1, self.W2) + self.b2
        print('forward-z2')
        print(self.W2)
        print(self.b2)
        print(self.z2)
        self.a2 = self.sigmoid(self.z2)
        print('forward-a2')
        print(self.a2)
        print(self.a2.squeeze())
        return self.a2.squeeze()
        # END_YOUR_ANSWER

    @staticmethod
    def loss(pred, target):
        """
        Inputs
            pred: predicted probability (0 to 1), (B,)
            target: true label, 0 or 1, (B,)
        Outputs
            loss: negative log likelihood loss, (B,)
        """
        # BEGIN_YOUR_ANSWER
        eps = 1e-15
        print('loss-pred')
        print(pred)
        pred = np.clip(pred, eps, 1 - eps)
        print('loss-clip pred')
        print(pred)
        return -(target * np.log(pred) + (1 - target) * np.log(1 - pred))
        # END_YOUR_ANSWER

    def backward(self, pred, target):
        """
        Inputs
            pred: predicted probability (0 to 1), (B,)
            target: true label, 0 or 1, (B,)
        Outputs
            gradient: a dictionary of gradients, {"W1": ..., "b1": ..., "W2": ..., "b2": ...}
        """
        # BEGIN_YOUR_ANSWER
        pred = pred.reshape(-1, 1)
        print(pred)
        target = target.reshape(-1, 1)
        print(target)

        d_loss = pred - target
        # Backprop.
        d_W2 = np.matmul(self.a1.T, d_loss)
        print('d_W2')
        print(d_W2)
        d_b2 = np.sum(d_loss, axis=0, keepdims=True)
        print('d_b2')
        print(d_b2)
        d_sigmoid = self.sigmoid_derivative(self.a1)
        print('d_sigmoid')
        print(d_sigmoid)
        d_hidden_layer = np.matmul(d_loss, self.W2.T) * d_sigmoid
        print('d_hidden_layer')
        print(d_hidden_layer)
        d_W1 = np.matmul(self.a0.T, d_hidden_layer)
        print('d_W1')
        print(d_W1)
        d_b1 = np.sum(d_hidden_layer, axis=0, keepdims=True)
        print('d_b1')
        print(d_b1)
        
        return {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2}
        # END_YOUR_ANSWER
    
    def update(self, gradients, learning_rate):
        """
        A function to update the weights and biases using the gradients
        Inputs
            gradients: a dictionary of gradients, {"W1": ..., "b1": ..., "W2": ..., "b2": ...}
            learning_rate: step size for weight update
        Outputs
            None
        """
        # BEGIN_YOUR_ANSWER
        self.W1 -= learning_rate * gradients["W1"]
        self.W2 -= learning_rate * gradients["W2"]
        self.b1 -= learning_rate * gradients["b1"]
        self.b2 -= learning_rate * gradients["b2"]
        # END_YOUR_ANSWER

    def train(self, X, Y, epochs=100, learning_rate=0.1):
        """
        A training function to update the weights and biases using stochastic gradient descent
        Inputs
            X: input features, (N, 2), N: number of samples
            Y: true labels, (N,)
            epochs: number of epochs to train
            learning_rate: step size for weight update
        Outputs
            loss: the negative log likelihood loss of the last step
        """
        # BEGIN_YOUR_ANSWER
        for epoch in range(epochs):
            for x, y in zip(X, Y):
                print('train')
                print(x)
                print(y)
                x = x.reshape(1, -1)
                print(x)
                y = np.array([y])
                print(y)
                # Forward pass
                pred = self.forward(x)
                # Backprop
                gradients = self.backward(pred, y)
                # Update weights
                self.update(gradients, learning_rate)
                print(self.loss(pred, y))

        return self.loss(pred, y).sum()
        # END_YOUR_ANSWER

    def predict(self, x):
        return np.round(self.forward(x))