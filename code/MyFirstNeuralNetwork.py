import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import math


class MyFirstNeuralNetwork(nn.Module):
    def __init__(self, in_size=2, out_size=1, hidden_size=3):

        super(MyFirstNeuralNetwork, self).__init__()

        # Set the dimensionality of the network
        self.input_size = in_size
        self.output_size = out_size
        self.hidden_size = hidden_size

        # Initialize our weights
        self._init_weights()

    '''
    Initialize the weights
    '''
    def _init_weights(self):
        # Create an input tensor of shape (2,3)
        self.W_Input = torch.randn(self.input_size, self.hidden_size)

        # Create an output tensor of shape (3, 1)
        self.W_Output = torch.randn(self.hidden_size, self.output_size)

    '''
    Create the forward pass
    '''
    def forward(self, inputs):
        # Lets get the element wise dot product
        self.z = torch.matmul(inputs, self.W_Input)
        # We call the activation
        self.state = self._activation(self.z)
        # Pass it through the hidden layer
        self.z_hidden = torch.matmul(self.state, self.W_Output)
        # Finally activate the output
        output = self._activation(self.z_hidden)

        # Return the output
        return output

    '''
    Backpropagation algorithm implemented
    '''
    def backward(self, inputs, labels, output):
        # What is the error in output
        self.loss = labels - output

        # What is the delta loss based on the derivative
        self.loss_delta = self.loss * self._derivative(output)

        # Get the loss for the existing output weight
        self.z_loss = torch.matmul(self.loss_delta, torch.t(self.W_Output))

        # Compute the delta like before
        self.z_loss_delta = self.z_loss * self._derivative(self.state)

        # Finally propogate this to our existing weight tensors to update
        # the gradient loss
        self.W_Input += torch.matmul(torch.t(inputs), self.z_loss_delta)
        self.W_Output += torch.matmul(torch.t(self.state), self.loss_delta)

    '''
    Here we train the network
    '''
    def train(self, inputs, labels):
        # First we do the foward pass
        outputs = self.forward(inputs)
        # Then we do the backwards pass
        self.backward(inputs, labels, outputs)

    '''
    Here we perform inference
    '''
    def predict(self, inputs):
        pass

    '''
    Here we save the model
    '''
    def save(self):
        pass

    '''
    Our non-linear activation function
    '''
    def _activation(self, s):
        # Lets use sigmoid
        return 1 / (1 * torch.exp(-s))

    '''
    Our derivative function used for backpropagation
    Usually the sigmoid prime
    '''
    def _derivative(self, s):
        # derivative of sigmoid
        return s * (1 - s)
