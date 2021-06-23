# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
"""
This is the main entry point for MP6. You should only modify code
within this file and neuralnet_part2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# ^^^^^^^^^^^^ Mans did hella work



import numpy as np
import torch
# import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


# Classic Shallow Network Implementation from the 1980s
class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        """
        Initialize the layers of your neural network

        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param yhat - an (N,out_size) tensor
            @param y - an (N,) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param in_size: Dimension of input
        @param out_size: Dimension of output

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size
        We recommend setting the lrate to 0.01 for part 1

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        # print(lrate)
        lrate = 0.01

        # print("{} {}".format(in_size,out_size))
        
        # https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
        # https://stackoverflow.com/questions/46141690/how-to-write-a-pytorch-sequential-model

        # self.model = nn.Sequential(OrderedDict([(nn.Linear(in_size, 32)), (nn.ReLu()), (nn.ReLu(), nn.Linear(32, out_size))]))
        self.model = nn.Sequential(nn.Linear(in_size, 32), nn.ReLU(), nn.Linear(32, out_size))
        
        self.optimiser = optim.SGD(self.parameters(), lrate, weight_decay = 1e-2)
        # self.optimiser = optim.SGD(self.parameters(), lr=0.01, weight_decay = 1e-4)

    def forward(self, x):        
        """ A forward pass of your neural net (evaluates f(x)).

        @param x: an (N, in_size) torch tensor

        @return y: an (N, out_size) torch tensor of output from the network
        """

        # The forward() function should do a forward pass through your network. This means it should explicitly evaluate FW(x).
        # This can be done by simply calling your nn.Sequential() object defined in __init__()

        return self.model(x)

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        # outputs = net(inputs)
        
        # clear the gradient buffer.
        self.optimiser.zero_grad()
        yhat = self.forward(x)
        loss_value = self.loss_fn(yhat, y)
        # print(loss)
        loss_value.backward()
        self.optimiser.step()
    
        # loss_value.item()
        # loss_value.detach().cpu().numpy() 

        return loss_value.detach().cpu().numpy() #  This makes sure that the returned loss value is detached from the computational graph after one execution of the step() function and proper garbage collection can take place.


def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M,) torch tensor
    @param n_iter: int, the number of iterations of training
    @param batch_size: The size of each batch to train on. (default 100)

    # return all of these:

    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: A NeuralNet object

    # NOTE: This must work for arbitrary M and N
    """
   
    # Data Standardization: Convergence speed and accuracies can be improved greatly by simply centralizing your input data by subtracting the sample mean and dividing by the sample standard deviation. 
    # More precisely, you can alter your data matrix X by simply doing X:=(X−μ)/σ.
    
    print("Train_set: {}".format(train_set))
    print("Dev_set: {}".format(dev_set))
    lrate = 0.01
    
    train_mean = torch.mean(train_set)
    train_std = torch.std(train_set)
    train_set = (train_set - train_mean) / train_std

    dev_mean = torch.mean(dev_set)
    dev_std = torch.std(dev_set)
    dev_set = (dev_set - dev_mean) / dev_std

    # print(len(train_set[0]))
    net = NeuralNet(lrate, nn.CrossEntropyLoss(), len(train_set[0]), 2)
    
    losses = []

    batch = int(len(train_set) / batch_size)
    print("batch: {}".format(batch))
    # Train
    for epoch in range(n_iter):
        running_loss = 0.0
        count = 0
        for i in range(batch-1):

            # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
            # Tutorial was followed slightly

            # get the inputs; data is a list of [inputs, labels]
            labels = train_labels[batch_size * i : batch_size * (i + 1)]
            inputs = train_set[batch_size * i : batch_size * (i + 1)]

            loss = net.step(inputs, labels)
            
            running_loss += loss

        losses.append(running_loss)
    yhats = np.argmax(net.forward(dev_set).detach().cpu().numpy(), 1)    # yhats = NONE
    
    return losses, yhats, net
