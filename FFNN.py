""" General outline of my Feed Forward Neural Network"""
""" will end up being a logistic regression model"""
""" may implement my own model from scratch rather than using pytorch"""
""" using this as a guideline as I have experience with pytorch"""

import torch
import torch.nn as nn
import numpy as np


# Import data
""" being done by another group member"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# need to convert data into pytorch Tensor data

# number of words/features
n_features = 500
# uses data.shape[1] once data retrieved
n_classes = 8

loss = nn.MSELoss()

class ffNN(nn.Module):
    def __init__(self):
        super(ffNN, self).__init__()

        # will add more layers
        self.l1 = nn.linear(n_features, n_classes)
        self.softmax = nn.softmax()

    def forward(self, vector):
        pred = self.l1(vector)
        return self.softmax(pred)

def train(model, data):
    model.train()
    ffnn.forward(vector)
    #target = label
    # tbd based on how data is organized

    for idx, vector in enumerate(data):
        optimizer.zero_grad()
        output = model(vector)
        loss = nn.MSELoss(output, target)
        loss.backward()
        optimizer.step()
        # add steps for details on accuracy to keep track


def test(model):
    #todo
    '''
    ffnn.forward(vector)'''
    model.eval
    for idx, vector in enumerate(data):
        model.no

def main():
    ffnn = ffNN()
    train(ffnn)

#main()
