""" General outline of my Feed Forward Neural Network"""
""" will end up being a logistic regression model"""
""" may implement my own model from scratch rather than using pytorch"""
""" using this as a guideline as I have experience with pytorch"""

import torch
import torch.nn as nn


# Import data
""" being done by another group member"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# number of words/features
n_features = 500
n_classes = 8

class ffNN(nn.Module):
    def __init__(self):
        super(ffNN, self).__init__()

        # will add more layers
        self.l1 = nn.linear(n_features, n_classes)
        self.softmax = nn.softmax()

    def forward(self, vector):
        pred = self.l1(vector)

def train(model):
    #todo 
    '''
    ffnn.forward(vector)'''

def test(model):
    #todo
    '''
    ffnn.forward(vector)'''

def main():
    ffnn = ffNN()
    train(ffnn)


