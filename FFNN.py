""" General outline of my Feed Forward Neural Network"""
""" Is currently a logistic regression model"""
""" currently can test with as many languages as needed"""
""" uses the support of the dataloader class created """
""" has one set of fully connected layers"""
""" last tested with 12 languages, with result of 86.75% accuracy on test set"""


"""
TODO:
    

"""
from FFNN_dataloader import dataloader
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
import torch.optim as optim

# Globals/Parameters (will clean up later)
# parameters
n_train = 1000
n_test = 200
epochs = 5
learning_rate = 0.01
momentum = 0.5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# number of words/features
#n_features = 500
# uses data.shape[1] once data retrieved

loss = nn.MSELoss()

# Objects
class ffNN(nn.Module):
    def __init__(self, n_features, n_languages):
        super(ffNN, self).__init__()

        # will add more layers
        self.l1 = nn.Linear(n_features, n_languages)
        self.softmax = nn.Softmax()

    def forward(self, vector):
        pred = self.l1(vector)
        return self.softmax(pred)


# functions

def train(model, optimizer, train_data, labels):

    # randomize ordering of training set
    indices = np.arange(len(train_data))
    np.random.shuffle(indices)
    loss = nn.MSELoss()
    print('Starting Training...')

    model.train()
    count = 0
    correct = 0

    for i in range(epochs):
        print('epoch: ', i)
        for idx in indices:
            optimizer.zero_grad()
            output = model(torch.from_numpy(train_data[idx]).float())
            output_np = output.detach().numpy()
            if np.argmax(output_np) == np.argmax(labels[idx]):
                correct += 1
            loss_out = loss(output, torch.from_numpy(labels[idx]).float())
            loss_out.backward()
            optimizer.step()
            count += 1
            if count%500 == 0:
                print('accuracy: ', correct, '/', 500, sep='', end='')
                print(' percent: ', (correct/500)*100, '%', sep='')
                count = 0
                correct = 0
          
        # add steps for details on accuracy to keep track


def test(model, test_data, labels, n_languages):
    #todo
    indices = np.arange(len(test_data))
    np.random.shuffle(indices)
    print('Testing...')

    model.eval
    correct = 0
    with torch.no_grad():
        for idx in indices:
            output = model(torch.from_numpy(test_data[idx]).float())
            output_np = output.detach().numpy()
            if np.argmax(output_np) == np.argmax(labels[idx]):
                correct += 1
    print('Test accuracy: ', correct, '/', (n_test*n_languages), sep='')
    print('percent: ', correct/(n_test*n_languages))

def main():
    data_loader = dataloader(n_train, n_test)
    data_loader.prepare_data()

    train_data, train_labels = data_loader.get_train()
    test_data, test_labels = data_loader.get_test()

    n_features = train_data.shape[1]
    n_languages = data_loader.get_n_languages()

    ffnn = ffNN(n_features, n_languages).float()
    optimizer = optim.SGD(ffnn.parameters(), lr=learning_rate,
                      momentum=momentum)
    train(ffnn, optimizer, train_data, train_labels)
    test(ffnn, test_data, test_labels, n_languages)


main()

