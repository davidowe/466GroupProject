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
from FFNN_confusionMatrix import confusionMatrix 
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
import torch.optim as optim

# Globals/Parameters (will clean up later)
# parameters
n_train = 400
n_test = 100
epochs = 5
learning_rate = 0.35
momentum = 0.5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# number of words/features
#n_features = 500
# uses data.shape[1] once data retrieved

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

    n_validation = len(train_data)//epochs

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    loss = nn.MSELoss().to(device)
    print('Starting Training...')

    count = 0
    correct = 0

    for i in range(epochs):
        print('epoch: ', i+1)
        model.train()
        val_indices = indices[:n_validation]
        remove_indices = np.arange(n_validation)
        indices = np.delete(indices, remove_indices)
        for idx in indices:
            optimizer.zero_grad()
            output = model(torch.from_numpy(train_data[idx]).float().to(device))
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
        validation(model, train_data, labels, val_indices)
        indices = np.append(indices, val_indices)
          
        # add steps for details on accuracy to keep track

def validation(model, val_data, val_labels, indices):
    model.eval()
    correct=0
    with torch.no_grad():
        for idx in indices:
            output = model(torch.from_numpy(val_data[idx]).float().to(device))
            output_np = output.detach().numpy()
            if np.argmax(output_np) == np.argmax(val_labels[idx]):
                correct += 1
    print('Validation accuracy: ', correct, '/', (len(indices)), sep='')
    print('percent: ', correct/(len(indices)))

def test(model, test_data, labels, n_languages, confusion):
    #todo
    indices = np.arange(len(test_data))
    np.random.shuffle(indices)
    print('Testing...')
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model.eval()
    correct = 0
    with torch.no_grad():
        for idx in indices:
            output = model(torch.from_numpy(test_data[idx]).float().to(device))
            output_np = output.detach().numpy()
            if confusion.add_confusion(output_np, labels[idx]):
                correct += 1
    print('Test accuracy: ', correct, '/', (n_test*n_languages), sep='')
    print('percent: ', correct/(n_test*n_languages))

def main():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    data_loader = dataloader(n_train, n_test)
    data_loader.prepare_data()

    train_data, train_labels = data_loader.get_train()
    test_data, test_labels = data_loader.get_test()

    n_features = train_data.shape[1]
    print(n_features)
    n_languages = data_loader.get_n_languages()

    ffnn = ffNN(n_features, n_languages).float().to(device)
    #optimizer = optim.Adam(ffnn.parameters(), lr=learning_rate)
    optimizer = optim.SGD(ffnn.parameters(), lr=learning_rate)

    train(ffnn, optimizer, train_data, train_labels)

    confusion = confusionMatrix(data_loader.get_languages())
    test(ffnn, test_data, test_labels, n_languages, confusion)

    confusion.print_confusion()

main()
