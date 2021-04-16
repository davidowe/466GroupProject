import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
import torch.optim as optim
from FFNN_dataloader import dataloader
from FFNN_confusionMatrix import confusionMatrix 

# Globals/Parameters
# parameters
n_train = 800
n_test = 200
epochs = 5
learning_rate = 0.9
momentum = 0.95
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Objects
# FeedForwardNeural Network 
class ffNN(nn.Module):
    def __init__(self, n_features, n_languages):
        super(ffNN, self).__init__()
        self.l1 = nn.Linear(n_features, n_languages)
        self.softmax = nn.Softmax()

    def forward(self, vector):
        pred = self.l1(vector)
        return self.softmax(pred)


# Functions

def train(model, optimizer, train_data, labels):
    # train the feed forward neural network
    # print the accuracy metrics for each training and validation

    # model - the FFNN object
    # optimizer - the optimizer object initialized
    # train_data - vectorized training data
    # labels - labels corresponding to the training data

    # randomize ordering of training set
    indices = np.arange(len(train_data))
    np.random.shuffle(indices)

    n_validation = len(train_data)//epochs

    # gives ability to train by gpu
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # initialize loss function to be used
    loss = nn.MSELoss().to(device)
    print('Starting Training...')

    count = 0
    correct = 0

    # loop through training data
    for i in range(epochs):
        print('epoch: ', i+1)
        model.train()

        # remove validation data for k-fold validation from training dataset
        val_indices = indices[:n_validation]
        remove_indices = np.arange(n_validation)
        indices = np.delete(indices, remove_indices)

        # perform training using training dataset
        # uses stochasic gradient descent
        # tracks accuracy measures throughout training
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

        # perform validataion on the set aside validation dataset
        validation(model, train_data, labels, val_indices)
        indices = np.append(indices, val_indices)


def validation(model, val_data, val_labels, indices):
    # test model using validation dataset and reports accuracy measures

    # model - the FFNN 
    # val_data - the training data
    # val_labels - the labels corresponding to the training data
    # indices - indices of training data which are to be used for 
    # validation.

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
    # test model with separate test dataset
    # prints accuracy metrics after testing

    # model - the FFNN
    # test data - the vectorized test data
    # labels - the associated labels
    # n-languages - number of languages for confusion matrix
    # confusion - the confusion matrix object

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
    # prepare data 
    # obtain the training and testing datasets
    # initialize FFNN object and optimizer object
    # train model
    # initialize confusion matrix
    # test model
    # print confusion matrix
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    data_loader = dataloader(n_train, n_test)
    data_loader.prepare_data()

    train_data, train_labels = data_loader.get_train()
    test_data, test_labels = data_loader.get_test()

    n_features = train_data.shape[1]
    print('# input features:')
    print(n_features)
    n_languages = data_loader.get_n_languages()

    ffnn = ffNN(n_features, n_languages).float().to(device)

    optimizer = optim.SGD(ffnn.parameters(), lr=learning_rate, momentum=momentum)

    train(ffnn, optimizer, train_data, train_labels)

    confusion = confusionMatrix(data_loader.get_languages())
    test(ffnn, test_data, test_labels, n_languages, confusion)

    confusion.print_confusion()

main()

