""" General outline of my Feed Forward Neural Network"""
""" Is currently a logistic regression model"""
""" currently only tests between english and dutch"""\

"""
TODO:
    Clean up code to eliminate global variables
    Add comments to explain what I am doing
    Test with varying model/layers
    Test with varying optimizers, loss functions, activation functions
    Hyperparameter tuning

"""

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from google.colab import drive
import os
from sklearn.feature_extraction.text import CountVectorizer
import torch.optim as optim

drive.mount('/content/drive/',force_remount=True)
root_path = 'drive/MyDrive/CMPUT466/'
#/content/drive/MyDrive/CMPUT466/EN-NL.txt

# Globals/Parameters (will clean up later)

# Import data
""" being done by another group member"""
#with open("EN-NL.txt") as ennl:
#    sentences = ennl.readlines()
SL = 'EN'
TL = 'NL'
data = pd.read_csv('EN -NL.txt', sep='\t', header = None)[[0, 1]].rename(columns = {0:SL, 1:TL})

# parameters
n_train = 9000
n_languages = 2
n_test = 1000
n_sentences = data[SL].shape[0]
print(n_sentences)
epochs = 1
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
def compile_data(data, n_train, n_test):
    EN_data = data[SL][:(n_train+n_test)]
    NL_data = data[TL][:(n_train+n_test)]
    combined = np.stack((EN_data, NL_data)).flatten()
    return combined

def vectorize_data(data_in):
    # input data should be a single array of sentences from all languages
    count_vectorizer = CountVectorizer(lowercase=True, stop_words=None, max_df=1.0, min_df=1, max_features=None, binary=True)
    return count_vectorizer.fit_transform(data_in).toarray()

def split_train(data, n_train):
    n_words = data.shape[1]
    train_data = np.zeros((n_train*n_languages, n_words))
    train_labels = np.zeros((n_train*n_languages, n_languages))

    for i in range(n_languages):
        start = i * (n_train+n_test)
        end = start + n_train
        i_start = i*n_train
        i_end = i_start+n_train
        train_data[i_start:i_end] = data[start:end]

        labels = np.zeros((n_train, n_languages))
        labels[:, i] = 1

        l_start = n_train*i
        l_end = n_train*(i+1)

        train_labels[l_start:l_end, :] = labels

    return train_data, train_labels

def split_test(data, n_train, n_test):
    n_words = data.shape[1]
    test_data = np.zeros((n_test*n_languages, n_words))
    test_labels = np.zeros((n_test*n_languages, n_languages))

    for i in range(n_languages):
        start = (n_train+n_test)*i + n_train
        end = start + n_test
        i_start = i*n_test
        i_end = i_start+n_test
        test_data[i_start:i_end] = data[start:end]

        labels = np.zeros((n_test, n_languages))
        labels[:, i] = 1

        l_start = n_test*i
        l_end = n_test*(i+1)

        test_labels[l_start:l_end, :] = labels

    return test_data, test_labels
    

def train(model, optimizer, train_data, labels):

    # randomize ordering of training set
    indices = np.arange(len(train_data))
    np.random.shuffle(indices)
    loss = nn.MSELoss()

    model.train()
    count = 0
    correct = 0
    for i in range(epochs):
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
            if count%100 == 0:
                print('accuracy: ', correct, '/', 100, sep='')
                count = 0
                correct = 0
          
        # add steps for details on accuracy to keep track


def test(model, test_data, labels):
    #todo
    indices = np.arange(len(test_data))
    np.random.shuffle(indices)

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
    combined_data = compile_data(data, n_train, n_test)
    vectorized_data = vectorize_data(combined_data)
    train_data, train_labels = split_train(vectorized_data, n_train)
    test_data, test_labels = split_test(vectorized_data, n_train, n_test)

    n_features = train_data.shape[1]

    ffnn = ffNN(n_features, n_languages).float()
    optimizer = optim.SGD(ffnn.parameters(), lr=learning_rate,
                      momentum=momentum)
    train(ffnn, optimizer, train_data, train_labels)
    test(ffnn, test_data, test_labels)
    print(ffnn.l1.weight.shape)


main()

