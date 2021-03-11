import pandas as pd
import os
from xml.dom import minidom
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import math

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

SL = 'EN'
TL = 'NL'
data = pd.read_csv('EN-NL.txt', sep='\t', header = None)[[0, 1]].rename(columns = {0:SL, 1:TL})
# print(data)

def set_train_and_test_data(data):
    training_data_set = []
    for i in range(0, 2000):
        training_data_set.append(data['EN'][i])
    for i in range(0, 2000):
        training_data_set.append(data['NL'][i])
    testing_data_set = []
    for n in range(2000, 3000):
        testing_data_set.append(data['EN'][n])
    for n in range(2000, 3000):
        testing_data_set.append(data['NL'][n])

    training_data_y = np.append(np.ones((1, 2000)), (np.negative(np.ones((1, 2000)))))
    testing_data_y = np.append(np.ones((1, len(data['EN'][2000:3000]))), np.negative(np.ones((1,len(data['EN'][2000:3000])))))
    count_vectorizer = CountVectorizer(lowercase=True, stop_words=None, max_df=1.0, min_df=1, max_features=None, binary=True)
    transformed_data_x = count_vectorizer.fit_transform(training_data_set + testing_data_set).toarray()

    training_data_x = transformed_data_x[0:4000, :]
    testing_data_x = transformed_data_x[4000:, :]
    return training_data_x, testing_data_x, training_data_y, testing_data_y

# Inspired from https://www.robots.ox.ac.uk/~az/lectures/ml/lect2.pdf
# Uses sub-gradient descent for weights
# Assumes bias of 0
def training(X, y, learning_rate = 0.001, lambda_param = 0.02, epochs = 500):
    weights = np.zeros(X.shape[1])
    for epoch in range(epochs):
        for idx, instance in enumerate(X):
            # Maximize the margin by minimizing W
            if y[idx] * (np.dot(instance, weights)) < 1: # If it matches, then update weights
                weights -= learning_rate * (lambda_param * weights - np.dot(instance, y[idx]))
            else: # Else, update weights and bias accordingly
                weights -= learning_rate * (lambda_param * weights) # Multiply by gradient, add to 0 
    return weights

def predict(Weights , X_dataset):
    return np.sign((np.dot(X_dataset, Weights)))

if __name__ == '__main__':
    training_data_set, testing_data_set, y_train, y_test = set_train_and_test_data(data)
    weights = training(training_data_set, y_train)
    print(accuracy_score(y_test,predict(weights,testing_data_set)))