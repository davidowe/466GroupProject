"""
EN-NL.txt is too large to put on GitHub
"""

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
import pickle
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from io import open
import itertools
import linecache
import random
from collections import Counter



def set_train_and_test_data(data,lang_pair):
    training_data_set = []
    for i in range(0, 4000):
        training_data_set.append(data[lang_pair[0]][i])
    for i in range(0, 4000):
        training_data_set.append(data[lang_pair[1]][i])
    testing_data_set = []
    for n in range(49000, 50000):
        testing_data_set.append(data[lang_pair[0]][n])
    for n in range(49000, 50000):
        testing_data_set.append(data[lang_pair[1]][n])

    training_data_y = np.append(np.ones((1, 4000)), (np.negative(np.ones((1, 4000)))))
    testing_data_y = np.append(np.ones((1, 1000)), np.negative(np.ones((1,1000))))
    count_vectorizer = CountVectorizer(lowercase=True, stop_words=None, max_df=1.0, min_df=1, max_features=None, binary=True)
    transformed_data_x = count_vectorizer.fit_transform(training_data_set).toarray()
    transformed_data_test_x = count_vectorizer.transform(testing_data_set).toarray()

    return transformed_data_x, transformed_data_test_x, training_data_y, testing_data_y, count_vectorizer

# Inspired from https://www.robots.ox.ac.uk/~az/lectures/ml/lect2.pdf
# Uses sub-gradient descent for weights
# Assumes bias of 0
def training(X, y, learning_rate = 0.0001, lambda_param = 0.02, epochs = 5):
    # print(f"Training with weight of size {X.shape[1]}")
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
    prediction = np.sign((np.dot(X_dataset, Weights)))
    prediction[prediction == 0] = random.choice([-1,1])
    return prediction

# From Smitha Dinesh Semwal at https://www.geeksforgeeks.org/python-find-most-common-element-in-each-column-in-a-2d-list/
def mostCommon(lst):
    return [Counter(col).most_common(1)[0][0] for col in zip(*lst)]
    
if __name__ == '__main__':
    lang_list = ["BG","CS","DA","DE","EL","EN","ES","ET","FI","FR","HR","HU","IT","LT","LV","MT","NL","PL","PT","RO","SK","SL","SV"]
    lang_pair_list = [sorted(x) for x in list(itertools.combinations(lang_list,2))]
    data_dict = {}
    
    for lang in lang_list:
        data_dict[lang] = list(filter(None,[x[:-3] for x in open("data/" + lang + ".tsv", 'r',encoding='utf-8').read().split("\n")]))[1:]


    x_file_test = list(filter(None,[x[:-3] for x in open("testing_VERY_FEW.tsv", 'r',encoding='utf-8').read().split("\n")]))
    y_file_test = list(filter(None,[x[-2:] for x in open("testing_VERY_FEW.tsv", 'r',encoding='utf-8').read().split("\n")]))
    print(f"The y_file_test is {y_file_test}")
    result = []
    i = 1
    for lang_pair in lang_pair_list:
        print(f"The model is {lang_pair[0]} {lang_pair[1]} {i}/{len(lang_pair_list)}")
        i+=1
        training_data_set, testing_data_set, y_train, y_test,countvectorizer = set_train_and_test_data(data_dict,lang_pair)
        weight = training(training_data_set,y_train)
        print(f"\tMy test set reports accuracy of {accuracy_score(y_test,predict(weight,testing_data_set))}")
        prediction = predict(weight,countvectorizer.transform(x_file_test).toarray())
        prediction = np.where(prediction == 1, lang_pair[0], lang_pair[1])

        result.append(prediction)
    print(f"\tI predict: {mostCommon(result)}")
    print(accuracy_score(y_file_test,mostCommon(result)))
    print(precision_score(y_file_test,mostCommon(result),average="weighted"))
    print(f1_score(y_file_test,mostCommon(result),average="weighted"))
    print(recall_score(y_file_test,mostCommon(result),average="weighted"))
    print(confusion_matrix(y_file_test,mostCommon(result)))

    

