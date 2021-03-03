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

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#CD = '../input/paralel-translation-corpus-in-22-languages/'
#CD = "C:\Assignments\CMPUT466\group project\"
SL = 'EN'
TL = 'NL'
data = pd.read_csv('EN-NL.txt', sep='\t', header = None)[[0, 1]].rename(columns = {0:SL, 1:TL})
#data = pd.read_csv('C:\Assignments\CMPUT466\group project\EN-NL.txt',names = ['EN', 'NL'], sep='\t')

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



    training_data_y = np.append(np.ones((1, 2000)), (np.zeros((1, 2000))))
    testing_data_y = np.append(np.ones((1, len(data['EN'][2000:3000]))), np.zeros((1,len(data['EN'][2000:3000]))))
    count_vectorizer = CountVectorizer(lowercase=True, stop_words=None, max_df=1.0, min_df=1, max_features=None, binary=True)
    transformed_data_x = count_vectorizer.fit_transform(training_data_set + testing_data_set).toarray()

    training_data_x = transformed_data_x[0:4000, :]
    testing_data_x = transformed_data_x[4000:, :]
    return training_data_x, testing_data_x, training_data_y, testing_data_y
def training(smooth_value, X, y):
    classes = np.unique(y)
    class_number = len(classes)
    variable_number = X.shape[1]

    one_list = []
    zero_list = []
    one_possibability_list = []
    zero_possibability_list = []
    for i in range(X.shape[0]):
        if y[i] == 1:
            one_list.append(i)
        elif y[i] == 0:
            zero_list.append(i)

    variable_matrix = np.zeros((2, variable_number))
    for i in one_list:
        for x in range(variable_number):
            if X[i, x] == 1:
                variable_matrix[0, x] += 1
    for i in zero_list:
        for x in range(variable_number):
            if X[i, x] == 1:
                variable_matrix[1, x] += 1

    p_x_y_1 = (variable_matrix[0, :] + smooth_value) / (len(one_list) + 2 * smooth_value)
    p_x_y_0 = (variable_matrix[1, :] + smooth_value) / (len(zero_list) + 2 * smooth_value)

    p_y_1 = len(one_list) / len(y)
    p_y_0 = len(zero_list) / len(y)



    plt.plot(p_x_y_1, variable_matrix[0])
    plt.show()
    return p_x_y_1, p_x_y_0, p_y_1, p_y_0, smooth_value
def testing(p_x_y_1, p_x_y_0, p_y_1, p_y_0, X, smooth):
    X_1 = np.zeros((X.shape[0], X.shape[1]))
    X_2 = np.zeros((X.shape[0], X.shape[1]))

    np.log(p_x_y_1)
    np.log(p_x_y_0)
    for x in range(X.shape[0]):
        X_1[x, :] = np.multiply(p_x_y_1, X[x, :])
        X_2[x, :] = np.multiply(p_x_y_0, X[x, :])


    one_result = []
    zero_result = []

    one_log = []
    zero_log = []
    for i in range(X_1.shape[0]):
        log_list_1 = []
        log_list_0 = []
        for x in range(X_1.shape[1]):
            if X_1[i, x] != 0:
                log_list_1.append(X_1[i, x])
            if X_2[i, x] != 0:
                log_list_0.append(X_2[i, x])


        one_log.append(log_list_1)
        zero_log.append(log_list_0)
    for i in one_log:
        one_result.append(np.log(smooth) + np.log(p_y_1) + sum(i))

    for x in zero_log:
        zero_result.append(np.log(smooth) + np.log(p_y_0) + sum(x))

    pred = []
    for i in range(len(one_result)):
        if one_result[i] > zero_result[i]:
            pred.append(1)
        else:
            pred.append(0)
    return pred

if __name__ == '__main__':
    training_data_set, testing_data_set, y_train, y_test = set_train_and_test_data(data)
    p_x_y_1, p_x_y_0, p_y_1, p_y_0, smooth = training(0.1, training_data_set, y_train)
    prediction = testing(p_x_y_1, p_x_y_0, p_y_1, p_y_0, testing_data_set, smooth)
    print("accuracy = {}".format(np.mean((y_test - prediction) == 0)))
