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
import nltk

#CD = '../input/paralel-translation-corpus-in-22-languages/'
#CD = "C:\Assignments\CMPUT466\group project\"
SL1 = 'EN'
TL1 = 'NL'
SL2 = 'EN'
TL2 = 'BG'
SL3 = 'EN'
TL3 = 'CS'
SL4 = 'EN'
TL4 = 'DE'
data1 = pd.read_csv('EN-NL.txt', sep='\t', header = None)[[0, 1]].rename(columns = {0:SL1, 1:TL1})
data2 = pd.read_csv('EN-BG.txt', sep='\t', header = None)[[0, 1]].rename(columns = {0:SL2, 1:TL2})
data3 = pd.read_csv('EN-CS.txt', sep='\t', header = None)[[0, 1]].rename(columns = {0:SL3, 1:TL3})
data4 = pd.read_csv('EN-DE.txt', sep='\t', header = None)[[0, 1]].rename(columns = {0:SL4, 1:TL4})
#data = pd.read_csv('C:\Assignments\CMPUT466\group project\EN-NL.txt',names = ['EN', 'NL'], sep='\t')
def generate_biword(input_sentence, n):
    #bigram_result = []
    splited = list(nltk.ngrams(input_sentence.split(' '), n))
    #for gram in splited:
        #bigram_result.append(' '.join(gram))

    #print(splited)
    return splited
def generate_bigram(input_sentence, n):
    split_sentence = input_sentence.split()
    bigram_result = []
    for word in split_sentence:
        splited = list(nltk.ngrams(word, n))
        for gram in splited:
            bigram_result.append(''.join(gram))
    #bigram_result = list(nltk.ngrams(input_sentence, 2))
    #print(bigram_result)
    return bigram_result
def set_train_and_test_data(data1, data2, data3):
    training_data_set = []
    for i in range(0, 2000):
        training_data_set.append(data1['EN'][i])
    for i in range(0, 2000):
        training_data_set.append(data1['NL'][i])
    for i in range(0, 2000):
        training_data_set.append(data2['BG'][i])
    for i in range(0, 2000):
        training_data_set.append(data3['CS'][i])
    testing_data_set = []
    for n in range(2000, 3000):
        testing_data_set.append(data1['EN'][n])
    for n in range(2000, 3000):
        testing_data_set.append(data1['NL'][n])
    for n in range(2000, 3000):
        testing_data_set.append(data2['BG'][n])
    for n in range(2000, 3000):
        testing_data_set.append(data3['CS'][n])


    training_data_y = np.append(np.ones((1, 2000)), np.zeros((1, 2000)))
    training_data_y = np.append(training_data_y, np.full((1, 2000), 2))
    training_data_y = np.append(training_data_y, np.full((1, 2000), 3))
    testing_data_y = np.append(np.ones((1, len(data1['EN'][2000:3000]))), np.zeros((1,len(data1['EN'][2000:3000]))))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data2['BG'][2000:3000])), 2))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data2['BG'][2000:3000])), 3))
    count_vectorizer = CountVectorizer(lowercase=True, stop_words=None, max_df=1.0, min_df=1, max_features=None, binary=False)
    transformed_data_x = count_vectorizer.fit_transform(training_data_set + testing_data_set).toarray()

    training_data_x = transformed_data_x[0:8000, :]
    testing_data_x = transformed_data_x[8000:, :]
    return training_data_x, testing_data_x, training_data_y, testing_data_y

def set_train_and_test_data_word(data1, data2, data3, data4, n):
    training_data_set = []
    for i in range(0, 1000):
        #training_data_set.append(data['EN'][i])
        #training_data_set.append(generate_bigram(data['EN'][i], n))
        training_data_set.append(generate_biword(data1['EN'][i], n))
    for i in range(0, 1000):
        #training_data_set.append(data['NL'][i])
        #training_data_set.append(generate_bigram(data['NL'][i], n))
        training_data_set.append(generate_biword(data1['NL'][i], n))
    for i in range(0, 1000):
        #training_data_set.append(data['NL'][i])
        #training_data_set.append(generate_bigram(data['NL'][i], n))
        training_data_set.append(generate_biword(data2['BG'][i], n))
    for i in range(0, 1000):
        #training_data_set.append(data['NL'][i])
        #training_data_set.append(generate_bigram(data['NL'][i], n))
        training_data_set.append(generate_biword(data3['CS'][i], n))
    for i in range(0, 1000):
        #training_data_set.append(data['NL'][i])
        #training_data_set.append(generate_bigram(data['NL'][i], n))
        training_data_set.append(generate_biword(data4['DE'][i], n))
    testing_data_set = []
    for j in range(1000, 2000):
        #testing_data_set.append(data['EN'][j])
        #testing_data_set.append(generate_bigram(data['EN'][j], n))
        testing_data_set.append(generate_biword(data1['EN'][j], n))
    for j in range(1000, 2000):
        #testing_data_set.append(data['NL'][j])
        #testing_data_set.append(generate_bigram(data['NL'][j], n))
        testing_data_set.append(generate_biword(data1['NL'][j], n))
    for j in range(1000, 2000):
        #testing_data_set.append(data['NL'][j])
        #testing_data_set.append(generate_bigram(data['NL'][j], n))
        testing_data_set.append(generate_biword(data2['BG'][j], n))
    for j in range(1000, 2000):
        #testing_data_set.append(data['NL'][j])
        #testing_data_set.append(generate_bigram(data['NL'][j], n))
        testing_data_set.append(generate_biword(data3['CS'][j], n))
    for j in range(1000, 2000):
        #testing_data_set.append(data['NL'][j])
        #testing_data_set.append(generate_bigram(data['NL'][j], n))
        testing_data_set.append(generate_biword(data4['DE'][j], n))

    training_data_y = np.append(np.ones((1, 1000)), (np.zeros((1, 1000))))
    training_data_y = np.append(training_data_y, np.full((1, 1000), 2))
    training_data_y = np.append(training_data_y, np.full((1, 1000), 3))
    training_data_y = np.append(training_data_y, np.full((1, 1000), 4))
    testing_data_y = np.append(np.ones((1, len(data1['EN'][1000:2000]))), np.zeros((1,len(data1['NL'][1000:2000]))))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data2['BG'][1000:2000])), 2))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data3['CS'][1000:2000])), 3))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data4['DE'][1000:2000])), 4))
    count_vectorizer = CountVectorizer(lowercase=True, stop_words=None, analyzer=lambda x:x, max_df=1.0, min_df=1, max_features=None, binary=True)
    #count_vectorizer = CountVectorizer(lowercase=True, stop_words=None, max_df=1.0, min_df=1, max_features=None, binary=True)
    transformed_data_x = count_vectorizer.fit_transform(training_data_set + testing_data_set).toarray()

    #print(transformed_data_x[:, 0])
    training_data_x = transformed_data_x[0:5000, :]
    testing_data_x = transformed_data_x[5000:, :]
    return training_data_x, testing_data_x, training_data_y, testing_data_y
def set_train_and_test_data_char(data1, data2, data3, data4, n):
    training_data_set = []
    for i in range(0, 1000):
        #training_data_set.append(data['EN'][i])
        training_data_set.append(generate_bigram(data1['EN'][i], n))
        #training_data_set.append(generate_biword(data['EN'][i], n))
    for i in range(0, 1000):
        #training_data_set.append(data['NL'][i])
        training_data_set.append(generate_bigram(data1['NL'][i], n))
        #training_data_set.append(generate_biword(data['NL'][i], n))
    for i in range(0, 1000):
        #training_data_set.append(data['NL'][i])
        training_data_set.append(generate_bigram(data2['BG'][i], n))
        #training_data_set.append(generate_biword(data['NL'][i], n))
    for i in range(0, 1000):
        #training_data_set.append(data['NL'][i])
        training_data_set.append(generate_bigram(data3['CS'][i], n))
        #training_data_set.append(generate_biword(data['NL'][i], n))
    for i in range(0, 1000):
        #training_data_set.append(data['NL'][i])
        training_data_set.append(generate_bigram(data4['DE'][i], n))
        #training_data_set.append(generate_biword(data['NL'][i], n))
    testing_data_set = []
    for j in range(1000, 2000):
        #testing_data_set.append(data['EN'][j])
        testing_data_set.append(generate_bigram(data1['EN'][j], n))
        #testing_data_set.append(generate_biword(data['EN'][j], n))
    for j in range(1000, 2000):
        #testing_data_set.append(data['NL'][j])
        testing_data_set.append(generate_bigram(data1['NL'][j], n))
        #testing_data_set.append(generate_biword(data['NL'][j], n))
    for j in range(1000, 2000):
        #testing_data_set.append(data['NL'][j])
        testing_data_set.append(generate_bigram(data2['BG'][j], n))
        #testing_data_set.append(generate_biword(data['NL'][j], n))
    for j in range(1000, 2000):
        #testing_data_set.append(data['NL'][j])
        testing_data_set.append(generate_bigram(data3['CS'][j], n))
        #testing_data_set.append(generate_biword(data['NL'][j], n))
    for j in range(1000, 2000):
        #testing_data_set.append(data['NL'][j])
        testing_data_set.append(generate_bigram(data4['DE'][j], n))
        #testing_data_set.append(generate_biword(data['NL'][j], n))


    training_data_y = np.append(np.ones((1, 1000)), (np.zeros((1, 1000))))
    training_data_y = np.append(training_data_y, np.full((1, 1000), 2))
    training_data_y = np.append(training_data_y, np.full((1, 1000), 3))
    training_data_y = np.append(training_data_y, np.full((1, 1000), 4))
    testing_data_y = np.append(np.ones((1, len(data1['EN'][1000:2000]))), np.zeros((1,len(data1['NL'][1000:2000]))))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data2['BG'][1000:2000])), 2))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data3['CS'][1000:2000])), 3))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data4['DE'][1000:2000])), 4))
    count_vectorizer = CountVectorizer(lowercase=True, stop_words=None, analyzer=lambda x:x, max_df=1.0, min_df=1, max_features=None, binary=True)
    #count_vectorizer = CountVectorizer(lowercase=True, stop_words=None, max_df=1.0, min_df=1, max_features=None, binary=True)
    transformed_data_x = count_vectorizer.fit_transform(training_data_set + testing_data_set).toarray()

    #print(transformed_data_x[:, 0])
    training_data_x = transformed_data_x[0:5000, :]
    testing_data_x = transformed_data_x[5000:, :]
    return training_data_x, testing_data_x, training_data_y, testing_data_y

def training(smooth_value, X, y):
    classes = np.unique(y)
    class_number = len(classes)
    variable_number = X.shape[1]

    one_list = []
    zero_list = []
    two_list = []
    three_list = []
    four_list = []
    one_possibability_list = []
    zero_possibability_list = []
    for i in range(X.shape[0]):

        if y[i] == 1:
            one_list.append(i)
        elif y[i] == 0:
            zero_list.append(i)
        elif y[i] == 2:
            two_list.append(i)
        elif y[i] == 3:
            three_list.append(i)
        elif y[i] == 4:
            four_list.append(i)

    variable_matrix = np.zeros((5, variable_number))
    for i in one_list:
        for x in range(variable_number):
            if X[i, x] == 1:
                variable_matrix[0, x] += 1
    for i in zero_list:
        for x in range(variable_number):
            if X[i, x] == 1:
                variable_matrix[1, x] += 1
    for i in two_list:
        for x in range(variable_number):
            if X[i, x] == 1:
                variable_matrix[2, x] += 1
    for i in three_list:
        for x in range(variable_number):
            if X[i, x] == 1:
                variable_matrix[3, x] += 1
    for i in four_list:
        for x in range(variable_number):
            if X[i, x] == 1:
                variable_matrix[4, x] += 1

    p_x_y_1 = (variable_matrix[0, :] + smooth_value) / (len(one_list) + 5 * smooth_value)
    p_x_y_0 = (variable_matrix[1, :] + smooth_value) / (len(zero_list) + 5 * smooth_value)
    p_x_y_2 = (variable_matrix[2, :] + smooth_value) / (len(two_list) + 5 * smooth_value)
    p_x_y_3 = (variable_matrix[3, :] + smooth_value) / (len(three_list) + 5 * smooth_value)
    p_x_y_4 = (variable_matrix[4, :] + smooth_value) / (len(four_list) + 5 * smooth_value)
    p_y_1 = len(one_list) / len(y)
    p_y_0 = len(zero_list) / len(y)
    p_y_2 = len(two_list) / len(y)
    p_y_3 = len(three_list) / len(y)
    p_y_4 = len(four_list) / len(y)

    return p_x_y_4, p_x_y_3, p_x_y_2, p_x_y_1, p_x_y_0, p_y_4, p_y_3, p_y_2, p_y_1, p_y_0, smooth_value
def test_model(p_x_y_4, p_x_y_3, p_x_y_2, p_x_y_1, p_x_y_0, p_y_4, p_y_3, p_y_2, p_y_1, p_y_0, X, smooth):
    X_1 = np.zeros((X.shape[0], X.shape[1]))
    X_2 = np.zeros((X.shape[0], X.shape[1]))
    X_3 = np.zeros((X.shape[0], X.shape[1]))
    X_4 = np.zeros((X.shape[0], X.shape[1]))
    X_5 = np.zeros((X.shape[0], X.shape[1]))
    np.log(p_x_y_1)
    np.log(p_x_y_0)
    np.log(p_x_y_2)
    np.log(p_x_y_3)
    np.log(p_x_y_4)
    for x in range(X.shape[0]):
        X_1[x, :] = np.multiply(p_x_y_1, X[x, :])
        X_2[x, :] = np.multiply(p_x_y_0, X[x, :])
        X_3[x, :] = np.multiply(p_x_y_2, X[x, :])
        X_4[x, :] = np.multiply(p_x_y_3, X[x, :])
        X_5[x, :] = np.multiply(p_x_y_4, X[x, :])
    one_result = []
    zero_result = []
    two_result = []
    three_result = []
    four_result = []
    one_log = []
    zero_log = []
    two_log = []
    three_log = []
    four_log = []
    for i in range(X_1.shape[0]):
        log_list_1 = []
        log_list_0 = []
        log_list_2 = []
        log_list_3 = []
        log_list_4 = []
        for x in range(X_1.shape[1]):
            if X_1[i, x] != 0:
                log_list_1.append(X_1[i, x])
            if X_2[i, x] != 0:
                log_list_0.append(X_2[i, x])
            if X_3[i, x] != 0:
                log_list_2.append(X_3[i, x])
            if X_4[i, x] != 0:
                log_list_3.append(X_4[i, x])
            if X_5[i, x] != 0:
                log_list_4.append(X_5[i, x])

        one_log.append(log_list_1)
        zero_log.append(log_list_0)
        two_log.append(log_list_2)
        three_log.append(log_list_3)
        four_log.append(log_list_4)
    for i in one_log:
        one_result.append(np.log(smooth) + np.log(p_y_1) + sum(i))

    for x in zero_log:
        zero_result.append(np.log(smooth) + np.log(p_y_0) + sum(x))

    for y in two_log:
        two_result.append(np.log(smooth) + np.log(p_y_2) + sum(y))

    for z in three_log:
        three_result.append(np.log(smooth) + np.log(p_y_3) + sum(z))

    for a in four_log:
        four_result.append(np.log(smooth) + np.log(p_y_4) + sum(a))

    pred = []

    for i in range(len(one_result)):

        answer = max([zero_result[i], one_result[i], two_result[i], three_result[i], four_result[i]])
        if answer == zero_result[i]:
            pred.append(0)
        elif answer == one_result[i]:
            pred.append(1)
        elif answer == two_result[i]:
            pred.append(2)
        elif answer == three_result[i]:
            pred.append(3)
        elif answer == four_result[i]:
            pred.append(4)

    return pred

#if __name__ == '__main__':
    #training_data_set, testing_data_set, y_train, y_test = set_train_and_test_data(data1, data2)
    #p_x_y_2, p_x_y_1, p_x_y_0, p_y_2, p_y_1, p_y_0, smooth_value = training(0.1, training_data_set, y_train)
    #prediction = test_model(p_x_y_2, p_x_y_1, p_x_y_0, p_y_2, p_y_1, p_y_0, testing_data_set, smooth_value)
    #print("accuracy = {}".format(np.mean((y_test - prediction) == 0)))

def processing_word(n):
    training_data_set, testing_data_set, y_train, y_test = set_train_and_test_data_word(data1, data2, data3, data4, n)
    p_x_y_4, p_x_y_3, p_x_y_2, p_x_y_1, p_x_y_0, p_y_4, p_y_3, p_y_2, p_y_1, p_y_0, smooth_value = training(0.1, training_data_set, y_train)
    prediction = test_model(p_x_y_4, p_x_y_3, p_x_y_2, p_x_y_1, p_x_y_0, p_y_4, p_y_3, p_y_2, p_y_1, p_y_0, testing_data_set, smooth_value)
    print('word accuracy = {}'.format(np.mean((y_test - prediction) == 0)))
    return np.mean((y_test - prediction) == 0)
def processing_char(n):
    training_data_set, testing_data_set, y_train, y_test = set_train_and_test_data_char(data1, data2, data3, data4, n)
    p_x_y_4, p_x_y_3, p_x_y_2, p_x_y_1, p_x_y_0, p_y_4, p_y_3, p_y_2, p_y_1, p_y_0, smooth_value = training(0.1, training_data_set, y_train)
    prediction = test_model(p_x_y_4, p_x_y_3, p_x_y_2, p_x_y_1, p_x_y_0, p_y_4, p_y_3, p_y_2, p_y_1, p_y_0, testing_data_set, smooth_value)
    print('char accuracy = {}'.format(np.mean((y_test - prediction) == 0)))
    return np.mean((y_test - prediction) == 0)
def draw_diagram():
    n = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    prediction_word = [processing_word(1), processing_word(2), processing_word(3), processing_word(4), processing_word(5), processing_word(6), processing_word(7), processing_word(8), processing_word(9), processing_word(10)]
    prediction_char = [processing_char(1), processing_char(2), processing_char(3), processing_char(4), processing_char(5), processing_char(6), processing_char(7), processing_char(8), processing_char(9), processing_char(10)]
    plt.plot(n, prediction_word, label="word ngrams")
    plt.plot(n, prediction_char, label="character ngrams")
    plt.xlabel('x - n value')
    plt.ylabel('y - accuracy')
    plt.legend()
    plt.show()
if __name__ == '__main__':
    draw_diagram()