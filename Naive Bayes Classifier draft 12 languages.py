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
language_list = ['EN','NL','BG','CS','DE','DA','EL','ES','ET','FI','FR','GA']
total_correct_assignment_word = 0
total_correct_assignment_char = 0


language_amount = 12
train_amount = 3000
test_amount = 100
total_amount = train_amount + test_amount

SL1 = 'EN'
TL1 = 'NL'
SL2 = 'EN'
TL2 = 'BG'
SL3 = 'EN'
TL3 = 'CS'
SL4 = 'EN'
TL4 = 'DE'
SL5 = 'EN'
TL5 = 'DA'

SL6 = 'EN'
TL6 = 'EL'
SL7 = 'EN'
TL7 = 'ES'
SL8 = 'EN'
TL8 = 'ET'
SL9 = 'EN'
TL9 = 'FI'
SL10 = 'EN'
TL10 = 'FR'
SL11 = 'EN'
TL11 = 'GA'
data1 = pd.read_csv('EN-NL.txt', sep='\t', header = None)[[0, 1]].rename(columns = {0:SL1, 1:TL1})
data2 = pd.read_csv('EN-BG.txt', sep='\t', header = None)[[0, 1]].rename(columns = {0:SL2, 1:TL2})
data3 = pd.read_csv('EN-CS.txt', sep='\t', header = None)[[0, 1]].rename(columns = {0:SL3, 1:TL3})
data4 = pd.read_csv('EN-DE.txt', sep='\t', header = None)[[0, 1]].rename(columns = {0:SL4, 1:TL4})
data5 = pd.read_csv('EN-DA.txt', sep='\t', header = None)[[0, 1]].rename(columns = {0:SL5, 1:TL5})

data6 = pd.read_csv('EN-EL.txt', sep='\t', header = None)[[0, 1]].rename(columns = {0:SL6, 1:TL6})
data7 = pd.read_csv('EN-ES.txt', sep='\t', header = None)[[0, 1]].rename(columns = {0:SL7, 1:TL7})
data8 = pd.read_csv('EN-ET.txt', sep='\t', header = None)[[0, 1]].rename(columns = {0:SL8, 1:TL8})
data9 = pd.read_csv('EN-FI.txt', sep='\t', header = None)[[0, 1]].rename(columns = {0:SL9, 1:TL9})
data10 = pd.read_csv('EN-FR.txt', sep='\t', header = None)[[0, 1]].rename(columns = {0:SL10, 1:TL10})
data11 = pd.read_csv('EN-GA.txt', sep='\t', header = None)[[0, 1]].rename(columns = {0:SL11, 1:TL11})
#data = pd.read_csv('C:\Assignments\CMPUT466\group project\EN-NL.txt',names = ['EN', 'NL'], sep='\t')

F1_dictionary_word = {}
F1_dictionary_char = {}
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
def set_train_and_test_data(data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11):
    training_data_set = []
    for i in range(0, train_amount):
        training_data_set.append(data1['EN'][i])
    for i in range(0, train_amount):
        training_data_set.append(data1['NL'][i])
    for i in range(0, train_amount):
        training_data_set.append(data2['BG'][i])
    for i in range(0, train_amount):
        training_data_set.append(data3['CS'][i])
    for i in range(0, train_amount):

        training_data_set.append(data4['DE'][i])
    for i in range(0, train_amount):

        training_data_set.append(data5['DA'][i])
    for i in range(0, train_amount):

        training_data_set.append(data6['EL'][i])
    for i in range(0, train_amount):

        training_data_set.append(data7['ES'][i])
    for i in range(0, train_amount):

        training_data_set.append(data8['ET'][i])
    for i in range(0, train_amount):

        training_data_set.append(data9['FI'][i])
    for i in range(0, train_amount):

        training_data_set.append(data10['FR'][i])
    for i in range(0, train_amount):

        training_data_set.append(data11['GA'][i])
    testing_data_set = []
    for n in range(train_amount, total_amount):
        testing_data_set.append(data1['EN'][n])
    for n in range(train_amount, total_amount):
        testing_data_set.append(data1['NL'][n])
    for n in range(train_amount, total_amount):
        testing_data_set.append(data2['BG'][n])
    for n in range(train_amount, total_amount):
        testing_data_set.append(data3['CS'][n])
    for n in range(train_amount, total_amount):

        testing_data_set.append(data4['DE'][n])
    for n in range(train_amount, total_amount):

        testing_data_set.append(data5['DA'][n])
    for n in range(train_amount, total_amount):

        testing_data_set.append(data6['EL'][n])
    for n in range(train_amount, total_amount):

        testing_data_set.append(data7['ES'][n])
    for n in range(train_amount, total_amount):

        testing_data_set.append(data8['ET'][n])
    for n in range(train_amount, total_amount):

        testing_data_set.append(data9['FI'][n])
    for n in range(train_amount, total_amount):

        testing_data_set.append(data10['FR'][n])
    for n in range(train_amount, total_amount):

        testing_data_set.append(data11['GA'][n])

    training_data_y = np.append(np.ones((1, train_amount)), (np.zeros((1, train_amount))))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 2))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 3))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 4))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 5))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 6))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 7))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 8))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 9))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 10))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 11))
    testing_data_y = np.append(np.ones((1, len(data1['EN'][train_amount:total_amount]))),
                               np.zeros((1, len(data1['NL'][train_amount:total_amount]))))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data2['BG'][train_amount:total_amount])), 2))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data3['CS'][train_amount:total_amount])), 3))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data4['DE'][train_amount:total_amount])), 4))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data5['DA'][train_amount:total_amount])), 5))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data6['EL'][train_amount:total_amount])), 6))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data7['ES'][train_amount:total_amount])), 7))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data8['ET'][train_amount:total_amount])), 8))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data9['FI'][train_amount:total_amount])), 9))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data10['FR'][train_amount:total_amount])), 10))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data11['GA'][train_amount:total_amount])), 11))
    count_vectorizer = CountVectorizer(lowercase=True, stop_words=None, analyzer=lambda x: x, max_df=1, min_df=1,
                                       max_features=None, binary=True)
    # count_vectorizer = CountVectorizer(lowercase=True, stop_words=None, max_df=1.0, min_df=1, max_features=None, binary=True)
    transformed_data_x = count_vectorizer.fit_transform(training_data_set + testing_data_set).toarray()

    # print(transformed_data_x[:, 0])
    total_train_amount = train_amount * language_amount
    training_data_x = transformed_data_x[0:total_train_amount, :]
    testing_data_x = transformed_data_x[total_train_amount:, :]
    return training_data_x, testing_data_x, training_data_y, testing_data_y

def set_train_and_test_data_word(data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, n):

    training_data_set = []
    for i in range(0, train_amount):

        training_data_set.append(generate_biword(data1['EN'][i], n))
    for i in range(0, train_amount):

        training_data_set.append(generate_biword(data1['NL'][i], n))
    for i in range(0, train_amount):

        training_data_set.append(generate_biword(data2['BG'][i], n))
    for i in range(0, train_amount):

        training_data_set.append(generate_biword(data3['CS'][i], n))
    for i in range(0, train_amount):

        training_data_set.append(generate_biword(data4['DE'][i], n))
    for i in range(0, train_amount):

        training_data_set.append(generate_biword(data5['DA'][i], n))
    for i in range(0, train_amount):

        training_data_set.append(generate_biword(data6['EL'][i], n))
    for i in range(0, train_amount):

        training_data_set.append(generate_biword(data7['ES'][i], n))
    for i in range(0, train_amount):

        training_data_set.append(generate_biword(data8['ET'][i], n))
    for i in range(0, train_amount):

        training_data_set.append(generate_biword(data9['FI'][i], n))
    for i in range(0, train_amount):

        training_data_set.append(generate_biword(data10['FR'][i], n))
    for i in range(0, train_amount):

        training_data_set.append(generate_biword(data11['GA'][i], n))
    testing_data_set = []
    for j in range(train_amount, total_amount):

        testing_data_set.append(generate_biword(data1['EN'][j], n))
    for j in range(train_amount, total_amount):

        testing_data_set.append(generate_biword(data1['NL'][j], n))
    for j in range(train_amount, total_amount):

        testing_data_set.append(generate_biword(data2['BG'][j], n))
    for j in range(train_amount, total_amount):

        testing_data_set.append(generate_biword(data3['CS'][j], n))
    for j in range(train_amount, total_amount):

        testing_data_set.append(generate_biword(data4['DE'][j], n))
    for j in range(train_amount, total_amount):

        testing_data_set.append(generate_biword(data5['DA'][j], n))
    for j in range(train_amount, total_amount):

        testing_data_set.append(generate_biword(data6['EL'][j], n))
    for j in range(train_amount, total_amount):

        testing_data_set.append(generate_biword(data7['ES'][j], n))
    for j in range(train_amount, total_amount):

        testing_data_set.append(generate_biword(data8['ET'][j], n))
    for j in range(train_amount, total_amount):

        testing_data_set.append(generate_biword(data9['FI'][j], n))
    for j in range(train_amount, total_amount):

        testing_data_set.append(generate_biword(data10['FR'][j], n))
    for j in range(train_amount, total_amount):

        testing_data_set.append(generate_biword(data11['GA'][j], n))

    training_data_y = np.append(np.ones((1, train_amount)), (np.zeros((1, train_amount))))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 2))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 3))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 4))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 5))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 6))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 7))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 8))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 9))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 10))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 11))
    testing_data_y = np.append(np.ones((1, len(data1['EN'][train_amount:total_amount]))), np.zeros((1,len(data1['NL'][train_amount:total_amount]))))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data2['BG'][train_amount:total_amount])), 2))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data3['CS'][train_amount:total_amount])), 3))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data4['DE'][train_amount:total_amount])), 4))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data5['DA'][train_amount:total_amount])), 5))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data6['EL'][train_amount:total_amount])), 6))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data7['ES'][train_amount:total_amount])), 7))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data8['ET'][train_amount:total_amount])), 8))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data9['FI'][train_amount:total_amount])), 9))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data10['FR'][train_amount:total_amount])), 10))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data11['GA'][train_amount:total_amount])), 11))
    count_vectorizer = CountVectorizer(lowercase=True, stop_words=None, analyzer=lambda x:x, max_df=0.5, min_df=4, max_features=None, binary=True)
    #count_vectorizer = CountVectorizer(lowercase=True, stop_words=None, max_df=1.0, min_df=1, max_features=None, binary=True)
    transformed_data_x = count_vectorizer.fit_transform(training_data_set + testing_data_set).toarray()

    #print(transformed_data_x[:, 0])
    total_train_amount = train_amount * language_amount
    training_data_x = transformed_data_x[0:total_train_amount, :]
    testing_data_x = transformed_data_x[total_train_amount:, :]


    return training_data_x, testing_data_x, training_data_y, testing_data_y
def set_train_and_test_data_char(data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, n):
    training_data_set = []
    for i in range(0, train_amount):
        training_data_set.append(generate_bigram(data1['EN'][i], n))
    for i in range(0, train_amount):
        training_data_set.append(generate_bigram(data1['NL'][i], n))
    for i in range(0, train_amount):
        training_data_set.append(generate_bigram(data2['BG'][i], n))
    for i in range(0, train_amount):
        training_data_set.append(generate_bigram(data3['CS'][i], n))
    for i in range(0, train_amount):
        training_data_set.append(generate_bigram(data4['DE'][i], n))
    for i in range(0, train_amount):
        training_data_set.append(generate_bigram(data5['DA'][i], n))
    for i in range(0, train_amount):
        training_data_set.append(generate_bigram(data6['EL'][i], n))
    for i in range(0, train_amount):
        training_data_set.append(generate_bigram(data7['ES'][i], n))
    for i in range(0, train_amount):
        training_data_set.append(generate_bigram(data8['ET'][i], n))
    for i in range(0, train_amount):
        training_data_set.append(generate_bigram(data9['FI'][i], n))
    for i in range(0, train_amount):
        training_data_set.append(generate_bigram(data10['FR'][i], n))
    for i in range(0, train_amount):
        training_data_set.append(generate_bigram(data11['GA'][i], n))
    testing_data_set = []
    for j in range(train_amount, total_amount):
        testing_data_set.append(generate_bigram(data1['EN'][j], n))
    for j in range(train_amount, total_amount):
        testing_data_set.append(generate_bigram(data1['NL'][j], n))
    for j in range(train_amount, total_amount):
        testing_data_set.append(generate_bigram(data2['BG'][j], n))
    for j in range(train_amount, total_amount):
        testing_data_set.append(generate_bigram(data3['CS'][j], n))
    for j in range(train_amount, total_amount):
        testing_data_set.append(generate_bigram(data4['DE'][j], n))
    for j in range(train_amount, total_amount):
        testing_data_set.append(generate_bigram(data5['DA'][j], n))
    for j in range(train_amount, total_amount):
        testing_data_set.append(generate_bigram(data6['EL'][j], n))
    for j in range(train_amount, total_amount):
        testing_data_set.append(generate_bigram(data7['ES'][j], n))
    for j in range(train_amount, total_amount):
        testing_data_set.append(generate_bigram(data8['ET'][j], n))
    for j in range(train_amount, total_amount):
        testing_data_set.append(generate_bigram(data9['FI'][j], n))
    for j in range(train_amount, total_amount):
        testing_data_set.append(generate_bigram(data10['FR'][j], n))
    for j in range(train_amount, total_amount):
        testing_data_set.append(generate_bigram(data11['GA'][j], n))

    training_data_y = np.append(np.ones((1, train_amount)), (np.zeros((1, train_amount))))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 2))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 3))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 4))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 5))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 6))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 7))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 8))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 9))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 10))
    training_data_y = np.append(training_data_y, np.full((1, train_amount), 11))
    testing_data_y = np.append(np.ones((1, len(data1['EN'][train_amount:total_amount]))),
                               np.zeros((1, len(data1['NL'][train_amount:total_amount]))))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data2['BG'][train_amount:total_amount])), 2))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data3['CS'][train_amount:total_amount])), 3))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data4['DE'][train_amount:total_amount])), 4))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data5['DA'][train_amount:total_amount])), 5))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data6['EL'][train_amount:total_amount])), 6))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data7['ES'][train_amount:total_amount])), 7))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data8['ET'][train_amount:total_amount])), 8))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data9['FI'][train_amount:total_amount])), 9))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data10['FR'][train_amount:total_amount])), 10))
    testing_data_y = np.append(testing_data_y, np.full((1, len(data11['GA'][train_amount:total_amount])), 11))
    count_vectorizer = CountVectorizer(lowercase=True, stop_words=None, analyzer=lambda x: x, max_df=0.5, min_df=4,
                                       max_features=None, binary=True)
    # count_vectorizer = CountVectorizer(lowercase=True, stop_words=None, max_df=1.0, min_df=1, max_features=None, binary=True)
    transformed_data_x = count_vectorizer.fit_transform(training_data_set + testing_data_set).toarray()

    # print(transformed_data_x[:, 0])
    total_train_amount = train_amount * language_amount
    training_data_x = transformed_data_x[0:total_train_amount, :]
    testing_data_x = transformed_data_x[total_train_amount:, :]
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
    five_list = []
    six_list = []
    seven_list = []
    eight_list = []
    nine_list = []
    ten_list = []
    eleven_list = []
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
        elif y[i] == 5:
            five_list.append(i)
        elif y[i] == 6:
            six_list.append(i)
        elif y[i] == 7:
            seven_list.append(i)
        elif y[i] == 8:
            eight_list.append(i)
        elif y[i] == 9:
            nine_list.append(i)
        elif y[i] == 10:
            ten_list.append(i)
        elif y[i] == 11:
            eleven_list.append(i)

    variable_matrix = np.zeros((language_amount, variable_number))
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
    for i in five_list:
        for x in range(variable_number):
            if X[i, x] == 1:
                variable_matrix[5, x] += 1
    for i in six_list:
        for x in range(variable_number):
            if X[i, x] == 1:
                variable_matrix[6, x] += 1
    for i in seven_list:
        for x in range(variable_number):
            if X[i, x] == 1:
                variable_matrix[7, x] += 1
    for i in eight_list:
        for x in range(variable_number):
            if X[i, x] == 1:
                variable_matrix[8, x] += 1
    for i in nine_list:
        for x in range(variable_number):
            if X[i, x] == 1:
                variable_matrix[9, x] += 1
    for i in ten_list:
        for x in range(variable_number):
            if X[i, x] == 1:
                variable_matrix[10, x] += 1
    for i in eleven_list:
        for x in range(variable_number):
            if X[i, x] == 1:
                variable_matrix[11, x] += 1


    p_x_y_1 = (variable_matrix[0, :] + smooth_value) / (len(one_list) + language_amount * smooth_value)
    p_x_y_0 = (variable_matrix[1, :] + smooth_value) / (len(zero_list) + language_amount * smooth_value)
    p_x_y_2 = (variable_matrix[2, :] + smooth_value) / (len(two_list) + language_amount * smooth_value)
    p_x_y_3 = (variable_matrix[3, :] + smooth_value) / (len(three_list) + language_amount * smooth_value)
    p_x_y_4 = (variable_matrix[4, :] + smooth_value) / (len(four_list) + language_amount * smooth_value)
    p_x_y_5 = (variable_matrix[5, :] + smooth_value) / (len(five_list) + language_amount * smooth_value)
    p_x_y_6 = (variable_matrix[6, :] + smooth_value) / (len(six_list) + language_amount * smooth_value)
    p_x_y_7 = (variable_matrix[7, :] + smooth_value) / (len(seven_list) + language_amount * smooth_value)
    p_x_y_8 = (variable_matrix[8, :] + smooth_value) / (len(eight_list) + language_amount * smooth_value)
    p_x_y_9 = (variable_matrix[9, :] + smooth_value) / (len(nine_list) + language_amount * smooth_value)
    p_x_y_10 = (variable_matrix[10, :] + smooth_value) / (len(ten_list) + language_amount * smooth_value)
    p_x_y_11 = (variable_matrix[11, :] + smooth_value) / (len(eleven_list) + language_amount * smooth_value)
    p_y_1 = len(one_list) / len(y)
    p_y_0 = len(zero_list) / len(y)
    p_y_2 = len(two_list) / len(y)
    p_y_3 = len(three_list) / len(y)
    p_y_4 = len(four_list) / len(y)
    p_y_5 = len(five_list) / len(y)
    p_y_6 = len(six_list) / len(y)
    p_y_7 = len(seven_list) / len(y)
    p_y_8 = len(eight_list) / len(y)
    p_y_9= len(nine_list) / len(y)
    p_y_10 = len(ten_list) / len(y)
    p_y_11 = len(eleven_list) / len(y)

    return p_x_y_11,p_x_y_10,p_x_y_9,p_x_y_8,p_x_y_7,p_x_y_6,p_x_y_5, p_x_y_4, p_x_y_3, p_x_y_2, p_x_y_1, p_x_y_0,p_y_11,p_y_10,p_y_9,p_y_8,p_y_7,p_y_6, p_y_5, p_y_4, p_y_3, p_y_2, p_y_1, p_y_0, smooth_value
def test_model(p_x_y_11,p_x_y_10,p_x_y_9,p_x_y_8,p_x_y_7,p_x_y_6,p_x_y_5, p_x_y_4, p_x_y_3, p_x_y_2, p_x_y_1, p_x_y_0,p_y_11,p_y_10,p_y_9,p_y_8,p_y_7,p_y_6, p_y_5, p_y_4, p_y_3, p_y_2, p_y_1, p_y_0, X, smooth):
    X_1 = np.zeros((X.shape[0], X.shape[1]))
    X_2 = np.zeros((X.shape[0], X.shape[1]))
    X_3 = np.zeros((X.shape[0], X.shape[1]))
    X_4 = np.zeros((X.shape[0], X.shape[1]))
    X_5 = np.zeros((X.shape[0], X.shape[1]))
    X_6 = np.zeros((X.shape[0], X.shape[1]))
    X_7 = np.zeros((X.shape[0], X.shape[1]))
    X_8 = np.zeros((X.shape[0], X.shape[1]))
    X_9 = np.zeros((X.shape[0], X.shape[1]))
    X_10 = np.zeros((X.shape[0], X.shape[1]))
    X_11 = np.zeros((X.shape[0], X.shape[1]))
    X_12 = np.zeros((X.shape[0], X.shape[1]))
    np.log(p_x_y_1)
    np.log(p_x_y_0)
    np.log(p_x_y_2)
    np.log(p_x_y_3)
    np.log(p_x_y_4)
    np.log(p_x_y_5)
    np.log(p_x_y_6)
    np.log(p_x_y_7)
    np.log(p_x_y_8)
    np.log(p_x_y_9)
    np.log(p_x_y_10)
    np.log(p_x_y_11)
    for x in range(X.shape[0]):
        X_1[x, :] = np.multiply(p_x_y_1, X[x, :])
        X_2[x, :] = np.multiply(p_x_y_0, X[x, :])
        X_3[x, :] = np.multiply(p_x_y_2, X[x, :])
        X_4[x, :] = np.multiply(p_x_y_3, X[x, :])
        X_5[x, :] = np.multiply(p_x_y_4, X[x, :])
        X_6[x, :] = np.multiply(p_x_y_5, X[x, :])
        X_7[x, :] = np.multiply(p_x_y_6, X[x, :])
        X_8[x, :] = np.multiply(p_x_y_7, X[x, :])
        X_9[x, :] = np.multiply(p_x_y_8, X[x, :])
        X_10[x, :] = np.multiply(p_x_y_9, X[x, :])
        X_11[x, :] = np.multiply(p_x_y_10, X[x, :])
        X_12[x, :] = np.multiply(p_x_y_11, X[x, :])
    one_result = []
    zero_result = []
    two_result = []
    three_result = []
    four_result = []
    five_result = []
    six_result = []
    seven_result = []
    eight_result = []
    nine_result = []
    ten_result = []
    eleven_result = []
    one_log = []
    zero_log = []
    two_log = []
    three_log = []
    four_log = []
    five_log = []
    six_log = []
    seven_log = []
    eight_log = []
    nine_log = []
    ten_log = []
    eleven_log = []
    for i in range(X_1.shape[0]):
        log_list_1 = []
        log_list_0 = []
        log_list_2 = []
        log_list_3 = []
        log_list_4 = []
        log_list_5 = []
        log_list_6 = []
        log_list_7 = []
        log_list_8 = []
        log_list_9 = []
        log_list_10 = []
        log_list_11 = []
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
            if X_6[i, x] != 0:
                log_list_5.append(X_6[i, x])
            if X_7[i, x] != 0:
                log_list_6.append(X_7[i, x])
            if X_8[i, x] != 0:
                log_list_7.append(X_8[i, x])
            if X_9[i, x] != 0:
                log_list_8.append(X_9[i, x])
            if X_10[i, x] != 0:
                log_list_9.append(X_10[i, x])
            if X_11[i, x] != 0:
                log_list_10.append(X_11[i, x])
            if X_12[i, x] != 0:
                log_list_11.append(X_12[i, x])

        one_log.append(log_list_1)
        zero_log.append(log_list_0)
        two_log.append(log_list_2)
        three_log.append(log_list_3)
        four_log.append(log_list_4)
        five_log.append(log_list_5)
        six_log.append(log_list_6)
        seven_log.append(log_list_7)
        eight_log.append(log_list_8)
        nine_log.append(log_list_9)
        ten_log.append(log_list_10)
        eleven_log.append(log_list_11)
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
    for b in five_log:
        five_result.append(np.log(smooth) + np.log(p_y_5) + sum(b))
    for c in six_log:
        six_result.append(np.log(smooth) + np.log(p_y_6) + sum(c))
    for d in seven_log:
        seven_result.append(np.log(smooth) + np.log(p_y_7) + sum(d))
    for e in eight_log:
        eight_result.append(np.log(smooth) + np.log(p_y_8) + sum(e))
    for f in nine_log:
        nine_result.append(np.log(smooth) + np.log(p_y_9) + sum(f))
    for g in ten_log:
        ten_result.append(np.log(smooth) + np.log(p_y_10) + sum(g))
    for h in eleven_log:
        eleven_result.append(np.log(smooth) + np.log(p_y_11) + sum(h))

    pred = []

    for i in range(len(one_result)):

        answer = max([zero_result[i], one_result[i], two_result[i], three_result[i], four_result[i], five_result[i], six_result[i], seven_result[i], eight_result[i], nine_result[i], ten_result[i], eleven_result[i]])
        if answer == zero_result[i]:
            pred.append(0)
            F1_dictionary_word[language_list[0]][i]['predicted'] = 1
            F1_dictionary_char[language_list[0]][i]['predicted'] = 1
        elif answer == one_result[i]:
            pred.append(1)
            F1_dictionary_word[language_list[1]][i]['predicted'] = 1
            F1_dictionary_char[language_list[1]][i]['predicted'] = 1
        elif answer == two_result[i]:
            pred.append(2)
            F1_dictionary_word[language_list[2]][i]['predicted'] = 1
            F1_dictionary_char[language_list[2]][i]['predicted'] = 1
        elif answer == three_result[i]:
            pred.append(3)
            F1_dictionary_word[language_list[3]][i]['predicted'] = 1
            F1_dictionary_char[language_list[3]][i]['predicted'] = 1
        elif answer == four_result[i]:
            pred.append(4)
            F1_dictionary_word[language_list[4]][i]['predicted'] = 1
            F1_dictionary_char[language_list[4]][i]['predicted'] = 1
        elif answer == five_result[i]:
            pred.append(5)
            F1_dictionary_word[language_list[5]][i]['predicted'] = 1
            F1_dictionary_char[language_list[5]][i]['predicted'] = 1
        elif answer == six_result[i]:
            pred.append(6)
            F1_dictionary_word[language_list[6]][i]['predicted'] = 1
            F1_dictionary_char[language_list[6]][i]['predicted'] = 1
        elif answer == seven_result[i]:
            pred.append(7)
            F1_dictionary_word[language_list[7]][i]['predicted'] = 1
            F1_dictionary_char[language_list[7]][i]['predicted'] = 1
        elif answer == eight_result[i]:
            pred.append(8)
            F1_dictionary_word[language_list[8]][i]['predicted'] = 1
            F1_dictionary_char[language_list[8]][i]['predicted'] = 1
        elif answer == nine_result[i]:
            pred.append(9)
            F1_dictionary_word[language_list[9]][i]['predicted'] = 1
            F1_dictionary_char[language_list[9]][i]['predicted'] = 1
        elif answer == ten_result[i]:
            pred.append(10)
            F1_dictionary_word[language_list[10]][i]['predicted'] = 1
            F1_dictionary_char[language_list[10]][i]['predicted'] = 1
        elif answer == eleven_result[i]:
            pred.append(11)
            F1_dictionary_word[language_list[11]][i]['predicted'] = 1
            F1_dictionary_char[language_list[11]][i]['predicted'] = 1

    return pred

#if __name__ == '__main__':
    #training_data_set, testing_data_set, y_train, y_test = set_train_and_test_data(data1, data2)
    #p_x_y_2, p_x_y_1, p_x_y_0, p_y_2, p_y_1, p_y_0, smooth_value = training(0.1, training_data_set, y_train)
    #prediction = test_model(p_x_y_2, p_x_y_1, p_x_y_0, p_y_2, p_y_1, p_y_0, testing_data_set, smooth_value)
    #print("accuracy = {}".format(np.mean((y_test - prediction) == 0)))

def processing_word(n):
    training_data_set, testing_data_set, y_train, y_test = set_train_and_test_data_word(data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, n)
    copy_test_data_y = []
    copy_test_data_y.extend(y_test.tolist())
    init_dictionary(copy_test_data_y)
    p_x_y_11,p_x_y_10,p_x_y_9,p_x_y_8,p_x_y_7,p_x_y_6,p_x_y_5, p_x_y_4, p_x_y_3, p_x_y_2, p_x_y_1, p_x_y_0,p_y_11,p_y_10,p_y_9,p_y_8,p_y_7,p_y_6, p_y_5, p_y_4, p_y_3, p_y_2, p_y_1, p_y_0, smooth_value = training(0.1, training_data_set, y_train)
    prediction = test_model(p_x_y_11,p_x_y_10,p_x_y_9,p_x_y_8,p_x_y_7,p_x_y_6,p_x_y_5, p_x_y_4, p_x_y_3, p_x_y_2, p_x_y_1, p_x_y_0,p_y_11,p_y_10,p_y_9,p_y_8,p_y_7,p_y_6, p_y_5, p_y_4, p_y_3, p_y_2, p_y_1, p_y_0, testing_data_set, smooth_value)
    print('word accuracy = {}'.format(np.mean((y_test - prediction) == 0)))
    #print(calcualte_F1_word())
    calculate_average_score_word()
    return np.mean((y_test - prediction) == 0)
def processing_char(n):
    training_data_set, testing_data_set, y_train, y_test = set_train_and_test_data_char(data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, n)
    copy_test_data_y = []
    copy_test_data_y.extend(y_test.tolist())
    init_dictionary(copy_test_data_y)
    p_x_y_11,p_x_y_10,p_x_y_9,p_x_y_8,p_x_y_7,p_x_y_6,p_x_y_5, p_x_y_4, p_x_y_3, p_x_y_2, p_x_y_1, p_x_y_0,p_y_11,p_y_10,p_y_9,p_y_8,p_y_7,p_y_6, p_y_5, p_y_4, p_y_3, p_y_2, p_y_1, p_y_0, smooth_value = training(0.1, training_data_set, y_train)
    prediction = test_model(p_x_y_11,p_x_y_10,p_x_y_9,p_x_y_8,p_x_y_7,p_x_y_6,p_x_y_5, p_x_y_4, p_x_y_3, p_x_y_2, p_x_y_1, p_x_y_0,p_y_11,p_y_10,p_y_9,p_y_8,p_y_7,p_y_6, p_y_5, p_y_4, p_y_3, p_y_2, p_y_1, p_y_0, testing_data_set, smooth_value)
    print('char accuracy = {}'.format(np.mean((y_test - prediction) == 0)))
    #print(calcualte_F1_char())
    calculate_average_score_char()
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
'''''''''
def init_dictionary(copy_test_data_y):

    for i in range(language_amount):
        F1_dictionary_word[language_list[i]] = {}
        F1_dictionary_char[language_list[i]] = {}
        for n in range(test_amount):
            F1_dictionary_word[language_list[i]][n] = {}
            F1_dictionary_word[language_list[i]][n]['true'] = 0
            F1_dictionary_word[language_list[i]][n]['predicted'] = 0
            F1_dictionary_char[language_list[i]][n] = {}
            F1_dictionary_char[language_list[i]][n]['true'] = 0
            F1_dictionary_char[language_list[i]][n]['predicted'] = 0
    for i in range(language_amount):
        for n in range(test_amount):
            #print(int(copy_test_data_y[i * 100 + n]))
            #print(language_list[int(copy_test_data_y[i * 100 + n])])
            if language_list[int(copy_test_data_y[i * 100 + n])] == language_list[i]:
                F1_dictionary_word[language_list[int(copy_test_data_y[i * 100 + n])]][n]['true'] = 1
            if language_list[int(copy_test_data_y[i * 100 + n])] == language_list[i]:
                F1_dictionary_char[language_list[int(copy_test_data_y[i * 100 + n])]][n]['true'] = 1
'''''''''
def init_dictionary(copy_test_data_y):

    for i in range(language_amount):
        F1_dictionary_word[language_list[i]] = {}
        F1_dictionary_char[language_list[i]] = {}
        for n in range(language_amount * test_amount):
            F1_dictionary_word[language_list[i]][n] = {}
            F1_dictionary_word[language_list[i]][n]['true'] = 0
            F1_dictionary_word[language_list[i]][n]['predicted'] = 0
            F1_dictionary_char[language_list[i]][n] = {}
            F1_dictionary_char[language_list[i]][n]['true'] = 0
            F1_dictionary_char[language_list[i]][n]['predicted'] = 0
    for i in range(language_amount):
        for n in range(test_amount):


            F1_dictionary_word[language_list[int(copy_test_data_y[i * 100 + n])]][i * 100 + n]['true'] = 1

            F1_dictionary_char[language_list[int(copy_test_data_y[i * 100 + n])]][i * 100 + n]['true'] = 1

def calcualte_F1_word():
    global total_correct_assignment_word
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for language in language_list:
        for value in F1_dictionary_word[language].values():

            if value['true'] == 0 and value['predicted'] == 0:
                TN += 1
            elif value['true'] == 1 and value['predicted'] == 1:
                TP += 1
                total_correct_assignment_word += 1
            elif value['true'] == 1 and value['predicted'] == 0:
                FN += 1
            elif value['true'] == 0 and value['predicted'] == 1:
                FP += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * ((precision * recall) / (precision + recall))

    return TP, FP, FN, TN, precision, recall, F1
def calcualte_F1_char():
    global total_correct_assignment_char
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for language in language_list:
        for value in F1_dictionary_char[language].values():

            if value['true'] == 0 and value['predicted'] == 0:
                TN += 1
            elif value['true'] == 1 and value['predicted'] == 1:
                TP += 1
                total_correct_assignment_char += 1
            elif value['true'] == 1 and value['predicted'] == 0:
                FN += 1
            elif value['true'] == 0 and value['predicted'] == 1:
                FP += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * ((precision * recall) / (precision + recall))

    return TP, FP, FN, TN, precision, recall, F1
def calculate_F1_language_char(language):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for value in F1_dictionary_char[language].values():

        if value['true'] == 0 and value['predicted'] == 0:
            TN += 1
        elif value['true'] == 1 and value['predicted'] == 1:
            TP += 1

        elif value['true'] == 1 and value['predicted'] == 0:
            FN += 1
        elif value['true'] == 0 and value['predicted'] == 1:
            FP += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    F1 = 2 * ((precision * recall) / (precision + recall))

    return TP, FP, FN, TN, precision, recall, F1


def calculate_F1_language_word(language):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for value in F1_dictionary_word[language].values():

        if value['true'] == 0 and value['predicted'] == 0:
            TN += 1
        elif value['true'] == 1 and value['predicted'] == 1:
            TP += 1

        elif value['true'] == 1 and value['predicted'] == 0:
            FN += 1
        elif value['true'] == 0 and value['predicted'] == 1:
            FP += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    F1 = 2 * ((precision * recall) / (precision + recall))

    return TP, FP, FN, TN, precision, recall, F1
def calculate_average_score_word():
    recall_list_word = 0
    precision_list_word = 0
    F_score_list_word = 0

    for i, language in enumerate(language_list):

        TP1, FP1, FN1, TN1, precision1, recall1, F1_1 = calculate_F1_language_word(language)
        recall_list_word += precision1
        precision_list_word += recall1
        F_score_list_word += F1_1

    average_recall_1 = recall_list_word / language_amount
    average_precision_1 = precision_list_word / language_amount
    average_F_score_1 = F_score_list_word / language_amount

    print(average_recall_1, average_precision_1, average_F_score_1)

def calculate_average_score_char():

    recall_list_char = 0
    precision_list_char = 0
    F_score_list_char = 0
    for i, language in enumerate(language_list):
        TP2, FP2, FN2, TN2, precision2, recall2, F1_2 = calculate_F1_language_char(language)


        recall_list_char += precision2
        precision_list_char += recall2
        F_score_list_char += F1_2

    average_recall_2 = recall_list_char / language_amount
    average_precision_2 = precision_list_char / language_amount
    average_F_score_2 = F_score_list_char / language_amount

    print(average_recall_2, average_precision_2, average_F_score_2)
if __name__ == '__main__':
    draw_diagram()




