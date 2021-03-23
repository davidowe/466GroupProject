import csv
import json
from io import open
import os

### Quick User Guide ###
'''Example
#data link: https://drive.google.com/drive/u/0/folders/1DWGmF902NCMSBjIRurih-trcUXZm8HqQ

from preReader import preReader

pr = preReader('EN','NL')
pr.initialize()

training_EN, training_NL = pr.trainingSet()

testing_EN, testing_NL = pr.trainingSet()

ngramUnique = pr.ngramUnique(6,training_EN)

'''

'''TODO:

A list of every unique ngram and their counts would be useful to me
Wouldn't need to go past about 5 grams 6 grams at most


'''


class preReader:

    def __init__(self,src,trg):
        self.src = src
        self.trg = trg
        self.dataset = (self.wholeDataTest())

    '''reads out the dataset in the list of
    (sentence, label)
                src                 trg
    return: [(sentence,label)] [(sentence,label)]
    '''
    def wholeDataTest(self):
        
        with open('data/' + self.src + '.tsv',encoding = 'utf-8') as fsrc :
            read_src =  csv.reader(fsrc,delimiter = '\t')
            next(read_src)
            out_src = []
            for row in read_src:
                if len(row)>0:
                    sentence, label = row   
                    out_src.append((sentence,label))

        with open('data/' + self.trg + '.tsv',encoding = 'utf-8') as ftrg :
            read_trg = csv.reader(ftrg,delimiter = '\t')
            next(read_trg)

            out_trg = []
            for row in read_trg:
                if len(row)>0:
                    sentence, label = row   
                    out_trg.append((sentence,label))
        

        return out_src, out_trg

    # def initialize(self):
    #     self.dataset = (self.wholeDataTest())
    '''80 20 split
    '''
    def trainingSet(self,split = 0.8):
        out_src, out_trg = self.dataset
        
        return out_src[:int(len(out_src)*split)], out_trg[:int(len(out_trg)*split)]
    
    def testSet(self,split = 0.2):
        out_src, out_trg = self.dataset

        return out_src[-1*int(len(out_src)*split):], out_trg[-1*int(len(out_trg)*split):]

    def validationSet(self,tSplit = 0.6,vSplit = 0.2):
        out_src, out_trg = self.dataset

        return out_src[int(len(out_src)*tSplit):int(len(out_src)*(tSplit+vSplit))], out_trg[int(len(out_src)*tSplit):int(len(out_src)*(tSplit+vSplit))] 


    '''
    A list of every unique ngram and their counts would be useful to me
    Wouldn't need to go past about 5 grams 6 grams at most
    '''
    def ngramUnique(self, n, process):
    
        unique = {}
        for doc in process:
            sentence = doc[0]
            for i in range(len(sentence)):
                ngramText = sentence[i:i+n].lower()
                if ngramText not in unique:
                    unique[ngramText] = 1.0
                   
                elif ngramText in unique:
                    unique[ngramText] += 1.0

        return unique

'''
pr = preReader('EN','NL')

training_EN, training_NL = pr.trainingSet()

testing_EN, testing_NL = pr.testSet()

ngramUnique = pr.ngramUnique(6,testing_NL)
'''