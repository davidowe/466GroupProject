import csv
import json
from io import open
import os

### Quick User Guide ###
'''Example
#data link: https://drive.google.com/drive/u/0/folders/1DWGmF902NCMSBjIRurih-trcUXZm8HqQ

from preReader import preReader

pr = preReader('EN','NL')

training_EN, training_NL = pr.trainingSet()

testing_EN, testing_NL = pr.trainingSet()

'''




class preReader:

    def __init__(self,src,trg):
        self.src = src
        self.trg = trg
        
    '''reads out the dataset in the list of
    (sentence, label)
                src                 trg
    return: [(sentence,label)] [(sentence,label)]
    '''
    def wholeDataTest(self):
        
        fsrc = open('data/' + self.src + '.tsv',encoding = 'utf-8')
        ftrg = open('data/' + self.trg + '.tsv',encoding = 'utf-8')
       

        read_src =  csv.reader(fsrc,delimiter = '\t')
        next(read_src)

        read_trg = csv.reader(ftrg,delimiter = '\t')
        next(read_trg)

        out_src = []
        for row in read_src:
            if len(row)>0:
                sentence, label = row   
                out_src.append((sentence,label))
        
        out_trg = []
        for row in read_trg:
            if len(row)>0:
                sentence, label = row   
                out_trg.append((sentence,label))
    

        return out_src, out_trg
    
    '''80 20 split
    '''
    def trainingSet(self,split = 0.8):
        out_src, out_trg = self.wholeDataTest()
        
        return out_src[:int(len(out_src)*split)], out_trg[:int(len(out_trg)*split)]
    
    def testSet(self,split = 0.2):
        out_src, out_trg = self.wholeDataTest()

        return out_src[-1*int(len(out_src)*split):], out_trg[-1*int(len(out_trg)*split):]

    def validationSet(self,tSplit = 0.6,vSplit = 0.2):
        out_src, out_trg = self.wholeDataTest()

        return out_src[int(len(out_src)*tSplit):int(len(out_src)*(tSplit+vSplit))], out_trg[int(len(out_src)*tSplit):int(len(out_src)*(tSplit+vSplit))] 
    
        
