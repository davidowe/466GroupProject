# 466 Group
- Jionghao Chen
- Zi Xuan Zhang
- Brian Qi
- Andrew Choi
- Owen Randall

The text files (such as "EN-NL.txt") are not imported yet into GitHub due to file size. Download them at https://www.kaggle.com/hgultekin/paralel-translation-corpus-in-22-languages

Preprocessed files at https://drive.google.com/drive/u/0/folders/1DWGmF902NCMSBjIRurih-trcUXZm8HqQ


** Running SVM **

When you run scrambledata.py, you create a file with 1000 sentences from each of the 23 languages, assuming the .tsv language files are in a "data" folder.
The Python file then writes them into a document, 'testing_VERY_FEW.tsv', which is read by svm.py
svm.py can then be called, assuming that 'testing_VERY_FEW.tsv' is in the same directory, and the .tsv language files are in a "data" folder.