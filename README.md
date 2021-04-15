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

** Running Naive Bayes Classifier **

Please use 'Naive Bayes Classifier 23 languages.py', do not use the draft version. When the user run 'Naive Bayes Classifier 23 languages.py', the user need to put the 23 langauges files and the 'Naive Bayes Classifier 23 languages.py' into the same directory. These 23 languages file are 'EN-NL','EN-BG','EN-CS','EN-DE','EN-DA','EN-EL','EN-ES','EN-ET','EN-FI','EN-FR', 'EN-SV', 'EN-HR', 'EN-HU', 'EN-IT', 'EN-LT', 'EN-LV', 'EN-MT', 'EN-PL', 'EN-PT', 'EN-RO', 'EN-SK', 'EN-SL'. Here is the link to download them https://www.kaggle.com/hgultekin/paralel-translation-corpus-in-22-languages?select=EN-HU
