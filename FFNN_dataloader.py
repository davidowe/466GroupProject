import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class dataloader:
    def __init__(self, train, test):
        self.languages = ['BG.tsv',
                          'CS.tsv',
                          'DA.tsv',
                          'DE.tsv',
                          'EL.tsv',
                          'EN.tsv',
                          'ES.tsv',
                          'ET.tsv',
                          'FI.tsv',
                          'FR.tsv',
                          'HR.tsv',
                          'HU.tsv',
                          'IT.tsv',
                          'LT.tsv',
                          'LV.tsv',
                          'MT.tsv',
                          'NL.tsv',
                          'PL.tsv',
                          'PT.tsv',
                          'RO.tsv',
                          'SK.tsv',
                          'SL.tsv',
                          'SV.tsv']
        self.n_languages = len(self.languages)
        self.n_train=train
        self.n_test=test
        self.n = self.n_train + self.n_test
        self.data = np.array([])
        self.labels = np.zeros((self.n*self.n_languages, self.n_languages))
        self.count_vectorizer = None
        self.vec_data = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

    def load_data(self):
        # loads all data into a single array
        # n - number of train and test data to load per language
        n = self.n_train + self.n_test
        for i, language in enumerate(self.languages):
            loaded_lang = pd.read_csv(language, sep='\t', header=None, nrows=(self.n+1))
            self.data = np.append(self.data, loaded_lang[0][1:n+1])
            self.labels[i * self.n : i * self.n + self.n, i] = 1

    def vectorize_data(self):
        # vectorizes entire dataset
        # should be run after load_data(self) has been run
        self.count_vectorizer = CountVectorizer(lowercase=True, stop_words=None, max_df=1.0, min_df=2, max_features=None)
        self.vec_data = self.count_vectorizer.fit_transform(self.data).toarray()
        del self.data

    def split_data(self):
        # splits data into training set and test set
        # also creates the corresponding labels where each language is a
        # one hot vector label
        # vectorize_data(self) should have been run prior to split_data(self)
        n_train_rows = self.n_train*self.n_languages
        n_test_rows = self.n_test*self.n_languages
        n_columns = self.vec_data.shape[1]
        self.train_x = np.zeros((n_train_rows, n_columns))
        self.train_y = np.zeros((n_train_rows, self.n_languages))
        self.test_x = np.zeros((n_test_rows, n_columns))
        self.test_y = np.zeros((n_test_rows, self.n_languages))
        for i in range(self.n_languages):
            #indices for entire dataset
            tr_s = i*self.n
            tr_e = tr_s + self.n_train
            tst_s = tr_e
            tst_e = tst_s + self.n_test

            #indices for split dataset
            train_s = i*self.n_train
            test_s = i*self.n_test
            train_e = train_s + self.n_train
            test_e = test_s + self.n_test

            self.train_x[train_s:train_e] = self.vec_data[tr_s:tr_e]
            self.train_y[train_s:train_e] = self.labels[tr_s:tr_e]
            self.test_x[test_s:test_e] = self.vec_data[tst_s:tst_e]
            self.test_y[test_s:test_e] = self.labels[tst_s:tst_e]

    def get_train(self):
        # returns training data and training data labels
        return self.train_x, self.train_y 

    def get_test(self):
        # returns test data and test data labels
        return self.test_x, self.test_y

    def get_n_languages(self):
        # returns the number of languages used
        return self.n_languages

    def prepare_data(self):
        # combines actions of importing raw data, vectorizing data, 
        # and splitting data
        print('preparing data...')
        self.load_data()
        print('vectorizing')
        self.vectorize_data()
        print('splitting')
        self.split_data()
        print('data prepared')

    def get_languages(self):
        # returns a list of languages
        languages = []
        for l in self.languages:
            language = l[:2]
            languages.append(language)
        return languages
