import pandas as pd
import numpy as np

class confusionMatrix:
    def __init__(self, languages):
        # self.languages - an array or languages
        # self.n_lang - number of languages
        # self.confusion - confusion matrix
        self.languages = languages
        self.n_lang = len(self.languages)
        self.confusion = np.zeros((self.n_lang, self.n_lang), dtype=np.int32)

    def add_confusion(self, pred, gt):
        # pred - prediction array
        # gt - ground truth one-hot label
        # returns true if prediction is correct
        p_index = np.argmax(pred)
        l_index = np.argmax(gt)
        self.confusion[l_index][p_index] += 1
        return (p_index == l_index)

    def print_confusion(self):
        # needs better formatting
        table = pd.DataFrame(self.confusion, self.languages, self.languages)
        pd.set_option('max_columns', None)
        print(table)