import numpy as np

class confusionMatrix:
    def __init__(self, languages):
        # self.languages - an array or languages
        # self.n_lang - number of languages
        # self.confusion - confusion matrix
        self.languages = languages
        self.n_lang = len(self.languages)
        self.confusion = np.zeros((self.n_lang, self.n_lang))

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
        print('\t', end='')
        for lang in self.languages:
            print(lang, end=' ')
        print('\n')

        for i, lang in enumerate(self.languages):
            print(lang, end=' ')
            for count in self.confusion[i]:
                print(count, end=' ')
            print('\n')   