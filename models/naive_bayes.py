"""
Naive Bayes implementation for solving SMS spam filter task. Based on the KDNuggets article:
https://www.kdnuggets.com/2020/06/naive-bayes-algorithm-everything.html 
"""

import re
from collections import defaultdict
import numpy as np


class NaiveBayesClassifier:
    """
    The model fits categorical data by optimizing the likelihood of the features using Bayesian inference assuming the features
    in each example set are independent and equivariant. The features likelihood defines the set of parameters of categorcial distribution
    used for predictions during inference.
    """
    def __init__(self, vocabulary, num_classes=2, smoothing=1):
        self._smoothing = smoothing
        self._vocabular = vocabulary
        self._num_classes = num_classes
        self.init_params()
    
    @property
    def parameters(self):
        return self._parameters

    def init_params(self):
        self._parameters = defaultdict(dict)
        for i in range(self._num_classes):
            self._parameters[str(i)]['likelihood'] = {entity: 0 for entity in self._vocabulary}
            self._parameters[str(i)]['prior'] = 0

    def fit(self, data):
        # # Isolating spam and ham messages first
        spam_messages = data[data['Label'] == 'spam']
        ham_messages = data[data['Label'] == 'ham']

        # priorss
        self._parameters['0']['prior'] = len(spam_messages) / len(data)
        self._parameters['1']['prior'] = len(ham_messages) / len(data)

        # # N_Spam - total number of entities in spam messages
        n_words_per_spam_message = spam_messages['SMS'].apply(len)
        n_spam = n_words_per_spam_message.sum()

        # # N_Ham - total number of entities in spam messages
        n_words_per_ham_message = ham_messages['SMS'].apply(len)
        n_ham = n_words_per_ham_message.sum()

        # # N_Vocabulary
        n_vocabulary = len(self._vocabulary)

        # Initiate parameters

        # Calculate parameters
        for word in self._vocabulary:
            n_word_given_spam = spam_messages[word].sum() # spam_messages already defined
            p_word_given_spam = (n_word_given_spam + self._smoothing) / (n_spam + self._smoothing * n_vocabulary)
            self._parameters['0']['likelihood'][word] = p_word_given_spam

            n_word_given_ham = ham_messages[word].sum() # ham_messages already defined
            p_word_given_ham = (n_word_given_ham + self._smoothing) / (n_ham + self._smoothing * n_vocabulary)
            self._parameters['1']['likelihood'][word] = p_word_given_ham
    
    def evaluate(self, data):
        preds = list(data['SMS'].apply(self.predict))
        gt = list(data['Label'].apply(lambda x: {'spam': 0, 'ham': 1}[x]))
        accuracy = self.get_classifier_accuracy(gt, preds)
        print(f"Total accuracy: {accuracy}")
    
    def get_classifier_accuracy(self, gt, preds):
        gt = np.array(gt)
        preds = np.array(preds)
        total = len(gt)
        tptn = len(gt[gt == preds])
        return round(tptn / total, 4)

    def predict(self, input):
        message = re.sub('\W', ' ', input)
        message = message.lower().split()
        probs = [p['prior'] for p in self._parameters.values()]

        for word in message:
            for idx, params in self._parameters.items():
                likelihood = params['likelihood']
                word_likelihood = likelihood.get(word)
                if word_likelihood is not None:
                    probs[int(idx)] *= word_likelihood
        return np.argmax(probs)


if __name__ == '__main__':
    from data_processing.spam_datasource import SpamDatasource
    datasource = SpamDatasource(data_path='/home/spacepanda/workspace/projects/classical-ml/dataset/spam.csv')
    trainset, testset = datasource.get_dataframes()
    voc = datasource.vocabulary
    model = NaiveBayesClassifier(vocabulary=voc)
    model.fit(trainset)
    model.evaluate(testset)
    # message = 'Meet you downstairs'
    # label = model.predict(message)
    # print(message)
    # print(datasource.classes[label])
