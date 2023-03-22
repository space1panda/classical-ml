"""
Naive Bayes implementation for solving SMS spam filter task. Based on the KDNuggets article:
https://www.kdnuggets.com/2020/06/naive-bayes-algorithm-everything.html 
"""

import re
from collections import defaultdict
import numpy as np


class NaiveBayesClassifier:
    """
    The model fits categorical data distribution by optimizing the likelihood of the features using Bayesian inference. Features in each example are assumed to be iid.
    The object updates relative frequencies learnt from data on the call. These
    are parameters of the model, thus obtaining them can be defined as a
    training process.
    """

    def __init__(self, datasource, num_classes=2, laplacian_smoothing=1):
        self._smoothing = laplacian_smoothing
        self._num_classes = num_classes
        self._datasource = datasource
        self.init_params()
    
    @property
    def parameters(self):
        return self._parameters
    
    def __call__(self, input):
        # updating likelihoods of the given word to belong to each class
        data, word = input
        for idx, (c, data) in enumerate(data):
            n_word_given_class = data[word].sum()
            p_word_given_class = (
                n_word_given_class + self._smoothing) / (
                n_spam + self._smoothing * self._n_vocabulary)
            self._parameters[str(idx)]['likelihood'][word] = p_word_given_class

    def init_params(self):
        self._parameters = defaultdict(dict)
        for i in range(self._num_classes):
            self._parameters[str(i)]['likelihood'] = {entity: 0 for entity in self._vocabulary}
            self._parameters[str(i)]['prior'] = 0
        self._n_vocabulary = len(self._datasource)

        
        self._parameters['0']['prior'] = len(spam_messages) / len(data)
        self._parameters['1']['prior'] = len(ham_messages) / len(data)
        n_words_per_spam_message = spam_messages['SMS'].apply(len)
        n_spam = n_words_per_spam_message.sum()
    
    def evaluate(self, data):
        preds = list(data['SMS'].apply(self.inference))
        gt = list(data['Label'].apply(lambda x: {'spam': 0, 'ham': 1}[x]))
        accuracy = self.get_classifier_accuracy(gt, preds)
        print(f"Total accuracy: {accuracy}")
    
    def get_classifier_accuracy(self, gt, preds):
        gt = np.array(gt)
        preds = np.array(preds)
        total = len(gt)
        tptn = len(gt[gt == preds])
        return round(tptn / total, 4)

    def inference(self, input):
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
