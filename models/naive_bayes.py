"""
Naive Bayes implementation for solving SMS spam filter task. Based on the KDNuggets article:
https://www.kdnuggets.com/2020/06/naive-bayes-algorithm-everything.html 
"""


class TabularNaiveBayesClassifier:
    def __init__(self, vocabulary, num_classes=2, smoothing=1):
        self._smoothing = smoothing
        self._parameters = None
        self._vocabulary = vocabulary
        self._num_classes = num_classes
        
        self.init_params()
    
    @property
    def parameters(self):
        return self._parameters

    def init_params(self):
        self._parameters = {}
        for i in range(self._num_classes):
            self._parameters[str(i)] = {entity: 0 for entity in self._vocabulary}


    def fit(self, data):
        # # Isolating spam and ham messages first
        spam_messages = data[data['Label'] == 'spam']
        ham_messages = data[data['Label'] == 'ham']

        # prior

        # p_spam = len(spam_messages) / len(data)
        # p_ham = len(ham_messages) / len(data)

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
            self._parameters['0'][word] = p_word_given_spam

            n_word_given_ham = ham_messages[word].sum() # ham_messages already defined
            p_word_given_ham = (n_word_given_ham + self._smoothing) / (n_ham + self._smoothing * n_vocabulary)
            self._parameters['1'] = p_word_given_ham


    def predict(self, data):
        pred_cls = None
        return pred_cls


if __name__ == '__main__':
    from data_processing.spam_datasource import SpamDatasource
    datasource = SpamDatasource(data_path='/home/spacepanda/workspace/projects/classical-ml/dataset/spam.csv')
    trainset, testset = datasource.get_dataframes()
    voc = datasource.vocabulary
    model = TabularNaiveBayesClassifier(vocabulary=voc)
    print(model.parameters)
    model.fit(trainset)
    print(model.parameters)


