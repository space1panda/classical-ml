import pandas as pd


class SpamDatasource(object):
    def __init__(self, data_path, mode='train'):
        self._data_path = data_path
        self._df = pd.read_csv(
            self._data_path, sep='\t', header=None, names=['Label', 'SMS'])
    
    def get_dataframes(self):

        # TODO: refactor

        data_randomized = self._df.sample(frac=1, random_state=1)
        training_test_index = round(len(data_randomized) * 0.8)
        training_set = data_randomized[:training_test_index].reset_index(drop=True)

        # cleaning

        training_set['SMS'] = training_set['SMS'].str.replace('\W', ' ') # Removes punctuation
        training_set['SMS'] = training_set['SMS'].str.lower()

        training_set['SMS'] = training_set['SMS'].str.split()

        vocabulary = []
        for sms in training_set['SMS']:
            for word in sms:
                vocabulary.append(word)
        vocabulary = list(set(vocabulary))
        
        word_counts_per_sms = {unique_word: [0] * len(training_set['SMS']) for unique_word in vocabulary}

        for index, sms in enumerate(training_set['SMS']):
            for word in sms:
                word_counts_per_sms[word][index] += 1
        
        word_counts = pd.DataFrame(word_counts_per_sms)
        training_set_clean = pd.concat([training_set, word_counts], axis=1)
        test_set = data_randomized[training_test_index:].reset_index(drop=True)
        return training_set_clean, test_set


if __name__ == '__main__':
    import logging
    params = ['/home/spacepanda/workspace/projects/classical-ml/dataset/spam.csv']
    datasource = SpamDatasource(*params)
    train, test = datasource.get_dataframes()
    print(train.head())
    print(test.head())