import pandas as pd


class SpamDatasource(object):
    def __init__(self, data_path, mode='train', classes=['spam', 'ham'], train_test_split=0.8):
        self._data_path = data_path
        self._df = pd.read_csv(
            self._data_path, sep='\t', header=None, names=['Label', 'SMS'])
        self._split = train_test_split
        self._classes = classes
    
    @property
    def vocabulary(self):
        return self._vocabulary
    
    @property
    def classes(self):
        return self._classes
    
    def get_dataframes(self):

        # shuffle the data entries
        data_randomized = self._df.sample(frac=1, random_state=1)

        # split and reset indices
        training_test_index = round(len(data_randomized) * self._split)
        training_set = data_randomized[:training_test_index].reset_index(drop=True)
        test_set = data_randomized[training_test_index:].reset_index(drop=True)

        # cleaning

        training_set['SMS'] = training_set['SMS'].str.replace('\W', ' ', regex=True) # Removes punctuation
        training_set['SMS'] = training_set['SMS'].str.lower()

        # Split SMS sequences into entities and save unique enitities

        training_set['SMS'] = training_set['SMS'].str.split()
        vocabulary = []
        for sms in training_set['SMS']:
            for word in sms:
                vocabulary.append(word)
        vocabulary = list(set(vocabulary))
        self._vocabulary = vocabulary
        
        # prepare grid for storing entity frequencies

        word_counts_per_sms = {unique_word: [0] * len(training_set['SMS']) for unique_word in vocabulary}

        # fill in the grid and convert to dataframe

        # TODO can this be parallized or use lookup frequency table only when necessary

        for index, sms in enumerate(training_set['SMS']):
            for word in sms:
                word_counts_per_sms[word][index] += 1
        
        word_counts = pd.DataFrame(word_counts_per_sms)
        
        training_set_clean = pd.concat([training_set, word_counts], axis=1)
        return training_set_clean, test_set


if __name__ == '__main__':
    import logging
    params = ['/home/spacepanda/workspace/projects/classical-ml/dataset/spam.csv']
    datasource = SpamDatasource(*params)
    train, test = datasource.get_dataframes()
    print(train.head())
    print(test.head())