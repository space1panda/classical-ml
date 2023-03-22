import pandas as pd
from .tabular import TabularDatasource


class SpamDatasource(TabularDatasource):
    """
    Datasource for processing SMS Spam collection dataset:
    https://archive.ics.uci.edu/ml/datasets/sms+spam+collection.
    Implemetation based on KDNuggets article:
    https://www.kdnuggets.com/2020/07/spam-filter-python-naive-bayes-scratch.html
    Datasource performs data wrangling using pandas. Additionally the object
    prepares lookup table with feature frequencies which can be used for
    statistical modelling (e.g. Naive Bayes)
    """

    def __init__(
            self, data_path, classes=['spam', 'ham'], train_test_split=0.8,
            eval=False):
        self._data_path = data_path
        self._split = train_test_split
        self._classes = classes
        self._eval = eval
        self.prepare_data()
    
    @property
    def eval(self):
        return self._eval
    
    eval.setter
    def eval(self, flag):
        self._eval = flag
    
    @property
    def vocabulary(self):
        return self._vocabulary

    def __getitem__(self, idx):
        if not self.eval:
            return self._vocabulary[idx]
        return self._test_set[idx]
        
    def __len__(self):
        if not self.eval:
            return len(self._train_set)
        return len(self._test_set)
    
    def dataframe_cleaning(self, df):
        df['SMS'] = df['SMS'].str.replace('\W', ' ', regex=True)
        df['SMS'] = df['SMS'].str.lower()

    def prepare_vocabulary(self, df):
        # Split SMS sequences into entities and save unique enitities
        vocabulary = []
        for sms in df['SMS']:
            for word in sms:
                vocabulary.append(word)
        vocabulary = list(set(vocabulary))
        self._vocabulary = vocabulary

    def prepare_data(self):
        data_randomized = self._df.sample(frac=1, random_state=1)
        train_test_index = round(len(self._df) * self._split)
        self.prepare_train_set(data_randomized, train_test_index)
        self.prepare_test_set(data_randomized, train_test_index)
    
    def prepare_train_set(self, df, split_idx):
        train_set = df[:split_idx]
        train_set = train_set.reset_index(drop=True)
        train_set['SMS'] = train_set['SMS'].str.split()
        self.dataframe_cleaning(train_set)
        self.prepare_vocabulary()
        
        # Prepare per sms num of occurencies of each word in vocabulary

        word_counts_per_sms = {
            unique_word: [0] * len(df['SMS']) for unique_word in
            self.vocabulary
            }
        for index, sms in enumerate(df['SMS']):
            for word in sms:
                word_counts_per_sms[word][index] += 1
        
        word_counts = pd.DataFrame(word_counts_per_sms)
        training_set_clean = pd.concat([train_set, word_counts], axis=1)
        self._train_set = {}
        for c in self._classes:
            self._train_set[c] = training_set_clean[
                training_set_clean['Label'] == c]

    def prepare_test_set(self, df, split_idx):
        self._test_set = df[split_idx:].reset_index(drop=True)


if __name__ == '__main__':
    import logging
    params = ['/home/spacepanda/workspace/projects/classical-ml/dataset/spam.csv']
    datasource = SpamDatasource(*params)
    train, test = datasource.get_dataframes()
    print(train.head())
    print(test.head())