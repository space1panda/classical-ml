import pandas as pd
import os
from models.naive_bayes import NaiveBayesClassifier
from data_processing.spam import SpamDatasource

# prepare dataframes

root = '/home/spacepanda/workspace/projects/classical-ml/'
dataset_path = 'dataset/spam.csv'
split_ratio = 0.8
random_seed = 1

df = pd.read_csv(
    os.path.join(root, dataset_path), sep='\t', header=None, names=['Label','SMS'])

shuffled_df = df.sample(frac=1, random_state=1)
train_test_index = round(len(shuffled_df) * split_ratio)
train_df = shuffled_df[:train_test_index]
test_df = shuffled_df[train_test_index:].reset_index(drop=True)

train_datasouce = SpamDatasource(train_df)
test_datasource = SpamDatasource(test_df, test_mode=True)

model = NaiveBayesClassifier(train_datasouce.vocabulary)

for input in train_datasouce:
    model(input)

model.eval = True

for input in test_datasource:
    model(input)

datasource = SpamDatasource(
    data_path=)

print(datasource[0])
# trainset, testset = datasource.get_dataframes()
# voc = datasource.vocabulary
# model = NaiveBayesClassifier(vocabulary=voc, num_classes=2)
# model.fit(trainset)
# model.evaluate(testset)
# message = 'Meet you downstairs'
# label = model.predict(message)
# print(message)
# print(datasource.classes[label])