"""
Naive Bayes implementation for solving SMS spam filter task. Based on the KDNuggets article:
https://www.kdnuggets.com/2020/06/naive-bayes-algorithm-everything.html 
"""

import numpy as np
import os


# 1. Import data corpus

data_root = '/home/spacepanda/workspace/real_data'
f_name = 'SMSSpamCollection'
num_classes = 2

with open(os.path.join(data_root, f_name), 'r') as fr:
    data_corpus = fr.readlines()

"""
The Spam data consists of examples of word sequences classified as spam ('spam') or no-spam ('ham'). Our task is to learn
the posterior probability of the sequence being spam or no-spam. To simplify the modelling of the likelihood of the Bayes' Theorem,
we will use Naive Bayes assumption, which implies conditional independece of the events. Given that, the joint likelihood of that message belongs
to some class, simplifies to the product of message feature frequencies in the given class.
"""

# 2. Define the func to calculate posterior distribution. We use a classic Bayes Theorem for each class:

get_class_posterior = lambda class_prior, features_likelihood: class_prior * features_likelihood

# Our final classifier is an argmax of the predicted posterior probs of each class

def nb_classifier(num_classes, class_priors, input):
    class_probs = []
    for c in range(num_classes):
        class_probs.append(get_class_posterior(class_priors[c], get_likelihood(input, c)))
    return np.argmax(class_probs)

"""
In accordance with the Bayes Theorem, we are updating our prior belief about the class (which is simply a frequency of the class in the
data observed)
with the evidence that particular words (features) have higher chance of appearance in the class. The marginal probability
of the evidence is dropped down  because it's constant for all classes of the dataset, and serves only as a normalization factor.
We may or may not add it, it's irrelevant for classes scoring. Given the input features (word sequence), we can then calculate the posterior
probability for each class
"""

def get_likelihood(features, lookup, c):
    total_lh = 1
    for feature in features:
        f_lh = lookup[feature, c]
        total_lh *= f_lh
    return total_lh


# Now, let's get back to data and make some preparations. We should arrive at having training/test sets as well as likelihood tables
# for each unique feature (word) in corpus;

import pandas as pd

sms_spam = pd.read_csv(
    os.path.join(data_root, f_name), sep='\t', header=None, names=['Label', 'SMS'])
class_priors = sms_spam['Label'].value_counts(normalize=True)
print(sms_spam)
print(dict(class_priors))

for prior in class_priors:
    print(prior)
  

