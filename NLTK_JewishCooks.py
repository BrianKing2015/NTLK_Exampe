import nltk
import random

from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

from nltk.tokenize import word_tokenize, sent_tokenize

"""
The goal is to determine whether a recipe from project Gutenberg's cookbook repository is Jewish or not.
These texts range from 1846 to 1919

Link to bookshelf: 
http://www.gutenberg.org/wiki/Cookery_(Bookshelf)
Links to 4 texts:
http://www.gutenberg.org/ebooks/31534
http://www.gutenberg.org/ebooks/8542
http://www.gutenberg.org/ebooks/12350
http://www.gutenberg.org/ebooks/12327



Example is mostly just to work some of the challenges using NLTK and sklearn libraries. 
Initially take 4 cookbooks downloaded from Gutenberg.
Use the sentence tokenizer to turn them into features (TODO better system to parse into full recipe)
Use the word tokenizer to find the 5000 most common words so they can be used to sort 
Start out using Naive Bayes directly from NLTK.
Then use a few algorithms from sklearn. 
Use a vote system to see if there is any different between the different algorithms.

TODO: 
Parse the text so that each full recipe is its own feature
Stanardize the measurements and terms. To remove them as defining features for each book

"""

# Takes a variable number of classifiers (5) as input and return whether the classifiers all agree and with what confidence
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
        

# Read in all 4 books, create sentence level objects out of the books
jewish = open("JewCook.txt","r").read()
jewish1 = open("JewIntCook.txt","r").read()
notJ = open("SundayCook.txt","r").read()
notJ1 = open("MomCook.txt","r").read()
jewish_sent = sent_tokenize (jewish)
jewish_sent1 = sent_tokenize (jewish1)
notJ_sent = sent_tokenize (notJ)
notJ_sent1 = sent_tokenize (notJ1)

# Tag each sentence as coming from either a Jewish cookbook or a non-Jewish cookbook while adding them to documents variable
documents = []
for r in jewish_sent:
    documents.append( (r, "Jew") )

for r in jewish_sent1:
    documents.append( (r, "Jew") )

for r in notJ_sent:
    documents.append( (r, "Not") )

for r in notJ_sent1:
    documents.append( (r, "Not") )

# Create a variable to contain all of the words (all_words) in all cookbooks, then tokenize all cookbooks
all_words = []
jewish_words = word_tokenize(jewish)
jewish_words1 = word_tokenize(jewish1)
notJ_words = word_tokenize(notJ)
notJ_words1 = word_tokenize(notJ1)

# Add all of the words in lowercase to the all_words variable
for w in jewish_words:
    all_words.append(w.lower())

for w in jewish_words1:
    all_words.append(w.lower())

for w in notJ_words:
    all_words.append(w.lower())

for w in notJ_words1:
    all_words.append(w.lower())

# Check the frequency within the all_words variable
all_words = nltk.FreqDist(all_words)

# Take only the most common 5000 words
word_features = list(all_words.keys())[:5000]

# Looks for the most common words in the Documents (sentence level objects) 
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# Calls the find_features function in a list comprehension to create a featureset
featuresets = [(find_features(rev), category) for (rev, category) in documents]

# prints the total number of features so that the training and testing sizes can be set correctly, then shuffles them so it isn't 100% Jewish then 100% not
print (len(featuresets))
random.shuffle(featuresets)

# Creates the training and testing data sets:      
training_set = featuresets[:12000]
testing_set =  featuresets[12000:]


# Calls each of the classifiers, gives them the training data and then the testing data.
# Each one returns accuracy level 
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
# returns the most informative words within the set
classifier.show_most_informative_features(50)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


voted_classifier = VoteClassifier(
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)