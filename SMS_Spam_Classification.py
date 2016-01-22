__author__ = 'Vineets'

# Spam filterning classification problem using Logistic Regression

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv('SMSSpamCollection', delimiter='\t', header=None)

print df.head()
print "Number of spam messages: ", df[df[0] == 'spam'][0].count()
print "Number of ham messages: ", df[df[0] == 'ham'][0].count()

"""
df[0] contains the labels
df[1] contains the text message
"""
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1], df[0])

# TfidfVectorizer combines CountVectorizer and TfidfTransformer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)


# Implementing logistic regression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
confusionmatrix = confusion_matrix(y_test, predictions)
print confusionmatrix

scores = cross_val_score(classifier, X_train, y_train, cv=5)
precisions = cross_val_score(classifier, X_train, y_train, cv=5, scoring='precision')
recalls = cross_val_score(classifier, X_train, y_train, cv=5, scoring='recall')
f1s = cross_val_score(classifier, X_train, y_train, cv=5, scoring='f1')
print 'Cross validation score: ', np.mean(scores)
print 'Precision: ', np.mean(precisions)
print 'Recall: ', np.mean(recalls)
print 'F1:', np.mean(f1s)
