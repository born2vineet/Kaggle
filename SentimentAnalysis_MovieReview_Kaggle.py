__author__ = 'Vineets'
"""
Kaggle Competition
Classify sentiments og phrases takem from movie reviews in Rotten Tomatoes
The Sentiment column contains the response variable.
Labels:
0 - negative
1 - somewhat negative
2 - neutral
3 - somewhat positive
4 - positive
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

"""
Data Exploration:

df = pd.read_csv('train.tsv', header=0, delimiter='\t')
print df.count()
print df.head()

print df['Sentiment'].describe()
print df['Sentiment'].value_counts()
print df['Sentiment'].value_counts() / df['Sentiment'].count()
"""

def main():
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression())
    ])
    parameters = {
        'vect__max_df': (0.25, 0.5),
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__use_idf': (True, False),
        'clf__C': (0.1, 1, 10)
    }
    df = pd.read_csv('train.tsv', header=0, delimiter='\t')
    X, y = df['Phrase'], df['Sentiment'].as_matrix()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    print "Best score: %0.3f" % grid_search.best_score_
    print "Best parameter set: "
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])

    predictions = grid_search.predict(X_test)

    print 'Accuracy: ', accuracy_score(y_test, predictions)
    print 'Confusion Matrix: ', confusion_matrix(y_test, predictions)
    print 'Classification Report:', classification_report(y_test,predictions)
if __name__ == '__main__':
    main()