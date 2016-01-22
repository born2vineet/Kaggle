__author__ = 'Vineets'

"""
Semi-Supervised learning problem
Learn features by clustering unlabeled data and
use the learned features to build a supervised classifier.
"""

import numpy as np
import mahotas as mh
from mahotas.features import surf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.cluster import MiniBatchKMeans
import glob

# Loading them images, coverting it to greyscale and extracting SURF
# SURF descriptors describe interesting regions of an image in ways that
# are invariant to scale, rotation, and illumination.
all_instances_filenames = []
all_instances_targets = []
for f in glob.glob('cats-and-dogs-img/*.jpg'):
    target = 1 if 'cat' in f else 0
    all_instances_filenames.append(f)
    all_instances_targets.append(target)
surf_features = []
counter = 0

for f in all_instances_filenames:
    print 'Reading image:', f
    image = mh.imread(f, as_grey=True)
    surf_features.append(surf.surf(image)[:, 5:])

train_len = int(len(all_instances_filenames) * .60)
X_train_surf_features = np.concatenate(surf_features[:train_len])
X_test_surf_features = np.concatenate(surf_features[train_len:])
y_train = all_instances_targets[:train_len]
y_test = all_instances_targets[train_len:]

# Extracting descriptors into 300 clusters using MiniBatchKmeans
print 'Clustering', len(X_test_surf_features), 'features'
estimator = MiniBatchKMeans(n_clusters=300)
estimator.fit_transform(X_test_surf_features)

# Creating a 300-dimensional feature vector for training and testing data
X_train = []
for instance in surf_features[:train_len]:
    clusters = estimator.predict(instance)
    features = np.bincount(clusters)
    if len(features) < 300:
        features = np.append(features, np.zeros((1, 300-len(features))))
    X_train.append(features)

X_test = []
for instance in surf_features[train_len:]:
    clusters = estimator.predict(instance)
    features = np.bincount(clusters)
    if len(features) < 300:
        features = np.append(features, np.zeros(1, 300 - len(features)))
        X_test.append(features)

# Training Logistic Regression Classifier

clf = LogisticRegression(C=0.01, penalty='l2')

clf.fit_transform(X_train, y_train)
predictions = clf.predict(X_test)
print classification_report(y_test, predictions)
print 'Precision: ', precision_score(y_test, predictions)
print 'Recall: ', recall_score(y_test, predictions)
print 'Accuracy: ', accuracy_score(y_test, predictions)


