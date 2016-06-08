#!/usr/bin/env python3
import numpy as np
import pandas
import sklearn
from sklearn.ensemble import RandomForestClassifier


data = pandas.read_csv('train.csv')
labels = data['label'].values
features = data.iloc[:,1:].values

#label_train, label_test, feat_train, feat_test = sklearn.cross_validation.train_test_split(labels, features, test_size=0.2, random_state=0)
#for i in range(5):
#    rf_classifier = RandomForestClassifier(n_estimators=1000, n_jobs=12, criterion='entropy')
#    rf_classifier.fit(feat_train, label_train)
#    print(rf_classifier.score(feat_train, label_train), rf_classifier.score(feat_test, label_test))

rf_classifier = RandomForestClassifier(n_estimators=1000, n_jobs=7, criterion='gini')
rf_classifier.fit(features, labels)

test_data = pandas.read_csv('test.csv').values
predictions = rf_classifier.predict(test_data)

print 'ImageId,Label'
for i in enumerate(predictions):
    print(str(i[0]+1) + ',' + str(i[1]))

