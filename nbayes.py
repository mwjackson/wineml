import pandas as pd
import numpy as np

#df = pd.read_csv('winequality-red.csv', sep=';')
df = pd.read_csv('winequality-white.csv', sep=';')

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

Y = df['quality']
X = df.drop('quality', axis=1)

# naive bayes
# scale to 0-1 eg. MinMaxScaler().fit_transform(X)
pipeline = Pipeline([
  ('scale', MinMaxScaler()),
  ('clf', GaussianNB())
])

# random forest
pipeline = Pipeline([
  ('scale', MinMaxScaler()),
  ('clf', RandomForestClassifier())
])

# logistic regression
#pipeline = Pipeline([
#  ('scale', MinMaxScaler()),
#  ('clf', LogisticRegression())
#])


# shortcut version, only gives you 1 metrics eg. precision
#from sklearn.cross_validation import cross_val_score
#scores = cross_val_score(pipeline, X, Y, scoring='f1')
#scores = cross_val_score(pipeline, X, Y, scoring='f1_weighted', cv=cv_data)

# do cross validation manually
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

cv_data = StratifiedShuffleSplit(Y, n_iter=10, test_size=0.10)

results = []
conf_matrix = []

# for each iteration of cross validation, select the data, 
# fit the model with training set, predict with test set, and calculate the metrics
for train_index, test_index in cv_data:
    train_X, test_X = X[train_index], X[test_index]
    train_Y, test_Y = Y[train_index], Y[test_index]

    pipeline.fit(train_X, train_Y)
    predicted_Y = pipeline.predict(test_X)
    
    results.append(precision_recall_fscore_support(test_Y, predicted_Y, labels=range(1, 11)))
    conf_matrix.append(confusion_matrix(test_Y, predicted_Y, labels=range(1, 11)))
    
precision = pd.DataFrame([r[0] for r in results]).mean()
recall = pd.DataFrame([r[1] for r in results]).mean()
f1 = pd.DataFrame([r[2] for r in results]).mean()
support = pd.DataFrame([r[3] for r in results]).sum()

print 'f1:\n{0}\nprecision:\n{1}\nrecall:\n{2}\nsupport:\n{3}'.format(precision, recall, f1, support)

# re-fit the model with all the data and predict a single point
pipeline.fit(X, Y)
test_point = pd.DataFrame([[7.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0]])
print pipeline.predict_proba(test_point)