from __future__ import division
import os  # operating system commands
import pandas as pd
import numpy as np
from numpy import log #F test
from sklearn import metrics  
from sklearn.tree import DecisionTreeClassifier  # CART Classifier
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt  # 2D plotting
import statsmodels.api as sm  # logistic regression
import statsmodels.formula.api as smf  # R-like model specification
from patsy import dmatrices  # translate model specification into design matrices
from sklearn import svm  # support vector machines
from sklearn.ensemble import RandomForestClassifier  # random forest
import pdb
from sklearn import pipeline
from sklearn import cross_validation
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.preprocessing import Imputer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC
from sklearn.decomposition import PCA
from sklearn import preprocessing
from math import sqrt
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals.six import StringIO 
from sklearn import tree
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
import pylab as p 
from datetime import datetime
from multiprocessing import Pool

# class for debugging errors
class MyObj(object):
    def __init__(self, num_loops):
        self.count = num_loops

    def go(self):
        for i in range(self.count):
            pdb.set_trace()
            print i
        return
# Time the script; probably need to add Multiprocessing Module to speed up
startTime = datetime.now()

dir=('./')

twitter_df = pd.read_csv(dir + 'twitter_df.csv')
twitter_df.source = twitter_df.source.fillna('Unknown')
twitter_df.time_zone = twitter_df.time_zone.fillna('Unknown')
twitter_df.location = twitter_df.location.fillna('Not_provided')
twitter_df.to_csv('dataframe.csv', index=False, parse_dates=['tmstamp'])
# Use the aggregate DataFrame to perform Linear Regression on Response of Retweet Counts
train, test = train_test_split(twitter_df, test_size=.3, random_state=0)

testing = pd.DataFrame(test, columns = ['index', 'follower_count', 'friend_count', 'geo', 'location', 'name', 'place', 'retweet_count', 'screen_name', 'source', 'status_id', 'status_text', 'time_zone', 'timestamp', 'surge_pricing', 'free_rides', 'promo', 'driver', 'food', 'controversy', 'regulations'])
training = pd.DataFrame(train, columns = ['index', 'follower_count', 'friend_count', 'geo', 'location', 'name', 'place', 'retweet_count', 'screen_name', 'source', 'status_id', 'status_text', 'time_zone', 'timestamp', 'surge_pricing', 'free_rides', 'promo', 'driver', 'food', 'controversy', 'regulations'])
training['follower_count_l'] = training['follower_count'].replace(0, 1e-6)
training['follower_count_l'] = np.log(training['follower_count_l'])
testing['follower_count_l'] = testing['follower_count'].replace(0, 1e-6)
testing['follower_count_l'] = np.log(testing['follower_count_l'])
training['friend_count_l'] = training['friend_count'].replace(0, 1e-6)
training['friend_count_l'] = np.log(training['friend_count_l'])
testing['friend_count_l'] = testing['friend_count'].replace(0, 1e-6)
testing['friend_count_l'] = np.log(testing['friend_count_l'])
le=preprocessing.LabelEncoder()
training['time_zone_l'] = le.fit(training['time_zone'])
testing['time_zone_l'] = le.fit(testing['time_zone'])
training['location_l'] = le.fit(training['location'])
testing['location_l'] = le.fit(testing['location'])
training['source_l'] = le.fit(training['source'])
testing['source_l'] = le.fit(testing['source'])
training['retweet_count_l'] = np.log(training['retweet_count'].replace(0, 1e-6))
testing['retweet_count_l'] = np.log(testing['retweet_count'].replace(0, 1e-6))

# taking a look at the distribution of retweet count, in order to discern a breaking point
# training['retweet_count'].hist(bins=30)
# p.show()
# training['retweet_count_l'].hist(bins=30)
# p.show()
# testing['retweet_count'].hist(bins=30)
# p.show()
# testing['retweet_count_l'].hist(bins=30)
# p.show()

"""
#statmodels OLS first
y, X = dmatrices('retweet_count_l ~ surge_pricing + free_rides + promo + driver + food + controversy + regulations', data=training, return_type='dataframe')
# Define the model from above Patsy-created variables, using Statsmodels
#print sm.OLS(y,X).fit().summary()
#print sm.OLS(y,X).fit().params
#print 'r sqd is : ', sm.OLS(y,X).fit().rsquared
rainbow=sm.stats.linear_rainbow(sm.OLS(y,X).fit())
print 'Rainbow Test for Linearity is ', rainbow
y_hat, X_hat = dmatrices('retweet_count_l ~ surge_pricing + free_rides + promo + driver + food + controversy + regulations', data=testing, return_type='dataframe')
y_pred = sm.OLS(y,X).fit().predict(X_hat)
testing['retweet_pred_smols'] = pd.Series(y_pred)
"""
#make array adjustments for scikit learn
numeric_cols = ['friend_count_l', 'follower_count_l']
x_num_train = training[numeric_cols].as_matrix()
x_num_test = testing[numeric_cols].as_matrix()
cat_cols = ['time_zone', 'location', 'source']
cat_train = training[cat_cols]
cat_test = testing[cat_cols]

x_cat_train = cat_train.T.to_dict().values()
x_cat_test = cat_test.T.to_dict().values()
vectorizer = DV(sparse=False)
vec_x_cat_train = vectorizer.fit_transform(x_cat_train)
vec_x_cat_test = vectorizer.transform(x_cat_test)

x_train = np.hstack((x_num_train, vec_x_cat_train))
x_test = np.hstack((x_num_test, vec_x_cat_test))

target_train = np.array(training['retweet_count_l'])
target_test = np.array(testing['retweet_count_l'])


print "training regression..."
regr = linear_model.LinearRegression(normalize=True)
# Train the model using the training sets
regr.fit(x_train, target_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(x_test) - target_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x_test, target_test))

# The mean square error
print 'mse ', np.mean((regr.predict(x_test)-target_test)**2)
y_lr = regr.predict(x_test)
print y_lr.shape
testing['retweet_pred_scilr'] = pd.DataFrame(y_lr)

#Decision Tree Regression
print(__doc__)
clf = DecisionTreeRegressor(max_depth=7)
clf.fit(x_train, target_train)

y_fit_1 = clf.predict(x_test)
print 'cross val score for DecisionTreeRegressor'
print cross_val_score(clf, x_train, target_train, cv=10)
testing['retweet_pred_dtr'] = pd.DataFrame(y_fit_1)
plt.figure()
plt.plot(x_test, y_fit_1, c="r", linewidth=.5, alpha=.3)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()


#Random Forest
clf = RandomForestRegressor(n_estimators=200, max_depth=5, 
                            min_samples_leaf=3)
clf.fit(x_train, target_train)
y_fit_200 = clf.predict(x_test)
print 'cross val score for Random Forest'
print cross_val_score(clf, x_train, target_train, cv=10)
testing['retweet_pred_rf'] = pd.DataFrame(y_fit_200)
plt.figure()
plt.plot(x_train, target_train, '.k', alpha=0.3)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Random Forest")
plt.legend()
plt.show()
"""
# use binarization to enable use of different algorithms for the target, Retweet Count level of '30' will be
# the threshold from which we label 0 or 1 in the retweet_count labels
lb1 = preprocessing.Binarizer(threshold=(np.percentile((training['retweet_count']),95)))
lb2 = preprocessing.Binarizer(threshold=(np.percentile((testing['retweet_count']),95)))
newtarget_train = lb1.fit_transform(np.array(training['retweet_count']).astype(int))
newtarget_test = lb2.fit_transform(np.array(testing['retweet_count']).astype(int))
clf = RF(n_estimators=7)
t = clf.fit(x_train, newtarget_train)
y_fit = t.predict(x_test)
# print y_fit.shape
print "avg prec accuracy:", metrics.average_precision_score(newtarget_test, y_fit)
print "rocauc:", metrics.roc_auc_score(newtarget_test, y_fit)
testing['retweet_pred_brf'] = pd.DataFrame(y_fit)
print "mean of training retweet_count is:  ", training['retweet_count'].mean()
"""
twitter_df.to_csv(dir + 'twit_machine.csv')
testing.to_csv(dir + 'twit_machine_test.csv')

print "Time taken to solve: "
print datetime.now() - startTime
