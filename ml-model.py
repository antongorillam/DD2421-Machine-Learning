# !pip install imblearn
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sb
from imblearn.over_sampling import SMOTE
'''
Handling and cleaning the training data
'''
train_df = pd.read_csv (r'TrainOnMe-2.csv')
train_df=train_df.dropna() # Drop rows with NaN
train_df = train_df[~train_df.x6.str.contains("https")] # There is a YT link on x6 (https://youtu.be/0q_Bbd2SGtY), dropping it

train_df['x6'] = train_df['x6'].astype(float)
train_df = train_df.drop(['Unnamed: 0'], axis=1)
train_df['x12'].map({True : 1, False : 0})
train_df['x12'] = train_df['x12'].astype(float)

'''Creates dummy variable on x7, but not x12, since it is boolean, we keep it at float'''
feature_drop = ['x7','x1','x2','x6','x3', 'Erik Sven Williams', 'Jerry från Solna', 'Jerry Williams', 'Erik Sven Fernström', 'Jerry Fernström']
# feature_drop = ['x7']
x7_columns_dummy = pd.get_dummies(train_df['x7'])
train_df = pd.concat((train_df, x7_columns_dummy), axis=1)
train_df = train_df.drop(feature_drop, axis=1)


                              
'''Make y categorical random variable'''
y = train_df['y']
X = train_df.drop('y', axis=1)

'''Check for correlation on the features'''
# corr = train_df.corr()
# sb.heatmap(corr, annot=True)


'''Splitting the data into test and train data set'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

'''With Cross validation library'''
from scipy.stats import zscore
z_scores = np.abs(zscore(X))
filtered_entries = (z_scores < 2.9).all(axis=1)
X = X[filtered_entries] 
y = y[filtered_entries] 


'''Use SMOTH to syntetically make up för class imbalances'''
sm = SMOTE(random_state=0)
X, y = sm.fit_resample(X, y)
from sklearn.model_selection import cross_val_score


'''Random Forrest'''
from sklearn import ensemble
rf_classifier = ensemble.RandomForestClassifier(n_estimators=1000)
rf_score = cross_val_score(rf_classifier, X, y)
print(f'rf_score (cross_val): {rf_score.mean()}')


'''Remove outliers'''
from scipy.stats import zscore
z_scores = np.abs(zscore(X_train))
filtered_entries = (z_scores < 2.9).all(axis=1)
X_train = X_train[filtered_entries] 
y_train = y_train[filtered_entries] 


'''Use SMOTH to syntetically make up för class imbalances'''
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=0)
X_train, y_train = sm.fit_resample(X_train, y_train)


'''Feature Selection with information gains'''
from sklearn.feature_selection import mutual_info_classif
mutual_info = mutual_info_classif(X_train, y_train)
mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
mutual_info.sort_values(ascending=False)


'''Random Forrest'''
from sklearn import ensemble
rf_classifier = ensemble.RandomForestClassifier(n_estimators=1000, random_state=0)
rf_classifier = rf_classifier.fit(X_train, y_train)
rf_score = rf_classifier.score(X_test, y_test)
print(f'rf_score: {rf_score}')


'''Evaluating and printing final anwser'''
x_eval = pd.read_csv (r'EvaluateOnMe-2.csv')
x_eval['x6'] = x_eval['x6'].astype(float)
x_eval = x_eval.drop(['Unnamed: 0'], axis=1)
x_eval['x12'].map({True : 1, False : 0})
x_eval['x12'] = x_eval['x12'].astype(float)


x7_columns_dummy = pd.get_dummies(x_eval['x7'])
x_eval = pd.concat((x_eval, x7_columns_dummy), axis=1)
x_eval = x_eval.drop(feature_drop, axis=1)

y_pred = rf_classifier.predict(x_eval)


with open('eval_label.txt', 'w') as f:
    for label in y_pred:
        f.write("%s\n" % label)