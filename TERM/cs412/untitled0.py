import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


testdf= pd.read_csv('test.csv')
traindf= pd.read_csv('train.csv')


traindf['age'] = 2021 - traindf['year']
testdf['age'] = 2021 - testdf['year']
traindf.drop('year', inplace=True, axis=1)
testdf.drop('year', inplace=True, axis=1)

X_train = traindf.drop(['price'], axis='columns')
y_train = traindf['price']

cols_for_none = ('model','brand','fuelType')
for c in cols_for_none:
    X_train[c] = X_train[c].fillna("None") 
    testdf[c] = testdf[c].fillna("None")
    
cols_for_zero = ('age','mileage','mpg','tax','tax(£)')
for c in cols_for_zero:
    X_train[c] = X_train[c].fillna(0.0)
    testdf[c] = testdf[c].fillna(0.0) 
    
cols_for_mode = ('engineSize','transmission')
for c in cols_for_mode:
    X_train[c] = X_train[c].fillna(X_train[c].mode())
    testdf[c] = testdf[c].fillna(testdf[c].mode())    
    
X_train['engineSize'] = X_train['engineSize'].fillna(2.0)
X_train['transmission'] = X_train['transmission'].fillna('automatic')

testdf['engineSize'] = testdf['engineSize'].fillna(2.0)
testdf['transmission'] = testdf['transmission'].fillna('automatic')


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cols = ['model','fuelType','transmission','brand',
        'age','mileage','mpg','engineSize','tax']

for c in cols:
    le.fit(list(X_train[c].values))
    X_train[c] = le.transform(list(X_train[c].values))
    
for c in cols:
    le.fit(list(testdf[c].values))
    testdf[c] = le.transform(list(testdf[c].values)) 

X_train.drop('tax(£)', inplace=True, axis=1)
testdf.drop('tax(£)', inplace=True, axis=1)
X_train.drop('ID', inplace=True, axis=1)
testdf.drop('ID', inplace=True, axis=1)


#Show the important attributes in descending order
best_features = SelectKBest(score_func=f_regression, k=9)
top_features = best_features.fit(X_train,y_train)
scores = pd.DataFrame(top_features.scores_)
columns = pd.DataFrame(X_train.columns)
featureScores = pd.concat([columns, scores], axis=1)
featureScores.columns = ['Features','Scores']
print(featureScores.nlargest(11, 'Scores'))

rr = RandomForestRegressor(n_estimators= 1000, random_state=100)
gbr = GradientBoostingRegressor()
#Create function to displaying scores
def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())
    
#Training the Random Forest Regressor
print("Random Forest Regressor Scores")
scores = cross_val_score(rr, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
random_forest_scores = np.sqrt(-scores)
display_scores(random_forest_scores)
print("\n")