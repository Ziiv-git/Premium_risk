# importing libraries
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# loading the datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# filling the missing values
train['application_underwriting_score'] = train['application_underwriting_score'
].fillna(train['application_underwriting_score'].mean())

train['Count_3-6_months_late'] = train['Count_3-6_months_late'].fillna(
    train['Count_3-6_months_late'].median())

train['Count_6-12_months_late'] = train['Count_6-12_months_late'].fillna(
    train['Count_6-12_months_late'].median())

train['Count_more_than_12_months_late'] = train['Count_more_than_12_months_late'
].fillna(train['Count_more_than_12_months_late'].median())

test['application_underwriting_score'] = test['application_underwriting_score'
].fillna(test['application_underwriting_score'].mean())

test['Count_3-6_months_late'] = test['Count_3-6_months_late'].fillna(
    test['Count_3-6_months_late'].median())

test['Count_6-12_months_late'] = test['Count_6-12_months_late'].fillna(
    test['Count_6-12_months_late'].median())

test['Count_more_than_12_months_late'] = test['Count_more_than_12_months_late'
].fillna(test['Count_more_than_12_months_late'].median())


# dropping the id column from from both the train and test data set
train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)


# splitting into dependent and independent features
x = train.drop(['target'], axis=1)
y = train['target']


# one-hot catogorical columns
x = pd.concat([x, pd.get_dummies(train[['sourcing_channel',
                                               'residence_area_type']])],axis=1)
x.drop(['sourcing_channel', 'residence_area_type'], axis=1, inplace=True)

test = pd.concat([test, pd.get_dummies(test[['sourcing_channel',
                                              'residence_area_type']])],axis=1)
test.drop(['sourcing_channel', 'residence_area_type'], axis=1, inplace=True)



from sklearn.preprocessing import StandardScaler
# define standard scaler
scaler = StandardScaler()
# transform data
x = pd.DataFrame(scaler.fit_transform(x))
test = pd.DataFrame(scaler.transform(test))

#splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=4,
                                                    stratify=y)

# building the model and training
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
logr = LogisticRegression(max_iter=250)
logr.fit(x_train, y_train)

#saving the model to disk
pickle.dump(logr, open('model.pkl', 'wb'))
# loading the model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
print(pred = logr.predict(x_test))
