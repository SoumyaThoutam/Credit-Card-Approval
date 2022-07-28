# -*- coding: utf-8 -*-
"""CreditCardScoring.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-zE9V2nvKbE7y_qlhEs64c3gHnJbgTof
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools


# Load dataset
columns = ['is_male', 'age', 'debt', 'married', 'bank_customer','education_level', 
           'ethnicity', 'years_employed', 'prior_default', 'employed', 'credict_Score', 
           'drivers_license', 'citizen', 'zipcode', 'income', 'approved']
cc_apps = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ml-project/data/crx.data', header=None, names= columns)
# Drop the features drivers_license and zipcode
cc_apps = cc_apps.drop(['drivers_license', 'zipcode'], axis=1)

# Inspect data
cc_apps.head()

cc_apps.loc[cc_apps['age']=='?']['age']

cc_apps['age'] = cc_apps['age'].replace('?', np.NAN)
cc_apps = cc_apps.astype({'age':'float'})

# Print summary statistics
cc_apps_description = cc_apps.describe()
print(cc_apps_description)

print("\n")

# Print DataFrame information
cc_apps_info = cc_apps.info()
print(cc_apps_info)

print("\n")

# Inspect missing values in the dataset
print(cc_apps.tail(74))

# Import numpy
import numpy as np

# Inspect missing values in the dataset
print(cc_apps.tail(17))

# Replace the '?'s with NaN
cc_apps = cc_apps.replace('?', np.nan)

# Inspect the missing values again
print(cc_apps.tail(17))

# Impute the missing values with mean imputation
cc_apps.fillna(cc_apps.mean(), inplace=True)

# Count the number of NaNs in the dataset to verify
cc_apps.isnull().values.sum()

# Iterate over each column of cc_apps
for col in cc_apps.columns:
    # Check if the column is of object type
    if cc_apps[col].dtypes == 'object':
        # Impute with the most frequent value
        cc_apps = cc_apps.fillna(cc_apps[col].value_counts().index[0])

# Count the number of NaNs in the dataset and print the counts to verify
cc_apps.isnull().values.sum()

# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Instantiate LabelEncoder
le = LabelEncoder()

# Iterate over all the values of each column and extract their dtypes
for col in cc_apps.columns:
    # Compare if the dtype is object
    if cc_apps[col].dtype=='object':
    # Use LabelEncoder to do the numeric transformation
        cc_apps[col]=le.fit_transform(cc_apps[col])

cc_apps_df = cc_apps.copy()
cc_apps_df.head()

# Import train_test_split
from sklearn.model_selection import train_test_split


cc_apps = cc_apps.values

# Segregate features and labels into separate variables
X,y = cc_apps[:,0:13] , cc_apps[:,13]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                y,
                                test_size=0.33,
                                random_state=42)

# Import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# Instantiate MinMaxScaler and use it to rescale X_train and X_test
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)

rescaledX_train.shape

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Instantiate a LogisticRegression classifier with default parameter values
logreg = LogisticRegression(C=0.8,
                           random_state=0,
                           solver='lbfgs')

# Fit logreg to the train set
logreg.fit(rescaledX_train, y_train)

# Import confusion_matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Use logreg to predict instances from the test set and store it
y_pred = logreg.predict(rescaledX_test)

print('Accuracy Score of lr is {:.5}'.format(accuracy_score(y_test, y_pred)))
print(pd.DataFrame(confusion_matrix(y_test,y_pred)))

sns.set_style('white') 
class_names = ['0','1']
plot_confusion_matrix(confusion_matrix(y_test,y_pred),
                      classes= class_names, normalize = True, 
                      title='Normalized Confusion Matrix: Logistic Regression')

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Define the grid of values for tol and max_iter
tol = [0.01, 0.001, 0.0001]
max_iter = [100, 150, 200]

# Create a dictionary where tol and max_iter are keys and the lists of their values are corresponding values
param_grid = dict(tol=tol, max_iter=max_iter)

# Instantiate GridSearchCV with the required parameters
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)

# Use scaler to rescale X and assign it to rescaledX
rescaledX = scaler.fit_transform(X)

# Fit data to grid_model
grid_model_result = grid_model.fit(rescaledX, y)

# Summarize results
best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print("Best: %f using %s" % (best_score, best_params))

print("Best: %f using %s" % (best_score, best_params))

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

dt_model = DecisionTreeClassifier()
dt_model.fit(rescaledX_train, y_train)
dt_y_predict = dt_model.predict(rescaledX_test)



print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, dt_y_predict)))
print(pd.DataFrame(confusion_matrix(y_test,dt_y_predict)))

sns.set_style('white') 
class_names = ['0','1']
plot_confusion_matrix(confusion_matrix(y_test,dt_y_predict),
                      classes= class_names, normalize = True, 
                      title='Normalized Confusion Matrix: DecisionTreeClassifier')

dt_param_dict = {
    "criterion":['gini', 'entropy'],
    "max_depth": range(1,5),
    "min_samples_split": range(2,4),
    "min_samples_leaf": range(1,4)
}


dt_grid = GridSearchCV(dt_model,
                       param_grid=dt_param_dict
                       ,cv=10 )

dt_grid_model_result = dt_grid.fit(rescaledX_train, y_train)
# Summarize results
dt_best_score, dt_best_params = dt_grid_model_result.best_score_, dt_grid_model_result.best_params_
print("Best: %f using %s" % (dt_best_score, dt_best_params))

dt_best_params

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=250,
                              max_depth=12,
                              min_samples_leaf=16
                              )
rf_model.fit(X_train, y_train)
rf_y_predict = rf_model.predict(rescaledX_test)

print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, rf_y_predict)))
print(pd.DataFrame(confusion_matrix(y_test,rf_y_predict)))

sns.set_style('white') 
class_names = ['0','1']
plot_confusion_matrix(confusion_matrix(y_test,rf_y_predict),
                      classes= class_names, normalize = True, 
                      title='Normalized Confusion Matrix: RandomForestClassifier')

rf_param_dict = {
    "criterion":['gini', 'entropy'],
    "max_depth": range(1,5),
    "min_samples_split": range(2,4),
    "min_samples_leaf": range(1,4)
}


rf_grid = GridSearchCV(rf_model,
                       param_grid=rf_param_dict
                       ,cv=10 )

rf_grid_model_result = rf_grid.fit(rescaledX_train, y_train)
# Summarize results
rf_best_score, rf_best_params = rf_grid_model_result.best_score_, rf_grid_model_result.best_params_
print("Best: %f using %s" % (rf_best_score, rf_best_params))

from sklearn import svm

svm_model = svm.SVC(C = 0.8,
                kernel='linear')
svm_model.fit(X_train, y_train)
svm_y_predict = model.predict(X_test)

svm_accuracy = print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, svm_y_predict)))
print(pd.DataFrame(confusion_matrix(y_test,svm_y_predict)))

plot_confusion_matrix(confusion_matrix(y_test,svm_y_predict),
                      classes=class_names, normalize = True, 
                      title='Normalized Confusion Matrix: SVM')

# Dataset generation
data_dict = {'LR':85, 'DT':86, 'RF':87, 'SVM':83}
Algorithms = list(data_dict.keys())
Accuracy = list(data_dict.values())
fig = plt.figure(figsize = (5, 5))
#  Bar plot
plt.bar(Algorithms, Accuracy, color ='grey',
        width = 0.6)
plt.xlabel("Algorithms Performed")
plt.ylabel("Accuracy scores of Algos")
plt.title("Comparing algorithms")
plt.show()