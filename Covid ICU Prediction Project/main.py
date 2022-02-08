# CSCI-4105: Knowledge Discovery and Data Mining, Fall 2021
# Final Project: COVID-19 Dataset
# Nicholas Klapatch, Erick Garcia-Vargas, Thomas Olandt, Kyle Sacco

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, precision_score

# Preprocessing phase: Organize data for the classifier.

# Store spreadsheet data in a Pandas Dataframe for preprocessing.
dataframe = pd.read_excel("COVID_ICU_Prediction.xlsx", na_values=[''], header=0)

# Display a bar chart of how many rows in each window do not contain any nan values.
windows = ['0-2', '2-4', '4-6', '6-12', 'ABOVE_12']
counts = [dataframe[dataframe['WINDOW'] == window].dropna().count() for window in windows]
for i in range(len(windows)):
    plt.bar(x=windows[i], height=counts[i])
plt.title('Rows Containing Zero NAN Values')
plt.xlabel('Window (Hours)')
plt.ylabel('Frequency')
plt.show()

# Choose entry window with the most data points to work with (ABOVE_12).
dataframe = dataframe[dataframe['WINDOW'] == 'ABOVE_12']

# 25 most correlated features with an ICU visit, by absolute values.
print('\n25 most correlated features with ICU target variable:')
print('Feature Correlation with ICU')
print(dataframe.corr()['ICU'].sort_values(ascending=False, key=abs)[1:26])

# Extra credit option:
dataframe_50 = dataframe[(dataframe.corr()['ICU'].sort_values(ascending=False, key=abs).keys()[0:51])]
dataframe_50 = dataframe_50.dropna(axis=0)
# Renumber the indices to be [0, 1, 2, ..., len(dataframe_50) - 1]
dataframe_50 = dataframe_50.reset_index()

# Reduce dataframe to only contain features to be used in classifier.
dataframe = dataframe[['RESPIRATORY_RATE_DIFF_REL', 'BLOODPRESSURE_SISTOLIC_DIFF', 'HEART_RATE_DIFF', 'TEMPERATURE_DIFF_REL', 
                        'OXYGEN_SATURATION_MAX', 'ICU']]
# Drop rows containing nan values.
dataframe = dataframe.dropna(axis=0)
# Renumber the indices to be [0, 1, 2, ..., 383]
dataframe = dataframe.reset_index()

# Display a boxplot to identify if there are any outliers in the data (excluding target variable).
dataframe2 = dataframe.drop(['ICU', 'index'], axis=1)

labels = {}
for i in range(len(dataframe2.columns)):
    labels[i] = dataframe2.columns[i]

plt.boxplot(dataframe2)
plt.title('Outliers')
plt.xticks([i + 1 for i in range(len(labels))], list(labels.values()), rotation=15)
plt.show()

# Print outlier ICU values.
dataframe2 = dataframe[['HEART_RATE_DIFF', 'ICU']].copy()
print('\nOutlier HEART_RATE_DIFF ICU values:')
print(dataframe2.sort_values(ascending=False, by='HEART_RATE_DIFF')[0:30])

# Processing (Classification) phase. Create a model to predict ICU value of a record.

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# train_test_split accepts Pandas Dataframes, so each feature is stored in X, while the target
# variable is stored in y. 
X_train, X_test, y_train, y_test = train_test_split(dataframe.drop('ICU', axis=1), dataframe['ICU'])

print('\nTraining and test data split sizes:')
print('Size of X_train: %d' % (len(X_train)))
print('Size of X_test: %d' % (len(X_test)))

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# Use sklearn implementation of Logistic Regression, and its associated methods.
classifier = LogisticRegression(max_iter=500)
classifier.fit(X_train, y_train)

# Post processing phase. Analyze performance of classifier.

print('\nAnalysis of classifier performance:')
prediction = classifier.predict(X_test)
print('Accuracy of classifier:  %.2f' % (classifier.score(X_test, y_test)))
print('Precision of classifier: %.2f' % (precision_score(prediction, y_test)))
print('Recall of classifier:    %.2f' % (recall_score(prediction, y_test)))

# Print Confusion Matrix:
cm = confusion_matrix(prediction, y_test)
print('\n%24s' % ('Predicted Class'))
print('%19s%5s' % ('0', '1'))
print('%6s%3s%10s%5s' % ('Actual', '0', str(cm[0][0]), str(cm[0][1])))
print('%6s%3s%10s%5s' % ('Class', '1', str(cm[1][0]), str(cm[1][1])))

# EXTRA CREDIT CLASSIFIER.
print('\n\nExtra Credit Question: Find 50 features that maximize classifier performance.')
X_train, X_test, y_train, y_test = train_test_split(dataframe_50.drop('ICU', axis=1), dataframe_50['ICU'])
classifier = LogisticRegression(max_iter=1000) # increase max_iter for logistic regression, so it converges.
classifier.fit(X_train, y_train)

print('\nTraining and test data split sizes:')
print('Size of X_train: %d' % (len(X_train)))
print('Size of X_test: %d' % (len(X_test)))

print('\nAnalysis of classifier performance:')
prediction = classifier.predict(X_test)
print('Accuracy of classifier:  %.2f' % (classifier.score(X_test, y_test)))
print('Precision of classifier: %.2f' % (precision_score(prediction, y_test)))
print('Recall of classifier:    %.2f' % (recall_score(prediction, y_test)))

# Print Confusion Matrix:
cm = confusion_matrix(prediction, y_test)
print('\n%24s' % ('Predicted Class'))
print('%19s%5s' % ('0', '1'))
print('%6s%3s%10s%5s' % ('Actual', '0', str(cm[0][0]), str(cm[0][1])))
print('%6s%3s%10s%5s' % ('Class', '1', str(cm[1][0]), str(cm[1][1])))