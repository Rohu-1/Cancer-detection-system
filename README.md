# Cancer-detection-system
# import all the dependencies
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# data collection and processing

# loading the data from sklearn

breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
# print(breast_cancer_dataset)

# loading the data to a data frame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)

# print first 5 rows of dataset
print(data_frame.head())

# adding the target column to the data frame
data_frame['label'] = breast_cancer_dataset.target

# print last 5 rows of dataset
print(data_frame.tail())

# no of rows and columns in the dataset
print(data_frame.shape)

# getting information about the data

print(data_frame.info())

# checking for missing values
print(data_frame.isnull().sum())

# statistical measures about the data
print(data_frame.describe())

# checking the distribution of target values
print(data_frame['label'].value_counts())
# 1-->represents benign
# 0-->represents malignant

print(data_frame.groupby('label').mean())

# separating the features and target
x = data_frame.drop(columns='label', axis=1)
y = data_frame['label']
print(y)

# splitting the data into training data and testing data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
print(x.shape, x_train.shape, x_test.shape)

# Model training
# LogisticRegression

model = LogisticRegression()
# training the logistic regression model using training data

model.fit(x_train, y_train)

# Model evaluation
# Accuracy score

# accuracy on training data

x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print("Accuracy on the training data is", training_data_accuracy)

# accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test, x_test_prediction)
print("Accuracy on the test data is", test_data_accuracy)

# Building a predictive system

input_data = (
    12.4, 15.7, 82.57, 477.1, 0.1278, 0.17, 0.1578, 0.08089, 0.2087, 0.07613, 0.3345, 0.8902, 2.217, 27.19, 0.00751,
    0.03345, 0.03672, 0.01137, 0.02165, 0.005082, 15.47, 23.75, 103.4, 741.6, 0.1791, 0.5249, 0.5355, 0.1741, 0.3985,
    0.1244)

# change the input data into numpy array

input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy arrayas we are predicting for one datapoint

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print("the breast cancer is malignent")
else:
    print("the breast cancer is bening")
