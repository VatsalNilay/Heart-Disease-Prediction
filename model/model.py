from copyreg import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle


#loading the data
heart_data = pd.read_csv('model/Heart_Disease_Prediction.csv')

#separating the result from data to train our model
train_model = heart_data.drop(columns='Heart Disease', axis=1)
results = heart_data['Heart Disease']

#We have decided to train our model on 80% of the data and rest 20% for validating it.
train_model_train, train_model_test, result_train, result_test = train_test_split(train_model, results, test_size=0.2, stratify=results, random_state=2)


model_l = LogisticRegression()
model_l.fit(train_model_train.values, result_train.values)

#Accuracy on training data
model_l_predictions = model_l.predict(train_model_train.values)
model_l_accuracy = accuracy_score(model_l_predictions, result_train.values)

#Accuracy on test data
model_l_prediction = model_l.predict(train_model_test.values)
model_l_accuracy = accuracy_score(model_l_prediction,result_test.values)

data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

# change the input data to a numpy array
data_array = np.asarray(data)

# reshape the numpy array as we are predicting for only on instance
reshaped_data = data_array.reshape(1,-1)

prediction = model_l.predict(reshaped_data)
print(*prediction)

filename = 'finalized_model.sav'
pickle.dump(model_l, open(filename, 'wb'))

