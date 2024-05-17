import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('008_heart_disease.csv')
print(df.head())
print(df.shape)
print(df.describe())
print(df.info())
print(df.isnull().sum())

print(df['target'].value_counts())

X = df.drop('target', axis=1)
y = df['target']
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

model = LogisticRegression()
model.fit(X_train, y_train)

y_train_hat = model.predict(X_train)
y_hat_train_accuracy = accuracy_score(y_train_hat, y_train)
print('Accuracy score of y_train_hat data: ', y_hat_train_accuracy)

y_test_hat = model.predict(X_test)
y_hat_test_accuracy = accuracy_score(y_test_hat, y_test)
print('Accuracy score of y_test_hat data: ', y_hat_test_accuracy)

#Hacer predicciones
input_data = (63,1,3,145,233,1,0,150,0,2.3,0,0,1)
result = 1
input_data_asarray = np.asarray(input_data)
input_data_reshape = input_data_asarray.reshape(1,-1)
y_hat = model.predict(input_data_reshape)
if y_hat[0] == 1:
	print('Heart disease.')
else:
	print('No heart disease.')
print(y_hat)

