#Importar las librerias. 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn import svm 
from sklearn.metrics import accuracy_score

#Importar los datos 
df = pd.read_csv('002_diabetes.csv')
print(df.head())
print(df.shape)
print(df.describe())
print(df['Outcome'].value_counts())
print(df.groupby('Outcome').mean())

#preparar los datos 
X = df.drop(columns='Outcome', axis=1)
y = df['Outcome']
#Estandarizar la matriz.
scaler = StandardScaler()
scaler.fit(X)
data_standardized = scaler.transform(X)
X = data_standardized
y = df['Outcome']

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, stratify=y, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#Seleccionar el algorimto para el caso particular. Regresion. Clasificacion. Agrupamiento. 
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

y_train_hat = model.predict(X_train)
y_hat_train_accuracy = accuracy_score(y_train_hat, y_train)
print('Accuracy score of y_train_hat data: ', y_hat_train_accuracy)

y_test_hat = model.predict(X_test)
y_hat_test_accuracy = accuracy_score(y_test_hat, y_test)
print('Accuracy score of y_test_hat data: ', y_hat_test_accuracy)

#Hacer predicciones
input_data = (6,148,72,35,0,33.6,0.627,50)
result = 1
input_data_asarray = np.asarray(input_data)
input_data_reshape = input_data_asarray.reshape(1,-1)
standardize_data = scaler.transform(input_data_reshape)
y_hat = model.predict(standardize_data)
if y_hat[0] == 0:
	print('The person is not diabetic')
else:
	print('The person is diabetic')

print(y_hat)

