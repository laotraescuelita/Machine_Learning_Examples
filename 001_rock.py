#Importar las librerias. 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 


#Importar los datos y mirar información de las columnas.
df = pd.read_csv("001_rock.csv", header=None)
print(df.head())
print(df.shape)
print(df.describe())
print(df[60].value_counts())
print(df.groupby(60).mean())


#preparar los datos 
X = df.drop(columns=60, axis=1)
y = df[60]


#Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


#Seleccionar el algorimto para el caso particular. Regresion. Clasificacion. Agrupamiento. 
model = LogisticRegression()
model.fit(X_train, y_train)

X_train_prediction = model.predict(X_train)
data_train_accuracy = accuracy_score(X_train_prediction, y_train)
print('Accuracy on training data:', data_train_accuracy)

X_test_prediction = model.predict(X_test)
data_test_accuracy = accuracy_score(X_test_prediction, y_test)
print('Accuracy on testing data:', data_test_accuracy)

#Test con una muestra de la misma matriz. 
input_data = (0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032)
#Resultado es R 
input_data_asarray = np.asarray(input_data)
input_data_reshape = input_data_asarray.reshape(1,-1)
prediction = model.predict(input_data_reshape)

if prediction[0]=='R':	
	print('La muestra es una roca')
else:
	print('Es una mina')


