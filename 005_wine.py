#Importar las librerias. 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Importar los datos 
df = pd.read_csv('005_wine.csv')
print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df.describe())

#Visualizar algunos vectores
sns.catplot(x='quality', data=df, kind='count')
plt.show()

plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='volatile acidity', data=df)
plt.show()

plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='citric acid', data=df)
plt.show()

correlation = df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
plt.show()


#preparar los datos 
X = df.drop('quality', axis=1)
#Hacer solo dos variables de todas las categorias que ya existen
y = df['quality'].apply(lambda muestra: 1 if muestra >=7 else 0 )
print(X.shape, y.shape)

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#Seleccionar el algorimto para el caso particular. Regresion. Clasificacion. Agrupamiento. 
model = RandomForestClassifier()
model.fit(X_train, y_train)

y_train_hat = model.predict(X_train)
y_hat_train_accuracy = accuracy_score(y_train_hat, y_train)
print('Accuracy score of y_train_hat data: ', y_hat_train_accuracy)

y_test_hat = model.predict(X_test)
y_hat_test_accuracy = accuracy_score(y_test_hat, y_test)
print('Accuracy score of y_test_hat data: ', y_hat_test_accuracy)

#Hacer predicciones
input_data = (7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4)
result = 1
input_data_asarray = np.asarray(input_data)
input_data_reshape = input_data_asarray.reshape(1,-1)
y_hat = model.predict(input_data_reshape)
if y_hat[0] == 0:
	print('Bad quality')
else:
	print('Good quality')

print(y_hat)

