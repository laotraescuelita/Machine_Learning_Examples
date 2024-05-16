#Importar las librerias. 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.datasets import fetch_california_housing

#Importar los datos     
data = fetch_california_housing()
print(data.keys()) #['data', 'target', 'frame', 'target_names', 'feature_names', 'DESCR']
print(data.target_names)
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
print( df.head )
print(df.info() )
print(df.shape)
print(df.isnull().sum())
print(df.describe())

#preparar los datos 
correlation = df.corr() #Comparando los vectores para decidir si hay una corelacion positiva, negativa, ninguna.
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, annot=True, annot_kws={'size':8}, cmap='Blues')
plt.show()

#Train test split
X = df.drop(['target'], axis=1)
y = df['target']
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#Seleccionar el algorimto para el caso particular. Regresion. Clasificacion. Agrupamiento. 

model = XGBRegressor()
model.fit(X_train, y_train)

y_train_hat = model.predict(X_train)
y_hat_train_accuracy_1 = metrics.r2_score(y_train, y_train_hat)
y_hat_train_accuracy_2 = metrics.mean_absolute_error(y_train, y_train_hat)
print('R_square error of y_train_hat data: ', y_hat_train_accuracy_1)
print('Mean absolute error of y_train_hat data: ', y_hat_train_accuracy_2)


y_test_hat = model.predict(X_test)
y_hat_test_accuracy_1 = metrics.r2_score(y_test, y_test_hat)
y_hat_test_accuracy_2 = metrics.mean_absolute_error(y_test, y_test_hat)
print('R_square error of y_test_hat data: ', y_hat_test_accuracy_1)
print('Mean absolute error of y_test_hat data: ', y_hat_test_accuracy_2)

#Visualizar precios contra la predicción
plt.scatter(y_train, y_train_hat)
plt.xlabel('Actual prices')
plt.ylabel('Predicted prices')
plt.title('Actual prices vs Predicted prices')
plt.show()
