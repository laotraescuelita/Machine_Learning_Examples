#Importar las librerias. 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn import metrics 


#Importar los datos 
df = pd.read_csv('007_gold.csv')
df = df.drop('Date', axis=1)
print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df.info())
print(df.describe())


#preparar los datos 
correlation = df.corr()

plt.figure(figsize=(8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8})
plt.show()

print(correlation['GLD'])

sns.distplot(df['GLD'], color='green')
plt.show()


#Train test split
X = df.drop('GLD', axis=1)
y = df['GLD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


#Seleccionar el algorimto para el caso particular. Regresion. Clasificacion. Agrupamiento. 
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)


y_train_hat = model.predict(X_train)
y_hat_train_accuracy_1 = metrics.r2_score(y_train, y_train_hat)
y_hat_train_accuracy_2 = metrics.mean_absolute_error(y_train, y_train_hat)
print('R_square error of y_train_hat data: ', y_hat_train_accuracy_1)
print('Mean absolute error of y_train_hat data: ', y_hat_train_accuracy_2)
#Visualizar precios contra la predicción
y_train = list(y_train)
plt.plot(y_train, color='blue', label='Actual value')
plt.plot(y_train_hat, color='green', label='Predicted value')
plt.xlabel('Actual prices')
plt.ylabel('Predicted prices')
plt.title('Actual prices vs Predicted prices')
plt.show()

y_test_hat = model.predict(X_test)
y_hat_test_accuracy_1 = metrics.r2_score(y_test, y_test_hat)
y_hat_test_accuracy_2 = metrics.mean_absolute_error(y_test, y_test_hat)
print('R_square error of y_test_hat data: ', y_hat_test_accuracy_1)
print('Mean absolute error of y_test_hat data: ', y_hat_test_accuracy_2)
#Visualizar precios contra la predicción
y_test = list(y_test)
plt.plot(y_test, color='blue', label='Actual value')
plt.plot(y_test_hat, color='green', label='Predicted value')
plt.xlabel('Actual prices')
plt.ylabel('Predicted prices')
plt.title('Actual prices vs Predicted prices')
plt.show()
