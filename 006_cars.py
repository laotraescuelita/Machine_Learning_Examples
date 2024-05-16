#Importar las librerias. 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

#Importar los datos 
df = pd.read_csv('006_cars.csv')
print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())


#preparar los datos 
print(df.Fuel_Type.value_counts())
print(df.Seller_Type.value_counts())
print(df.Transmission.value_counts())

df.replace({'Fuel_Type':{'Petrol':0, 'Diesel':1, 'CNG':2 }}, inplace=True)
df.replace({'Seller_Type':{'Dealer':0, 'Individual':1 }}, inplace=True)
df.replace({'Transmission':{'Manual':0, 'Automatic':1 }}, inplace=True)
print(df.head())


#Train test split
X = df.drop(['Car_Name', 'Selling_Price'], axis=1)
y = df['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


#Seleccionar el algorimto para el caso particular. Regresion. Clasificacion. Agrupamiento. 
model = LinearRegression()
model.fit(X_train, y_train)

y_train_hat = model.predict(X_train)
y_hat_train_accuracy_1 = metrics.r2_score(y_train, y_train_hat)
y_hat_train_accuracy_2 = metrics.mean_absolute_error(y_train, y_train_hat)
print('R_square error of y_train_hat data: ', y_hat_train_accuracy_1)
print('Mean absolute error of y_train_hat data: ', y_hat_train_accuracy_2)
#Visualizar precios contra la predicción
plt.scatter(y_train, y_train_hat)
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
plt.scatter(y_test, y_test_hat)
plt.xlabel('Actual prices')
plt.ylabel('Predicted prices')
plt.title('Actual prices vs Predicted prices')
plt.show()


#Seleccionar el algorimto para el caso particular. Regresion. Clasificacion. Agrupamiento. 
model = Lasso()
model.fit(X_train, y_train)

y_train_hat = model.predict(X_train)
y_hat_train_accuracy_1 = metrics.r2_score(y_train, y_train_hat)
y_hat_train_accuracy_2 = metrics.mean_absolute_error(y_train, y_train_hat)
print('R_square error of y_train_hat data: ', y_hat_train_accuracy_1)
print('Mean absolute error of y_train_hat data: ', y_hat_train_accuracy_2)
#Visualizar precios contra la predicción
plt.scatter(y_train, y_train_hat)
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
plt.scatter(y_test, y_test_hat)
plt.xlabel('Actual prices')
plt.ylabel('Predicted prices')
plt.title('Actual prices vs Predicted prices')
plt.show()
