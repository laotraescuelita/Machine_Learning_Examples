import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


df = pd.read_csv('010_insurance.csv')
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())

sns.set()
plt.figure(figsize=(6,6))
sns.distplot(df['age'])
plt.title('Age distribution')
plt.show()

plt.figure(figsize=(6,6))
sns.countplot(x='sex', data=df)
plt.title('Sex distribution.')
plt.show()

plt.figure(figsize=(6,6))
sns.distplot(df['bmi'])
plt.title('Bmi distribution')
plt.show()

plt.figure(figsize=(6,6))
sns.countplot(x='children', data=df)
plt.title('Children.')
plt.show()
print(df['children'].value_counts())

plt.figure(figsize=(6,6))
sns.countplot(x='smoker', data=df)
plt.title('Smoker.')
plt.show()

plt.figure(figsize=(6,6))
sns.countplot(x='region', data=df)
plt.title('Region.')
plt.show()

plt.figure(figsize=(6,6))
sns.distplot(df['charges'])
plt.title('Charges')
plt.show()

df.replace({'sex':{'male':0, 'female':1}}, inplace=True)
df.replace({'smoker':{'yes':0, 'no':1}}, inplace=True)
df.replace({'region':{'southeast':0, 'southwest':1, 'northeast':2, 'northwest':3}}, inplace=True)

X = df.drop(columns='charges', axis=1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


model = LinearRegression()
model.fit(X_train, y_train)

y_train_hat = model.predict(X_train)
y_hat_train_accuracy_1 = metrics.r2_score(y_train, y_train_hat)
y_hat_train_accuracy_2 = metrics.mean_absolute_error(y_train, y_train_hat)
print('R_square error of y_train_hat data: ', y_hat_train_accuracy_1)
print('Mean absolute error of y_train_hat data: ', y_hat_train_accuracy_2)
#Visualizar precios contra la predicción
#plt.scatter(y_train, y_train_hat)
y_train = list(y_train)
plt.plot(y_train, color='blue', label='Actual value')
plt.plot(y_train_hat, color='green', label='Predicted value')
plt.title('Actual prices vs Predicted prices')
plt.show()

y_test_hat = model.predict(X_test)
y_hat_test_accuracy_1 = metrics.r2_score(y_test, y_test_hat)
y_hat_test_accuracy_2 = metrics.mean_absolute_error(y_test, y_test_hat)
print('R_square error of y_test_hat data: ', y_hat_test_accuracy_1)
print('Mean absolute error of y_test_hat data: ', y_hat_test_accuracy_2)
#Visualizar precios contra la predicción
#plt.scatter(y_train, y_train_hat)
y_test = list(y_test)
plt.plot(y_test, color='blue', label='Actual value')
plt.plot(y_test_hat, color='green', label='Predicted value')

plt.title('Actual prices vs Predicted prices')
plt.show()

