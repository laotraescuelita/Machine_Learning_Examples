#Importar las librerias. 
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


#Importar los datos 
df = pd.read_csv('004_loans.csv')
print(df.head())
print(df.shape)
print(df.describe())
print(df.isnull().sum())

#preparar los datos 
df = df.dropna()
print(df.isnull().sum())
print(df.replace({'Loan_Status':{'N':0,'Y':1}}, inplace=True) )
print(df.head())
print(df['Dependents'].value_counts())
df = df.replace(to_replace='3+', value=4)
print(df['Dependents'].value_counts())

#Visualizar algunos de los vectores
sns.countplot(x='Education', hue='Loan_Status', data=df)
plt.show()

sns.countplot(x='Married', hue='Loan_Status', data=df)
plt.show()

df.replace( {'Married':{'No':0, 'Yes':1}, 'Gender':{'Male':1,'Female':0},
	'Self_Employed':{'No':0, 'Yes':1}, 'Property_Area':{'Rural':0, 'Semiurban':1,'Urban':2},
	'Education':{'Graduate':1, 'Not Graduate':0}}, inplace=True )


#Train test split
X = df.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status']
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=1)
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


"""
#Hacer predicciones
input_data = df[0]
print(input_data)

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
"""