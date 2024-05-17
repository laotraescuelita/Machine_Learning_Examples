import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('009_creditcard.csv')
print(df.head())
print(df.tail())
print(df.shape)
print(df.describe())
print(df.info())
print(df.isnull().sum())

print(df['Class'].value_counts())

#Separar para analisar la matriz por su clase. 
legit = df[df.Class == 0]
fraud = df[df.Class == 1]
print(legit.shape, fraud.shape)
print(legit.Amount.describe())
print(fraud.Amount.describe())
print( df.groupby('Class').mean())

legit = legit.sample(n=492)
df = pd.concat([legit, fraud], axis=0)
print(df.head())
print(df.shape)
print(df['Class'].value_counts())

X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=1)
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
input_data = (0,-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215,149.62)
result = 0
input_data_asarray = np.asarray(input_data)
input_data_reshape = input_data_asarray.reshape(1,-1)
y_hat = model.predict(input_data_reshape)
if y_hat[0] == 0:
	print('No fraud.')
else:
	print('Fraud.')
print(y_hat)






