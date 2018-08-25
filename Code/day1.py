
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt


dataset = pd.read_csv('./../datasets/Data.csv')
X = dataset.iloc[ : , : ].values # get all values [rows,columns]
Y = dataset.iloc[ : , 3].values



#filling  empty values with the mean of the same field
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
# imputer = imputer.fit(X[ : , 1:3])   #returns self
X[ : , 1:3] = imputer.fit_transform(X[ : , 1:3])  # return numpy array (fits and transforms)
print()
print(X)

# Encoding categorical data
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0 ])
X[ : , 3] = labelencoder_X.fit_transform(X[ : , 3 ])


onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)

# Splitting the datasets into training sets and Test sets
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)

print("\nwahid\n")



sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

dataset = pd.read_csv('../datasets/studentscores.csv')
X = dataset.iloc[ : ,   : 1 ].values
Y = dataset.iloc[ : , 1 ].values

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/4, random_state = 0) 

print("XXXXXXXXXXXX")
print(X_train)
print("yyyyyyyyyyyy")
print(Y_train)

regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

plt.scatter(X_train , Y_train, color = 'red')
plt.plot(X_train , regressor.predict(X_train), color ='blue')
plt.show()


plt.scatter(X_test , Y_test, color = 'red') 
plt.plot(X_test , regressor.predict(X_test), color ='blue')
plt.show()