
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


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

print()
print(X)