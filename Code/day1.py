
import numpy as np
import pandas as pd

# print("printing...")

dataset = pd.read_csv('./../datasets/Data.csv')
X = dataset.iloc[ : , : ].values
Y = dataset.iloc[ : , 3].values

print(X[0][0])

# print(Y)
# print(dataset.iloc)

# print(type(dataset))