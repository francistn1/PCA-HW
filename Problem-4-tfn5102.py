import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler

x = np.array([[1, 2], [-1, 1], [0, 1], [2, 4], [3, 1]])

df = pd.DataFrame(x, columns=['A', 'B'])

# Centering
print("Mean standardized data: ", df.mean(axis=0))
print("Standard Deviation standardized data:", df.std(axis=0))

data_standardized = preprocessing.scale(df)

print("Mean standardized data: ", data_standardized.mean(axis=0))
print("Standard Deviation standardized data:", data_standardized.std(axis=0))

# Scaling
norm = MinMaxScaler().fit(df)
df_norm = norm.transform(df)

# Standardization
scale = StandardScaler().fit(df)
df_stand = scale.transform(df)

# Normalization
x_norm = preprocessing.scale(df)
