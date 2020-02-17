import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('servo.data')
data.columns = ['motor', 'screw', 'pgain', 'vgain', 'class']
num = LabelEncoder()
data['motor'] = num.fit_transform(data['motor'].astype('str'))
data['screw'] = num.fit_transform(data['screw'].astype('str'))
X = data[['motor','screw', 'pgain', 'vgain']]
Y = data['class']
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=42)

#model = LinearRegression()
#model = RandomForestRegressor()
model = DecisionTreeRegressor()
model.fit(X_train, Y_train)
acc = model.score(X_validation, Y_validation)
print(acc*100)
