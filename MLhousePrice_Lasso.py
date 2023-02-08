import os
os.system("clear")
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


df=pd.read_csv("/Users/amir/Desktop/train.csv")
# print(df.columns)
my_df=df[['LotArea', 'GarageArea','SalePrice']]

df = df.dropna()
#to drop rows wchich contain NaN
 
# To reset the indices
df = df.reset_index(drop = True)
# Print the dataframe

x= df.iloc[:,:-1]
y =df.iloc[:,-1:]
print(x.head)
print(y.head)


x_train , x_test, y_train, y_test =train_test_split( x, y, test_size=0.2 , random_state=11)
# print(x_test)
# print(y_test)
m = 0.4
lasso = Lasso(alpha=m)
lasso.fit(x_train, y_train)
print()
print(f"for Alpha(Penalty hyper parameter of {m})Score in lasso regression for test set is: {lasso.score(x_test,y_test)}, and for train set is: {lasso.score(x_train, y_train)}")
# print(r2_score(x_train, y_train))
# print(r2_score(y_test , x_test))
print()
print(lasso.coef_)