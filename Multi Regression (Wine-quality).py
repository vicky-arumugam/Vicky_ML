# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 16:59:37 2018

@author: ssn
"""

#Multiple-regression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
dataset=pd.read_csv('winequality-red.csv')
x=dataset.iloc[:,0:11].values
y=dataset.iloc[:,[11]].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train,y_train)
regressor.coef_
regressor.intercept_
y_pred=regressor.predict(x_test)
if y_pred>=7:
    print("good")
else:
    print("bad")    
pd
b = np.array(np.round(y_pred))
p = [b]
pd.DataFrame(p).transpose().to_csv("my_solution.csv", index = 0)