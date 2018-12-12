# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 17:05:33 2018

@author: ssn
"""

#multiple regression
import pandas as pd
import numpy as np
dataset=pd.read_csv("mul_reg.csv")
x=dataset.iloc[:,:4].values
y=dataset.iloc[:,4].values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
x[:,3] = labelencoder.fit_transform(x[:,3])
onehotencoder= OneHotEncoder(categorical_features=[3])
x = onehotencoder.fit_transform(x).toarray()
x=x[:,1:]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
linearreg=LinearRegression()
linearreg.fit(x_train,y_train)
y_pred=linearreg.predict(x_test)
import statsmodels.api as sf
OLS= sf.OLS(y,x).fit()
OLS.summary()
x=np.append(np.ones((50,1)).astype(int),x,axis=1)
x_b=x[:,[0,1,2,3,4,5]]
OLS= sf.OLS(y,x_b).fit()
OLS.summary()
x_b=x[:,[0,1,3,4,5]]
OLS= sf.OLS(y,x_b).fit()
OLS.summary()
x_b=x[:,[0,3,4,5]]
OLS= sf.OLS(y,x_b).fit()
OLS.summary()
x_b=x[:,[0,3,5]]
OLS= sf.OLS(y,x_b).fit()
OLS.summary()
x_b=x[:,[0,3]]
OLS= sf.OLS(y,x_b).fit()
OLS.summary()