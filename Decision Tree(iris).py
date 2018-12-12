# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 16:56:55 2018

@author: ssn
"""

#decision Tree
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
iris=load_iris()
test_idx=[0,50,100]
#train
train_target=np.delete(iris.target, test_idx)
train_data =np.delete(iris.data, test_idx, axis=0)
#testk
test_target= iris.target[test_idx]
test_data= iris.data[test_idx]
#fit
clf= tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)
res = np.array([5.1,3.5,1.4,0.2])
res = res.reshape(1,4)
a= clf.predict(res)
print (a)
a= clf.predict(test_data)
print (a)