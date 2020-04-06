#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree


# In[3]:


dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')


# In[4]:


data.head()


# In[5]:


y= data.quality
x= data.drop('quality', axis=1)


# In[6]:


data.head()


# In[8]:


x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2)


# In[9]:


x_train.head()


# In[12]:


x_train_scaled= preprocessing.scale(x_train)
x_train_scaled


# In[14]:


classifier=tree.DecisionTreeClassifier()
classifier.fit(x_train,y_train)


# In[16]:


confidence=classifier.score(x_test,y_test)
confidence


# In[19]:


y_pred = classifier.predict(x_test)


# In[26]:


X=np.array(y_pred).tolist()

#printing first 5 predictions
print(("\nThe prediction:\n"))
for i in range(0,5):
   print(X[i]) 
    
#printing first five expectations
print("\nThe expectation:\n")
print (y_test.head())


# In[ ]:




