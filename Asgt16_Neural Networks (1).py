#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraries
import pandas as pd
import numpy as np
#Plot Tools
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
#Model Building
from sklearn.preprocessing import StandardScaler
import sklearn
import keras
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import InputLayer,Dense
import tensorflow as tf
#Model Validation
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error


# ### Data

# In[2]:


data=pd.read_csv('C:/Users/17pol/Downloads/gas_turbines.csv')
data


# ### EDA and Preprocessing

# In[3]:


data.info()


# In[5]:


data.shape


# In[6]:


data.isna().sum()


# In[7]:


data.sample(10)


# In[8]:


data.describe()


# In[9]:


X = data.loc[:,['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'CDP', 'CO','NOX']]
y= data.loc[:,['TEY']]


# In[10]:


scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)


# In[11]:


def baseline_model():
    model = Sequential()
    model.add(Dense(10, input_dim=10, activation='tanh'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# In[12]:


estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=50, batch_size=100, verbose=False)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# In[13]:


estimator.fit(X, y)
prediction = estimator.predict(X)
prediction


# In[14]:


a=scaler.inverse_transform(prediction)
a


# In[15]:


b=scaler.inverse_transform(y)
b


# In[16]:


mean_squared_error(b,a)


# In[17]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


# In[18]:


estimator.fit(X_train, y_train)
prediction = estimator.predict(X_test)
prediction


# In[19]:


c=scaler.inverse_transform(prediction)
d=scaler.inverse_transform(y_test)
mean_squared_error(d,c)

