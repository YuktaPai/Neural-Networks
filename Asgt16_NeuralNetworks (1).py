#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install keras')
get_ipython().system('pip install tensorflow')
from keras.models import Sequential
from keras.layers import Dense


# ### Data

# In[4]:


import pandas as pd
import numpy as np
df=pd.read_csv('C:/Users/17pol/Downloads/forestfires.csv')
df


# ### EDA ans Preprocessing

# In[5]:


df.head()


# In[6]:


df.sample(10)


# In[7]:


df.info()


# In[8]:


df.shape


# In[9]:


df.isna().sum()


# In[10]:


df.describe()


# In[11]:


#scaling the numerical data( leaving the target variable )
df1=df.iloc[:,2:30]
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df_norm=scaler.fit_transform(df1)
df_norm


# ### PCA

# In[12]:


from sklearn.decomposition import PCA
pca=PCA(n_components=28)
pca_values=pca.fit_transform(df_norm)
pca_values


# In[13]:


var=pca.explained_variance_ratio_
var


# In[14]:


var1=np.cumsum(np.round(var,decimals=4)*100)
var1


# In[15]:


import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
plt.plot(var1,color='red')


# In[16]:


#hence here we will choose 24 pcs outoff 28 for further procedure
finaldf=pd.concat([pd.DataFrame(pca_values[:,0:24],columns=['pc1','pc2','pc3','pc4','pc5','pc6','pc7',
                                                             'pc8','pc9','pc10','pc11','pc12','pc13','pc14',
                                                             'pc15','pc16','pc17','pc18','pc19','pc20','pc21',
                                                             'pc22','pc23','pc24']),
                 df[['size_category']]], axis = 1)
finaldf.size_category.replace(('large','small'),(1,0),inplace=True)
finaldf


# #### #split the data into x and y

# In[17]:



array=finaldf.values
x=array[:,0:24]
y=array[:,24]


# In[18]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD


# In[19]:


model=Sequential()
model.add(Dense(12,input_dim=24,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x,y, validation_split=0.3,epochs=150,batch_size=10)


# In[20]:


#accuracy of model
scores=model.evaluate(x,y)


# In[21]:


print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[22]:


model1=Sequential()
model1.add(Dense(12,input_dim=24,activation='sigmoid'))
model1.add(Dense(8,activation='sigmoid'))
model1.add(Dense(1,activation='relu'))
model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model1.fit(x, y, validation_split=0.3, epochs=100, batch_size=15)


# In[23]:


#model accuracy
scores1=model1.evaluate(x,y)
print("%s: %.2f%%" % (model1.metrics_names[1], scores1[1]*100))


# In[24]:


model2=Sequential()
model2.add(Dense(12,input_dim=24,activation='relu'))
model2.add(Dense(8,activation='relu'))
model2.add(Dense(1,activation='relu'))
model2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model2.fit(x,y,epochs=100, validation_split=0.3,batch_size=15)


# In[25]:


#model accuracy
scores2=model2.evaluate(x,y)
print("%s: %.2f%%" % (model2.metrics_names[1], scores2[1]*100))


# In[26]:


model3=Sequential()
model3.add(Dense(12,input_dim=24,activation='relu'))
model3.add(Dense(8,activation='relu'))
model3.add(Dense(1,activation='relu'))
model3.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model3.fit(x,y,epochs=150, validation_split=0.3,batch_size=10)


# In[27]:


scores3 = model3.evaluate(x, y)
print("%s: %.2f%%" % (model3.metrics_names[1], scores3[1]*100))


# ### Conclusively, we can analyse that the best of all iteration is first one where accuracy of the system came as 92.65%

# In[ ]:




