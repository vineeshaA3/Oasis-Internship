#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# In[39]:


data=pd.read_csv("Iris.csv")


# In[43]:


print(data)


# In[44]:


data.head(20)


# In[45]:


data.tail(20)


# In[46]:


data.info()


# In[47]:


data.describe()


# In[48]:


data.shape


# In[49]:


data.size


# In[50]:


data.columns


# In[51]:


data.isnull().sum()


# # Data Cleaning

# In[52]:


data = data.drop('Id',axis = 1)
data


# # Data Visualization

# In[53]:


sb.pairplot(data, hue = 'Species', palette = 'hls')


# In[59]:


import pandas as pd

data = pd.read_csv('Iris.csv')
print(df.corr())


# In[60]:


pt.figure(figsize=(8 ,4))
sb.boxplot(data=data, orient="h", palette="Set2")
pt.show()


# In[61]:


sb.lmplot( x="SepalLengthCm", y="SepalWidthCm", data=data, fit_reg=False, hue='Species', legend=False)
pt.legend(loc='lower right')
pt.show()


# # Heatmap

# In[66]:


pt.figure(figsize=(8 ,4))
sb.heatmap(df.corr(), annot=True, cmap='viridis')
pt.show()


# # Model Training

# In[72]:


x_axis = df.drop('Species', axis=1)
x_axis.head()


# In[74]:


y_axis = data['Species']
y_axis.head()


# In[75]:


x_train, x_test, y_train, y_test = train_test_split(x_axis, y_axis, test_size= 0.2)


# # Using SVM

# In[76]:


data_model=SVC()


# In[77]:


data_model.fit(x_train,y_train)


# In[78]:


prediction = data_model.predict(x_test)


# In[79]:


prediction


# In[80]:


print(accuracy_score(y_test,prediction)*100)


# # Using Logistic Regression

# In[81]:


lreg_model = LogisticRegression()
lreg_model.fit(x_train, y_train)


# In[82]:


predict_1 = lreg_model.predict(x_test)


# In[83]:


print(accuracy_score(y_test,predict_1)*100)


# In[ ]:




