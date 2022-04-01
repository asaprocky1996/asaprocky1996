#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# cd downloads

# In[10]:


gym = pd.read_excel(r'C:\Users\dines\Documents\BEPEC\dataGYM.xlsx')
gym.head(10)


# In[14]:


gym['Class'] = LabelEncoder().fit_transform(gym['Class'])


# In[15]:


gym.head(10)


# In[16]:


gym['Class'].value_counts()


# In[35]:


X = gym.iloc[:,:3]
X.head()


# In[36]:


y = gym.iloc[:,5:]
y.head()


# In[41]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2)


# In[42]:


model_gym = RandomForestClassifier(n_estimators=20)

model_gym.fit(X_train , y_train)


# In[43]:


expected = y_test
predicted = model_gym.predict(X_test)


# In[44]:


print(metrics.classification_report(expected , predicted))
print(metrics.confusion_matrix(expected , predicted))


# In[46]:


import pickle
pickle.dump(model_gym ,open('model_gym.pkl','wb'))
model=pickle.load(open('model_gym.pkl','rb'))
print(model.predict([[53,4.10,91]]))


# In[ ]:




