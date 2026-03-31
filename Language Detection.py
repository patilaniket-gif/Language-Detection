#!/usr/bin/env python
# coding: utf-8

# In[10]:


#import all Libraries
import numpy as np
import pandas as pd

import sklearn
from sklearn.feature_extraction.text import CountVectorizer #text to number
#model impliment
from sklearn.model_selection import train_test_split
#algorithm Naive bayes
from sklearn.naive_bayes import MultinomialNB #word count


# In[2]:


data = pd.read_csv("language.csv")


# In[3]:


data


# In[11]:


#check null data
data.isnull().sum()


# In[14]:


data["language"].value_counts()


# In[16]:


data["Text"].value_counts()


# In[19]:




data.dtypes


# In[21]:


#creating array

x = np.array(data["Text"])
y = np.array(data["language"])


# In[26]:


x


# In[27]:


y


# In[28]:


#array to number for x(text)

cv = CountVectorizer()
x = cv.fit_transform(x)


# In[40]:


#Training and testing data

x_train,x_test , y_train,y_test = train_test_split(x,y, test_size=0.33, random_state=42)


# In[45]:


#create model

model = MultinomialNB()
model.fit(x_train,y_train)


# In[46]:


model.score(x_test,y_test)


# In[57]:


#output 

user = input("Enter the Text :")
data = cv.transform([user]).toarray() #for more sentence to for language
output = model.predict(data)
print(output)


# In[ ]:




