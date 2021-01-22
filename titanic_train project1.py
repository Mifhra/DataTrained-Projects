#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[14]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 


# In[4]:


df=pd.read_csv("E:\dowenlode/train.csv")


# In[5]:


df


# In[6]:


df.columns


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# there are 177 NaN in Age column,2 in Embarked
# and 687 Nan values in Cabin column 
# 

# In[10]:


freq_df=df["Survived"].value_counts()


# In[11]:


freq_df


# In[16]:


graph=sns.countplot(x="Survived",data=df)


# In[17]:


sns.heatmap(df.isnull())


# In[18]:


sns.countplot(x="Survived",hue="Sex",data=df)


# so more than 400 males were not able to survive while less than 100 female did not survived of total died people
# ,
# on the other side less than 100 male survived and more than 200 female survived of total survived people.

# In[20]:


#replacing the missing values in age column with mean of age column 
df["Age"].fillna(df["Age"].mean(),inplace=True)


# In[21]:


df


# In[23]:


df.isnull().sum()


# In[25]:


df.drop("Cabin",axis=1,inplace=True)


# In[26]:


df.isnull().sum()


# In[27]:


df.drop("Embarked",axis=1,inplace=True)


# In[30]:


df["Age"].hist(bins=4)


# we can observe that there most of the people on ship are in age group of 20 to 40.

# In[31]:


le=LabelEncoder()
df["Sex"]=le.fit_tranform(df["Age"])
    


# In[38]:


from sklearn.preprocessing import LabelEncoder 


# In[43]:


le=LabelEncoder()
df_new=le.fit_transform(df["Sex"])
    


# In[34]:


df["Sex"]


# In[35]:


df.head(20)


# In[44]:


le.classes_


# In[45]:


df_new


# In[47]:


data=df.drop("Sex",axis=1)


# In[48]:


data.head(2)


# In[49]:


data["Sex"]=df_new


# In[52]:


data["Sex"].head(2)


# In[53]:


data.head(2)


# In[54]:


data.head(20)


# In[66]:


data.head(2)


# sec coloumn converted to o,1 ie. numerical values

# In[56]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")



# In[67]:


data.drop("Ticket",axis=1,inplace=True)


# In[68]:


data.head(2)


# In[69]:


x=data.drop("Survived",axis=1)


# In[70]:


x


# In[71]:


y=data["Survived"]


# In[72]:


y


# In[73]:


train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=.30,random_state=45)


# In[74]:


train_x.shape


# In[75]:


test_x.shape


# In[76]:


train_y.shape


# In[77]:


test_y.shape


# In[78]:


lg=LogisticRegression()


# In[79]:


lg.fit(train_x,train_y)


# In[80]:


pred=lg.predict(test_x)


# In[81]:


pred


# In[82]:


print("Accuracy score:",accuracy_score(test_y,pred))


# In[83]:


print(confusion_matrix(test_y,pred))


# In[84]:


print(classification_report(test_y,pred))


# In[ ]:




