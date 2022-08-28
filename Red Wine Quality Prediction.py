#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score 


# # Data Collection

# In[8]:


#Loading the dataset to a Pandas Dataframe

wine_dataset = pd.read_csv("Winequality-red.csv")


# In[10]:


#number of rows & collumn in the dataset
wine_dataset.shape


# In[11]:


#first 5 rows of the dataset 
wine_dataset.head()


# In[12]:


# checking for missing values 

wine_dataset.isnull().sum()


# # Data Analysis and Visulaization 

# In[13]:


# Statistical measure of the dataset 

wine_dataset.describe()


# In[14]:


# number of values for each quality 
sns.catplot(x='quality', data= wine_dataset, kind ='count')


# In[15]:


# volatile acidity vs quality 
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='volatile acidity', data= wine_dataset)


# In[16]:


# citric acid vs quality 
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='citric acid', data= wine_dataset)


# # Correlation

# In[17]:


correlation = wine_dataset.corr()


# In[18]:


plt.figure(figsize=(10,10))
sns.heatmap(correlation,cbar=True, square= True, fmt='.1f',annot= True, annot_kws={'size':8}, cmap='Blues')


# # Data Preprocessing 

# In[19]:


#separate the data and lebel 
X= wine_dataset.drop('quality', axis=1)


# In[20]:


print(X)


# Label Binarization

# In[23]:


Y= wine_dataset['quality'].apply(lambda y_value:1 if y_value>=7 else 0)
print(Y)


# Train and Test Split

# In[24]:


X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2, random_state=3)


# In[25]:


print(Y.shape, Y_train.shape, Y_test.shape)


# Model Training

# Random Forest Classifier 

# In[27]:


model= RandomForestClassifier()


# In[28]:


model.fit(X_train, Y_train)


# Model Evaluation

# Accuracy Score 

# In[29]:


# accuracy on test data 
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[30]:


print('Accuracy:', test_data_accuracy)


# Building a Predictive System 

# In[32]:


input_data = (7.3,0.65,0.0,1.2,0.065,15.0,21.0,0.9946,3.39,0.47,10.0,)

# Changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the data as we are predicting the label for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==1):
    print('Good Quality Wine')
else:
    print('Bad Quality Wine')


# In[33]:


input_data = (7.5,0.5,0.36,6.1,0.071,17.0,102.0,0.9978,3.35,0.8,10.5,)

# Changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the data as we are predicting the label for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==1):
    print('Good Quality Wine')
else:
    print('Bad Quality Wine')

