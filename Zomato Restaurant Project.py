#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


zomato_data=pd.read_csv('zomato(2).csv')


# In[4]:


zomato_data


# In[5]:


zomato_data.head()


# In[6]:


zomato_data.shape


# In[7]:


zomato_data.columns


# In[8]:


zomato_data.info()


# In[9]:


zomato_data.describe()


# In[10]:


zomato_data.isnull().sum()


# In[11]:


[features for features in zomato_data.columns if zomato_data[features].isnull().sum()>0]


# In[13]:


plt.figure(figsize=(12,6))
sns.heatmap(zomato_data.isnull(), cbar=False,yticklabels=False,cmap="YlGnBu")


# **very very less number of data are misssing thats why we can see the thin black line in the above**
# 

# In[14]:


pip install openpyxl


# In[17]:


data_country=pd.read_csv('CountryCode.csv')
data_country.head()


# In[19]:


zomato_data.columns


# # Merge the table

# In[20]:


final_df=pd.merge(zomato_data,data_country,on='Country Code',how='left')
final_df.head(3)


# In[21]:


final_df.dtypes


# # Zomato is mostly availabe in india ,a/c to the records

# In[22]:


final_df.Country.value_counts()


# In[23]:


country_name=final_df.Country.value_counts().index
country_name


# In[24]:


country_value=final_df.Country.value_counts().values
country_value


# In[25]:


plt.pie(country_value[:3],labels=country_name[:3],autopct='%1.3f%%')


# # Maximum records or transaction are from India,then US,UK

# In[26]:


final_df.groupby(['Aggregate rating','Rating color','Rating text']).size()


# In[27]:


ratings=final_df.groupby(['Aggregate rating','Rating color','Rating text']).size().reset_index().rename(columns={0:'Rating counts'})
ratings


# # Observation

# 1. when rating is btwn 4.5 to 4.9--------------->Excellent
# 2. when rating is btwn 4.0 to 4.4--------------->very good
# 3. when rating is btwn 3.5 to 3.9--------------->good
# 4. when rating is btwn 3.0 to 3.4--------------->average
# 5. when rating is btwn 2.5 to 2.9--------------->average
# 6. when rating is btwn 2.0 to 2.4--------------->poor
# 

# In[29]:


plt.figure(figsize=(12,6))
sns.barplot(x='Aggregate rating',y='Rating counts',data=ratings)


# In[30]:


#palette is used to give color as per your reqiirement or color mapping
plt.figure(figsize=(12,6))
sns.barplot(x='Aggregate rating',y='Rating counts',data=ratings,hue='Rating color',palette=['beige','red','orange','yellow','green','green'])


# # Maximum number of rating are btwn 2.5 to 3.4
# 

# In[31]:


sns.countplot(x='Rating color',data=ratings,palette=['beige','red','orange','yellow','green','green'])


# In[32]:


final_df.columns


# In[33]:


final_df.groupby(['Aggregate rating','Country']).size().reset_index().head()


# # Maximum number of 0 ratings are from indian customers
# 
# 

# In[34]:


final_df.groupby(['Country','Currency']).size().reset_index()


# In[35]:


final_df.columns


# In[36]:


final_df.groupby(['Country','Has Online delivery']).size().reset_index()


# # Country in which food delivery online

# In[37]:


final_df[final_df['Has Online delivery']=='Yes'].Country.value_counts()


# **Online delivery in india and uae only from given dataset**

# In[38]:


final_df.groupby(['Has Online delivery']).size().reset_index()


# In[39]:


#pie chart w.r.t city
final_df.City.value_counts().index


# In[40]:


city_value=final_df.City.value_counts().values
city_labels=final_df.City.value_counts().index


# In[41]:


plt.pie(city_value[:5],labels=city_labels[:5],autopct='%1.3f%%')


# **# Maximum nuber of transaction happen from New delhi then grugram ,Noida**
