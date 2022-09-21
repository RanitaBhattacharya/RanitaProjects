#!/usr/bin/env python
# coding: utf-8

# # Avacado Project
# 

# # Importing Libraries

# In[17]:


import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings 
warnings.filterwarnings('ignore')


# # Feature Warnings 

# In[18]:


#import dataset

data = pd.read_csv('avocado.csv')


# In[19]:


data


# In[20]:


data.head()


# In[21]:


data.info()


# In[22]:


data.describe()


# In[23]:


data.isnull().sum()


# In[24]:


# renaming column names into meaningful names (refer kaggle's avacado dataset description)
data = data.rename(columns={'4046':'PLU_4046','4225':'PLU_4225','4770':'PLU_4770'})


# In[25]:


# removing unnecessary column
data = data.drop(['Unnamed: 0'],axis = 1)
data.head(10)


# In[26]:


data.tail()


# In[28]:


data['Date']=pd.to_datetime(data['Date'])
data['Month']=data['Date'].apply(lambda x:x.month)
data['Day']=data['Date'].apply(lambda x:x.day)
data.head()


# # Analysis of Average Prices

# In[29]:


import matplotlib.pyplot as plt

byDate=data.groupby('Date').mean()
plt.figure(figsize=(17,8),dpi=100)
byDate['AveragePrice'].plot()
plt.title('Average Price')


# **Hence the plot shows the average price of avocado at various points of time**
# 
# 

# In[30]:


byMonth = data.groupby("Month").mean()
plt.figure(figsize=(17,8),dpi=100)
plt.plot(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sept","Oct","Nov","Dec"],byMonth['AveragePrice'])
plt.title('Average Price Per Month')


# **From the above graph plotted for average price of avocado per month we can observe that the price rises for a while in February to March then it falls in April and then the month of May witnesses a rise in the average price. This rise reaches its zenith in the month of October and henceforth it starts to fall.**

# In[31]:


byDay = data.groupby("Day").mean()
plt.figure(figsize=(17,8),dpi=100)
byDay['AveragePrice'].plot()
plt.title('Average Price Per Day')


# **The above graph for average price per day implies that the price fluctuates in a similar manner at a regular interval.**
# 
# 

# In[32]:


import seaborn as sns

byRegion=data.groupby('region').mean()
byRegion.sort_values(by=['AveragePrice'], ascending=False, inplace=True)
plt.figure(figsize=(17,8),dpi=100)
sns.barplot(x = byRegion.index,y=byRegion["AveragePrice"],data = byRegion,palette='rocket')
plt.xticks(rotation=90)
plt.xlabel('Region')
plt.ylabel('Average Price')
plt.title('Average Price According to Region')


# # Analysis of Several Seasons

# In[10]:


# convert the type of Date feature from obj to datetime type
data['Date'] = pd.to_datetime(data['Date'])


# In[11]:


# categorizing into several seasons
def season_of_date(date):
    year = str(date.year)
    seasons = {'spring': pd.date_range(start='21/03/'+year, end='20/06/'+year),
               'summer': pd.date_range(start='21/06/'+year, end='22/09/'+year),
               'autumn': pd.date_range(start='23/09/'+year, end='20/12/'+year)}
    if date in seasons['spring']:
        return 'spring'
    if date in seasons['summer']:
        return 'summer'
    if date in seasons['autumn']:
        return 'autumn'
    else:
        return 'winter'


# In[12]:


# creating a new feature 'season' and assign the corresponding season for the Date using map function over our season_of_date function
data['season'] = data.Date.map(season_of_date)


# In[13]:


# now, we can see the season feature appended at the last
data.head(10)


# In[14]:


# no of observations for each seasons
data.season.value_counts()


# In[15]:


# droping date feature
data = data.drop(['Date'],axis = 1)


# # Taking Care of the Outliers

# In[33]:


plt.figure(figsize=(15,7),dpi=100)
sns.boxplot(data = data[[
 'AveragePrice',
 'Total Volume',
 '4046',
 '4225',
 '4770',
 'Total Bags',
 'Small Bags',
 'Large Bags',
 'XLarge Bags']])


# In[ ]:





# # Data Preprocessing
# 

# In[16]:


# converting categorical features of text data into model-understandable numerical data
label_cols = ['type','region','season']
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
data[label_cols] = data[label_cols].apply(lambda x : label.fit_transform(x))


# In[17]:


# Scaling the features and 
# spliting the label encoded features into distinct features inorder to prevent our model to think that columns have data with some kind of order or hierarchy
# column_tranformer allows us to combine several feature extraction or transformation methods into a single transformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
scale_cols = data.drop(['AveragePrice','type','year','region','season'],axis=1)
col_trans = make_column_transformer(
            (OneHotEncoder(), data[label_cols].columns),
            (StandardScaler(), scale_cols.columns),
            remainder = 'passthrough')


# # Train Test Split

# In[18]:


# splitting our dataset into train and test set such that 20% of observations are considered as test set
X = data.drop(['AveragePrice'],axis=1)
y = data.AveragePrice
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# # Regression Models
# 

# **Linear Regression**

# In[19]:


from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
pipe = make_pipeline(col_trans,linreg)
pipe.fit(X_train,y_train)


# In[20]:


y_pred_test = pipe.predict(X_test)


# In[21]:


from sklearn.metrics import mean_absolute_error,mean_squared_error
print('MAE for testing set: {}'.format(mean_absolute_error(y_pred_test,y_test)))
print('MSE for testing set: {}'.format(mean_squared_error(y_pred_test,y_test)))
print('RMSE for testing set: {}'.format(np.sqrt(mean_squared_error(y_pred_test,y_test))))


# **Support Vector Regressor (SVR)**

# In[28]:


from sklearn.svm import SVR
svr = SVR()
pipe = make_pipeline(col_trans,svr)
pipe.fit(X_train,y_train)


# In[29]:


y_pred_test = pipe.predict(X_test)


# In[30]:


print('MAE for testing set: {}'.format(mean_absolute_error(y_pred_test,y_test)))
print('MSE for testing set: {}'.format(mean_squared_error(y_pred_test,y_test)))
print('RMSE for testing set: {}'.format(np.sqrt(mean_squared_error(y_pred_test,y_test))))


# # Decision Tree Regressor

# In[31]:


from sklearn.tree import DecisionTreeRegressor
dr=DecisionTreeRegressor()
pipe = make_pipeline(col_trans,dr)
pipe.fit(X_train,y_train)


# In[32]:


y_pred_test = pipe.predict(X_test)


# In[33]:


print('MAE for testing set: {}'.format(mean_absolute_error(y_pred_test,y_test)))
print('MSE for testing set: {}'.format(mean_squared_error(y_pred_test,y_test)))
print('RMSE for testing set: {}'.format(np.sqrt(mean_squared_error(y_pred_test,y_test))))


# # Random Forest Regressor

# In[ ]:



from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor()
pipe = make_pipeline(col_trans,forest_model)
pipe.fit(X_train,y_train)


# In[ ]:


y_pred_test = pipe.predict(X_test)


# In[ ]:


print('MAE for testing set: {}'.format(mean_absolute_error(y_pred_test,y_test)))
print('MSE for testing set: {}'.format(mean_squared_error(y_pred_test,y_test)))
print('RMSE for testing set: {}'.format(np.sqrt(mean_squared_error(y_pred_test,y_test))))


# In[40]:


sns.distplot((y_test-y_pred_test),bins=50)


# In[ ]:




