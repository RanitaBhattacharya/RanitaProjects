#!/usr/bin/env python
# coding: utf-8

# Enhanced operation in Dictionary

# In[3]:


#dictionary with integer keys
intdict={10:"C++",20:"java",30:"python", 40:"Ruby" }


# In[4]:


intdict.keys()


# In[6]:


intdict.values()


# In[7]:


intdict.items()


# In[11]:


#adding a new key to a dictionary 

intdict.update({50:"Rails"})


# In[12]:


intdict


# In[13]:


#updating a value in given key

intdict[40]="Csharp"


# In[14]:


intdict


# In[15]:


#delete a key 
del intdict[50]


# In[16]:


intdict


# In[24]:


intdict.pop(40)


# In[25]:


intdict


# In[26]:


intdict.items()


# In[35]:


dict2={'id':[1,2,3], 'name':[Jardan,Ricky,Brawen],'Rank':[first,second,theird]}
dict2{}


# In[31]:


newdict={1:["Red":"Yelow":"Blue"],2:["white":"pink":"purple"],3:["brown":"black","light green"]}


# In[32]:


newdict


# In[ ]:




