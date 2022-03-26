#!/usr/bin/env python
# coding: utf-8

# Posetive Indexing

# In[31]:


fruitlist=["Apple","Bannana","Grapes","papaya","Kiwi","Berry"]


# In[32]:


fruitlist[4]


# In[33]:


fruitlist[0]


# In[34]:


fruitlist[2]


# In[35]:


fruitlist[3:5]


# In[36]:


fruitlist[1:5]


# In[37]:


fruitlist[1:4]


# Negative Indexing

# In[38]:


fruitlist


# In[39]:


fruitlist[-2]


# In[40]:


fruitlist[-5]


# Fruit Slicing

# In[41]:


fruitlist


# In[42]:


fruitlist[-2:-1]


# In[43]:


fruitlist[-3:-2]


# In[44]:


fruitlist[-3:-1]


# In[45]:


fruitlist[-4:-1]


# In[46]:


fruitlist[-5:-1]


# In[47]:


fruitlist[:]


# In[48]:


fruitlist[0:]


# In[49]:


fruitlist[:-1]


# In[50]:


fruitlist[-2:-1]


# In[51]:


fruitlist[-3:-1]


# In[52]:


fruitlist[-4:-1]


# In[53]:


fruitlist[-5:-1]


# In[54]:


fruitlist[-5:0]


# In[55]:


fruitlist[0:]


# In[56]:


fruitlist[:-1]


# Planet list

# In[57]:


plantlist=["Mercury","Venus","Earth"]


# In[58]:


len(plantlist)


# In[59]:


plantlist.append("saturn")


# In[60]:


plantlist


# In[61]:


plantlist.append("Mars")


# In[62]:


plantlist


# In[63]:


plantlist.insert(1,"pluto")


# In[64]:


plantlist


# In[66]:


plantlist.sort()


# In[67]:


plantlist


# In[68]:


plantlist.reverse()


# In[69]:


plantlist


# In[70]:


plantlist.extend(["Uranus","jupiter"])


# In[71]:


plantlist


# In[72]:


plantlist.pop()


# In[73]:


plantlist.index("Mars")


# In[74]:


plantlist.pop(4)


# In[75]:


plantlist


# In[77]:


plantlist.remove("pluto")


# In[78]:


plantlist


# In[80]:


plantlist.clear()


# In[81]:


plantlist


# In[82]:


del plantlist


# In[84]:


list1=[30,40,50,60]
list1


# In[85]:


list1[2]


# In[87]:


list1[3]


# In[ ]:


list[2]=677


# Tuples

# In[88]:


t1=(1,2,3,4,5)


# In[89]:


t1


# In[91]:


t1[2]


# Tuples are immutable

# In[ ]:




