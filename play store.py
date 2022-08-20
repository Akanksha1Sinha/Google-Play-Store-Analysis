#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("C:\\Users\\anuus\\OneDrive\\Desktop\\class\\googleplaystore.csv")


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.columns


# # Data clean up – Missing value treatment

# In[8]:


#Drop records where rating is missing since rating is our target/study variable


# In[9]:


df.isna().sum()


# In[10]:


df[df['Rating'].isnull()]


# In[11]:


df.dropna(subset = ['Rating'], inplace=True)


# In[12]:


df.Rating.isna().sum()


# In[13]:


df.shape


# In[14]:


df.isna().sum()


# In[15]:


#  Check the null values for the Android Ver column. 


# In[16]:


df[df['Android Ver'].isnull()]


# In[17]:


#    Are all 3 records having the same problem?

#    Yes, all 3 columns has same problem.


# In[18]:


#    Drop the 3rd record 


# In[19]:


df.drop([10472],inplace=True)


# In[20]:


df[df['Android Ver'].isnull()]


# In[21]:


#    Replace remaining missing values with the mode


# In[22]:


df1=df['Android Ver'].mode()


# In[23]:


df1


# In[24]:


df['Android Ver'].fillna(value=df1[0], inplace = True)


# In[25]:


df.isna().sum()


# In[26]:


#    Current ver – replace with most common value


# In[27]:


df[df['Current Ver'].isnull()]


# In[28]:


df1=df['Current Ver'].mode()


# In[29]:


df1


# In[30]:


df['Current Ver'].fillna(value=df1[0], inplace = True)


# In[31]:


df.isna().sum()


# # Data clean up – correcting the data types

# In[32]:


df.info()


# In[33]:


#        Size – remove ‘,’,‘+’,'M','K' sign, convert to integer


# In[34]:


df['Size'].unique()


# In[35]:


df['Size'] = df.Size.apply(lambda x: x.strip('+'))# Removing the + Sign


# In[36]:


df['Size'] = df.Size.apply(lambda x: x.replace(',', ''))# For removing the `,`


# In[37]:


df['Size'] = df.Size.apply(lambda x: x.replace('M', 'e+6'))# For converting the M to Mega


# In[38]:


df['Size'] = df.Size.apply(lambda x: x.replace('k', 'e+3'))# For convertinf the K to Kilo


# In[39]:


df['Size'] = df.Size.replace('Varies with device', np.NaN)


# In[40]:


df['Size'] = pd.to_numeric(df['Size']) # Converting the string to Numeric type


# In[41]:


df.info()


# In[42]:


df.isna().sum()


# In[43]:


df.dropna(subset = ['Size'], inplace=True)


# In[44]:


df.info()


# In[45]:


#    Installs – remove ‘,’ and ‘+’ sign, convert to integer


# In[46]:


df['Installs'].value_counts()


# In[47]:


df['Installs'].value_counts(normalize=True)


# In[48]:


df['Installs'] = df.Installs.apply(lambda x: x.strip('+'))


# In[49]:


df['Installs'] = df.Installs.apply(lambda x: x.replace(',', ''))


# In[50]:


df['Installs'] = pd.to_numeric(df['Installs'])


# In[51]:


df.info()


# In[52]:


#    Price variable – remove $ sign and convert to float


# In[53]:


df['Price'].value_counts()


# In[54]:


df['Price'] = df.Price.apply(lambda x: x.strip('$'))


# In[55]:


df['Price'] = pd.to_numeric(df['Price'])


# In[56]:


df.info()


# In[57]:


#    Convert all other identified columns to numeric


# In[58]:


df['Reviews'] = df.Reviews.astype(int)


# In[59]:


df.info()


# # Sanity checks – check for the following and handle accordingly

# In[60]:


df[df.Rating>5]


# In[61]:


#      Reviews should not be more than installs as only those who installed can review the app.


# In[62]:


df[df["Reviews"]>df["Installs"]]


# In[63]:


df.drop(df[df["Reviews"]>df["Installs"]].index,inplace=True)


# In[64]:


df[df["Reviews"]>df["Installs"]]


# # Identify and handle outliers – 

# # Price column

# In[65]:


#    Make suitable plot to identify outliers in price 


# In[66]:


sns.boxplot(df.Price)


# In[67]:


#        Do you expect apps on the play store to cost $200? Check out these cases


# In[68]:


df[df["Price"]>200]


# In[69]:


#      Limit data to records with price < $30


# In[70]:


df[(df['Price']>30)]


# In[71]:


df.drop(df[df["Price"]>30].index,inplace=True)


# In[72]:


df[df["Price"]>30]


# In[73]:


df[(df['Reviews']>1000000)]


# In[74]:


df.drop(df[df["Reviews"]>1000000].index,inplace=True)


# In[75]:


df[df["Reviews"]>1000000]


# In[76]:


#    Installs


# In[77]:


#    What is the 95th percentile of the installs?


# In[78]:


percentile = df.Installs.quantile(0.95) 
print(percentile,"is 95th percentile of Installs")


# In[79]:


#     Drop records having a value more than the 95th percentile


# In[80]:


df.drop(df[df["Installs"]>10000000.0].index,inplace=True)


# In[81]:


df[df["Installs"]>10000000.0]


# # Data analysis to answer business questions

# In[82]:


#     What is the distribution of ratings like? (use Seaborn) More skewed towards higher/lower values?


# In[83]:


sns.distplot(df['Rating'])
plt.show()
print('The skewness of this distribution is',df['Rating'].skew())


# In[84]:


#    What are the top Content Rating values?
#    Are there any values with very few records?
#    If yes, drop those as they won’t help in the analysis


# In[85]:


df['Content Rating'].value_counts()     #Everyone is highest


# In[86]:


df.drop(df[df['Content Rating']=="Unrated"].index , inplace=True)
df.drop(df[df['Content Rating']=="Adults only 18+"].index,inplace=True)


# In[87]:


#     Effect of size on rating
#     Make a joinplot to understand the effect of size on rating
#     Do you see any patterns?
#     How do you explain the pattern?


# In[88]:


sns.jointplot(x ='Size', y='Rating', data = df)
plt.show()


# In[89]:


#    Effect of price on rating


# In[90]:


#    Make a jointplot (with regression line)


# In[91]:


sns.jointplot(x='Price', y='Rating', data=df, kind='reg')
plt.show()


# In[92]:


df.corr()


# # Look at all the numeric interactions together 

# In[97]:


sns.pairplot(df, vars=['Reviews', 'Size', 'Rating', 'Price'], kind='reg')
plt.show()


# # Rating vs. content rating

# In[98]:


#     Make a bar plot displaying the rating for each content rating


# In[99]:


df.groupby(['Content Rating'])['Rating'].count().plot.bar(color="darkgreen")
plt.show()


# In[100]:


plt.boxplot(df['Rating'])
plt.show()


# In[ ]:




