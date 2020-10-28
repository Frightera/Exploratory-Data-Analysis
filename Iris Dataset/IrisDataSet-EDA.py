# # Iris Dataset Exploratory Data Analysis

# ## Analyzing Data

# #### Importing libraries

# In[1]:


import pandas as pd 
import numpy as np
print("Libraries are imported!")


# #### Load the data using read_csv method which is avaliable in pandas.

# In[2]:


data = pd.read_csv("Iris.csv")
print("Data loaded")


# #### Take a look at the data by using head() and tail()

# In[3]:


print(data.head()) #first 5 rows
print("-----------------------------------------------------------------------------")
print(data.tail()) # last five rows


# #### We need more information about our data in order to draw plots etc.
# - Exploring data one by one.

# In[4]:


print("data consists of {} (rows and columns)" .format(data.shape))


# In[5]:


data.columns # aka features


# In[6]:


data.dtypes # data types


# In[7]:


data.info() # if you want to see much more in one line code.


# #### As you might have noticed, species column contains 'Iris-' in the beginning.
# - By removing it, we can make the names shorter.

# In[8]:


data.drop(['Id'], axis = 1, inplace = True) # we wont use id in our data
data['Species'].head(-5)


# In[9]:


data.Species.value_counts() # let's remove "Iris-"


# In[10]:


data.Species = data.Species.str.replace('Iris-', '')
data.Species.head(-5)


# #### Let's be more statistical

# In[11]:


species = data.groupby(data['Species'])
species.describe()
# not easy to read?


# In[12]:


features = data.iloc[:,0:5]
features.head()


# In[13]:


features_table = features.describe()
features_table.rename({'50%': 'median'}, inplace = True)
# inplace = True statements overwrites the current dataframe that we are working on.
features_table


# In[14]:


data.groupby('Species').agg(['mean', 'median'])  # passing a list of recognized strings
data.groupby('Species').agg([np.mean, np.median])  # passing a list of explicit aggregation functions


# ## Visualization of the Data

# In[15]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
print('Successful')


# In[16]:


data.head()


# In[17]:


from random import randrange
colors = ['b', 'r', 'g', 'y']
for x in data.columns:
    if x == "Species":
        break
    for y in data.columns:
        if y == 'Species' or y == x:
            continue
        plt.figure(x)
        plt.scatter(data[x],data[y], color = colors[randrange(4)])
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()


# In[18]:


#check error message
import seaborn as sns
for each in data:
    plt.figure(each)
    sns.distplot(data[each], bins = 30)


# #### ValueError: could not convert string to float: 'setosa', hmm it seems like we don't want strings when we are plotting

# In[19]:


data['Species'] = [1 if each == 'setosa' else 2 if each == 'versicolor' else 3 for each in data['Species']]


# In[20]:


data.iloc[np.random.choice(np.arange(len(data)), 10, False)]


# #### Let's try it again..

# In[21]:


for each in data:
    plt.figure(each)
    sns.distplot(data[each], bins = 30, kde = False) #kde flag = false, only hist will be drawn.
    plt.ylabel('Frequency')
    plt.title("Distribution of {}" .format(each))


# #### Histogram of features

# In[22]:


type(features)


# In[23]:


g = features.plot.hist(bins = 30, alpha = 0.3)
g.set_xlabel('Size (in CM)')
plt.show()


# #### Boxplot and see outliers (if exists)

# In[31]:


ax = sns.boxplot(data=features, orient="h", palette="Set3")
ax.set_xlabel('Size')


# #### Examine the correlation between each of the measurements

# In[34]:


sns.pairplot(features, hue='Species');

