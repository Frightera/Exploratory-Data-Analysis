#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Exploratory Data Analysis - World

# #### Import required libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

print('Libraries imported! :)')


# #### Import data

# In[2]:


data = pd.read_csv('covid_19_data.csv', sep = ',')
data.head(-3)


# In[3]:


data.info()


# * Check for nan values and fill them with 'Unknown'

# In[4]:


data.isna().sum()


# In[5]:


data["Province/State"]= data["Province/State"].fillna('Unknown')
data.isna().sum()


# In[6]:


data['Country/Region'].value_counts()


# In[7]:


# Change as China rather than Mainland China
data['Country/Region'] = data['Country/Region'].replace('Mainland China', 'China')


# In[8]:


data['Active_cases'] = data['Confirmed'] - data['Deaths'] - data['Recovered']
data.head()


# In[9]:


# We've gotten the latest stats
df = data[data['ObservationDate'] == max(data['ObservationDate'])].reset_index()
df.head()


# In[10]:


df_world = df.groupby(["ObservationDate"])[["Confirmed","Active_cases","Recovered","Deaths"]].sum().reset_index()
df_world.head()


# ## EDA

# In[11]:


labels = ["Active Cases","Recovered","Deaths"]
values = df_world.loc[0, ["Active_cases","Recovered","Deaths"]]
fig = px.pie(df_world, values=values, names=labels,color_discrete_sequence=['rgb(77,146,33)','rgb(69,144,185)','rgb(77,77,77)'],hole=0.7)
fig.update_layout(
    title='Total cases in the world : '+str(df_world["Confirmed"][0]),
)
fig.show()


# In[12]:


by_time_data = data.groupby(["ObservationDate"])[
    ["Confirmed","Active_cases","Recovered","Deaths"]].sum().reset_index().sort_values("ObservationDate",ascending=True).reset_index(drop=True)
by_time_data.head(-4)    


# In[13]:


columns = ['Confirmed', 'Active_cases', 'Recovered', 'Deaths']

sns.set_style("darkgrid")
for each in columns:
    fig, ax = plt.subplots(figsize=(13, 9))
    sns.lineplot(x = by_time_data.index, y = each, data = by_time_data)
    plt.xlabel('days', fontsize = 14)
    plt.ylabel(str(each) + ' cases', fontsize = 14)
    plt.title(str(each) + ' cases over the World', fontsize = 14)
    ax.yaxis.set_major_formatter(ticker.EngFormatter())
    plt.show()


# In[14]:


each_country_data = df.groupby(["Country/Region"])["Confirmed","Active_cases","Recovered","Deaths"].sum().reset_index().sort_values("Confirmed",ascending=False).reset_index(drop=True)


# In[15]:


headerColor = 'grey'
rowEvenColor = 'lightgrey'
rowOddColor = 'white'

fig = go.Figure(data=[go.Table(
  header=dict(
    values=['<b>Country</b>','<b>Confirmed Cases</b>'],
    line_color='darkslategray',
    fill_color=headerColor,
    align=['left','center'],
      
    font=dict(color='white', size=12)
  ),
  cells=dict(
    values=[
      each_country_data['Country/Region'],
      each_country_data['Confirmed'],
      ],
    line_color='darkslategray',
    # 2-D list of colors for alternating rows
    fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor]*len(each_country_data)],
    align = ['left', 'center'],
    font = dict(color = 'darkslategray', size = 11)
    ))
])
fig.update_layout(
    title='Confirmed Cases In Each Country',
)
fig.show()


# In[16]:


each_country_data.head()


# In[17]:


sns.set_style("darkgrid")
plt.figure(figsize= (15,9))
ax = sns.barplot(x = each_country_data.iloc[:10,1], y = each_country_data.iloc[:10,0], data = each_country_data)
plt.xlabel('Deaths', fontsize = 14)
plt.ylabel('Country/Region', fontsize = 14)
plt.title('Highest 10 Confirmed Cases over the World', fontsize = 14)
ax.xaxis.set_major_formatter(ticker.EngFormatter())
plt.show()


# In[18]:


sorted_active_cases = (each_country_data.sort_values(by = ['Active_cases'], ascending = False).reset_index(drop = True))
sorted_active_cases.head(-3)


# In[19]:


sns.set_style("darkgrid")
plt.figure(figsize= (15,9))
ax = sns.barplot(x = sorted_active_cases.iloc[:10,2], y = sorted_active_cases.iloc[:10,0], data = sorted_active_cases)
plt.xlabel('Active Cases', fontsize = 14)
plt.ylabel('Country', fontsize = 14)
plt.title('Highest 10 Active Cases over the World', fontsize = 14)
ax.xaxis.set_major_formatter(ticker.EngFormatter())
plt.show()


# In[20]:


sorted_recovered_cases = (each_country_data.sort_values(by = ['Recovered'], ascending = False).reset_index(drop = True))
sorted_recovered_cases.head(-3)


# In[21]:


sns.set_style("darkgrid")
plt.figure(figsize= (15,9))
ax = sns.barplot(x = sorted_recovered_cases.iloc[:10,3], y = sorted_recovered_cases.iloc[:10,0], data = sorted_recovered_cases)
plt.xlabel('Recovered', fontsize = 14)
plt.ylabel('Country/Region', fontsize = 14)
plt.title('Highest 10 Recovered Cases over the World', fontsize = 14)
ax.xaxis.set_major_formatter(ticker.EngFormatter())
plt.show()


# In[22]:


sorted_death_cases = (each_country_data.sort_values(by = ['Deaths'], ascending = False).reset_index(drop = True))
sorted_death_cases.head(-3)


# In[23]:


sns.set_style("darkgrid")
plt.figure(figsize= (15,9))
ax = sns.barplot(x = sorted_death_cases.iloc[:10,4], y = sorted_death_cases.iloc[:10,0], data = sorted_death_cases)
plt.xlabel('Country', fontsize = 14)
plt.ylabel('Deaths', fontsize = 14)
plt.title('Highest 10 Deaths over the World', fontsize = 14)
ax.xaxis.set_major_formatter(ticker.EngFormatter())
plt.show()


# In[24]:


data_corona_evolution = data.groupby(["Country/Region","ObservationDate"])[["Confirmed","Active_cases","Recovered","Deaths"]].sum().reset_index().sort_values("ObservationDate",ascending=True).reset_index(drop=True)

for each in columns:    
    fig = px.choropleth(data_corona_evolution, locations=data_corona_evolution['Country/Region'],
                        color=data_corona_evolution[each],locationmode='country names', 
                        hover_name=data_corona_evolution['Country/Region'], 
                        color_continuous_scale=px.colors.sequential.deep,
                        animation_frame="ObservationDate")
    fig.update_layout(

        title='Evolution of ' +str(each) + ' cases In Each Country',
    )
    fig.show()


# In[25]:


fig = go.Figure(data=[go.Scatter(
    x=each_country_data['Country/Region'][0:10],
    y=each_country_data['Confirmed'][0:10],
    mode='markers',
    
    marker=dict(
        color=100+np.random.randn(500),
        size=(each_country_data['Confirmed'][0:10]/45000),
        showscale=True
        )
)])

fig.update_layout(
    title='Most 10 Infected Countries',
    xaxis_title="Countries",
    yaxis_title="Confirmed Cases",
    template='plotly_dark'
)
fig.show()

