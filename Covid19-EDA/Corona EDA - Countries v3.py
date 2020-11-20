#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Exploratory Data Analysis

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


# In[11]:


countries = df['Country/Region'].values

ctr_arr = np.array(countries) 
np.unique(ctr_arr)


# ## EDA

# ### Implemention of some methods
# * So now we don't have to write over and over again when plotting.
# * You can plot any country by using get_country & plot_all method.
# * If they are being plotted as unknown, then there is no info provided for states.

# In[12]:


columns = ['Confirmed', 'Active_cases', 'Recovered', 'Deaths']

def split_date(data):
    data["datetime"] = pd.to_datetime(data["ObservationDate"])
    data["month"] = data["datetime"].dt.month
    return data

def get_country(data,country):
    data = data[(data['Country/Region'] == country) ].reset_index(drop=True)
    split_date(data)
    return data

def get_actual_values(data):
    data = data[data['ObservationDate'] == max(data['ObservationDate'])].reset_index()
    data.drop(['SNo', 'Last Update'], axis = 1, inplace = True)
    return data

def line_plot_day_by_day(data):
    data_by_time = data.groupby(["ObservationDate","Country/Region"])["Confirmed","Deaths","Recovered","Active_cases"].sum().reset_index(drop=True)
    plt.figure(figsize = (14,9))
    ax = sns.lineplot(x = data_by_time.index  , y = 'Confirmed' ,label = 'Confirmed', data = data_by_time)
    sns.lineplot(x = data_by_time.index  , y = 'Active_cases' ,label = 'Active_Cases', data = data_by_time)
    sns.lineplot(x = data_by_time.index  , y = 'Deaths' ,label = 'Deaths', data = data_by_time)
    sns.lineplot(x = data_by_time.index  , y = 'Recovered' ,label = 'Recovered', data = data_by_time)
    ax.yaxis.set_major_formatter(ticker.EngFormatter())
    plt.title('Plotted day-by-day')
    plt.legend()

def line_plot_month(data):
    split_date(data)
    plt.figure(figsize = (14,9))
    ax = sns.lineplot(x = data.month  , y = 'Confirmed' ,label = 'Confirmed', data = data)
    sns.lineplot(x = data.month  , y = 'Active_cases' ,label = 'Active_Cases', data = data)
    sns.lineplot(x = data.month  , y = 'Deaths' ,label = 'Deaths', data = data)
    sns.lineplot(x = data.month  , y = 'Recovered' ,label = 'Recovered', data = data)
    plt.title('Plotted month-by-month')
    ax.yaxis.set_major_formatter(ticker.EngFormatter())
    plt.legend()
    plt.show()

def scatter_evolution(data):
    data = data.groupby(["ObservationDate","Country/Region"])["Confirmed","Deaths","Recovered","Active_cases"].sum().reset_index().reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['ObservationDate'], y=data['Confirmed'],
                        mode='lines',
                        name='Confirmed cases'))

    fig.add_trace(go.Scatter(x=data['ObservationDate'], y=data['Active_cases'],
                        mode='lines',
                        name='Active cases',line=dict( dash='dot')))
    fig.add_trace(go.Scatter(x=data['ObservationDate'], y=data['Deaths'],name='Deaths',
                                       marker_color='black',mode='lines',line=dict( dash='dot') ))
    fig.add_trace(go.Scatter(x=data['ObservationDate'], y=data['Recovered'],
                        mode='lines',
                        name='Recovered cases',marker_color='green'))
    fig.update_layout(
        title='Evolution of cases by time',
            template='plotly_white'
    )
    fig.show()

    
def bar_plot_province(data):
    get_actual_values(data)
    data = data.groupby(["Province/State"])["Confirmed","Active_cases","Recovered","Deaths"].sum().reset_index().sort_values("Confirmed",ascending=False).reset_index(drop=True)
    for each in columns:
        plt.figure(figsize = (14,9))
        ax = sns.barplot(x = each, y = 'Province/State', data = data)
        ax.xaxis.set_major_formatter(ticker.EngFormatter())
        plt.title(str(each) + ' cases by time in each Province')

def province_cases_pie(data):
    data = data[data['ObservationDate'] == max(data['ObservationDate'])].reset_index()
    data_state = data.groupby(["Province/State"])["Confirmed","Active_cases","Recovered","Deaths"].sum().reset_index().sort_values("Confirmed",ascending=False).reset_index(drop=True)
    fig = px.pie(data_state, values=data_state['Confirmed'], names=data_state['Province/State'],
                 title='Confirmed cases in provinces in ' + str(data['Country/Region'][0]),
                hole=.2)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.show()
    fig = px.pie(data_state, values=data_state['Active_cases'], names=data_state['Province/State'],
                 title='Active cases in provinces in ' + str(data['Country/Region'][0]),
                hole=.2)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.show()
    fig = px.pie(data_state, values=data_state['Recovered'], names=data_state['Province/State'],
                 title='Recovered cases in provinces in ' + str(data['Country/Region'][0]),
                hole=.2)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.show()
    fig = px.pie(data_state, values=data_state['Deaths'], names=data_state['Province/State'],
                 title='Death cases in ' + str(data['Country/Region'][0]),
                hole=.2)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.show()
    
        
def total_cases_pie(data):
    data = data[data['ObservationDate'] == max(data['ObservationDate'])].reset_index()
    data = data.groupby(["Country/Region"])["Confirmed","Deaths","Recovered","Active_cases"].sum().reset_index()
    labels = ["Active cases","Recovered","Deaths"]
    values = data.loc[0, ["Active_cases","Recovered","Deaths"]]
    fig = px.pie(data, values=values, names=labels, color_discrete_sequence=['green','royalblue','darkblue'], hole=0.5)
    fig.update_layout(
        title='Total cases in ' + str(data['Country/Region'][0]) + ' ' + str(data["Confirmed"][0]),
    )
    fig.show()
    
def plot_all(data):
    total_cases_pie(data)
    province_cases_pie(data)
    bar_plot_province(data)
    scatter_evolution(data)
    line_plot_day_by_day(data)
    line_plot_month(data)


# In[13]:


data_germany = get_country(data, 'Germany')
data_italy = get_country(data, 'Italy')
data_china = get_country(data, 'China')


# In[14]:


plot_all(data_china)


# In[15]:


plot_all(data_italy)


# In[ ]:


plot_all(data_germany)


# In[ ]:





# In[ ]:





# In[ ]:




