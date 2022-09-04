#!/usr/bin/env python
# coding: utf-8

# # Sales Forecasting of a Superstore using Time Series Models

# # 1. Installing & Importing Packages

# The below comments should be installed only for the first time

# In[8]:


#!pip install pandas
#!pip install numpy
#!pip install matplotlib
#pip install plotly
#pip install prophet


# In[1]:


import datetime as dt
import pandas as pd
import numpy as np
import random as rd
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py

import pickle
import gc
import os

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


# In[4]:


from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.tsa.api import VAR


# In[9]:


from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from prophet import Prophet
from datetime import datetime


# In[10]:


import statsmodels.api as sm
import pandas as pd
import plotly.express as px


# # 2.  Reading the Input files

# In[11]:


# specify datatypes before loading the data will save you a ton of memory.
dtypes = {'id': np.uint32,
          'store_nbr': np.uint8, 
          'item_nbr': np.uint32, 
          'unit_sales': np.float32,
          'class': np.uint16,
          'dcoilwtico':np.float16,
          'transactions':np.uint16,
          'cluster': np.uint32,
         'onpromotion' : 'object'}


# In[12]:


df_train = pd.read_csv(r"C:\users\u2195687\Input\train.csv",parse_dates=['date'],dtype = {
                    'onpromotion': 'object'
               })


# In[13]:


df_holiday =  pd.read_csv(r"C:\users\u2195687\Input\holidays_events.csv",parse_dates=['date'])
df_items =  pd.read_csv(r"C:\users\u2195687\Input\items.csv")
df_oil =  pd.read_csv(r"C:\users\u2195687\Input\oil.csv",parse_dates=['date'])
df_stores = pd.read_csv(r"C:\users\u2195687\Input\stores.csv")
df_transactions=pd.read_csv(r"C:\users\u2195687\Input\transactions.csv",parse_dates =['date'])


# In[14]:


df_train['Year'] = pd.DatetimeIndex(df_train['date']).year #— to get the year from the date.
df_train['Month'] = pd.DatetimeIndex(df_train['date']).month # To get month from the date.
df_train['Day'] = pd.DatetimeIndex(df_train['date']).day # To get month from the date.


# In[15]:


df_train.shape


# ## 2.1) Samples of Each Dataframe

# In[14]:


df_train.head()


# In[15]:


df_train.describe()

i) Train dataset has the prediction variable (unit_sales) inaddition to the date, id ,store number, on_promotion and the item_number.
ii) The Dependent variable unit_sales can be a integer , float or negative number(return products).
iii) It is evident has onpromotion has Nan value. Let us deep dive in removing them in later part of this section.
iv) Store number denotes the store in which sales happened.
v) item number denotes the item which has the sales value.
vi) Unit sales denotes the number of products sold on that particular day.
vii) onpromotion denotes whether the product was on promotion or not.
viii) id is the unique field.
# In[16]:


df_transactions.head()


# i) Transactions dataset has the store number and the transactions details.<br>
# ii) We already know that Store number is also available in the train dataset as well.<br>
# iii) Transactions denotes the total number of transactions happened on the particular day with respect to each store.

# In[17]:


df_holiday.head()

Holiday dataset shows whether each day is a work day or holiday.
# In[18]:


df_items.head()


# i) Item dataset has the item number, family, class and perishable( likely to decay) <br>

# In[19]:


df_oil.head()

Oil dataset has oil price for each day.
# In[20]:


df_stores.head()

i) Stores datset has the store number , city and state where the store is located <br>
ii) Cluster is used to group stores
# Using the above details , we need to find the Unit_sales prediction for the future year <br>

# ## 2.2) More details of Dataframes

# In[21]:


df_train.info()


# In[22]:


df_train.shape


# It has 12 million records which is very huge. so, we can consider the sales only for some records.<br>
# After visualisation , we can decide the factors to select records.
# 

# In[23]:


df_holiday.info()


# In[24]:


df_items.info()


# In[25]:


df_oil.info()


# In[26]:


df_stores.info()


# In[27]:


df_transactions.info()


# Date fields are of datetime type as we specify the dtype while importing. we can also create the day , month and year feature from the date feature

# # 3) Visualisation of Each dataframes

# ## 3.1) Train

# In[28]:


plt.figure(figsize=(8,4))
df_train['Year'].value_counts(sort = False).plot.bar(y=['unit_sales'],color=['green'],edgecolor='blue',xlabel='Year',
                                                     ylabel='Unit Sales')


# 2016 have the largest number of sales. 

# In[29]:


plt.figure(figsize=(8,4))
df_train['Month'].value_counts(sort = False).plot.bar(y=['unit_sales'],color=['Purple'],edgecolor='red',
                                                     xlabel = ' Month ',ylabel='Unit Sales')


# The distributions are more or less equal with respect to the months

# In[30]:


plt.figure(figsize=(10,6))
df_train['Day'].value_counts(sort = False).plot.bar(y=['unit_sales'],color=['cyan'],edgecolor='green',
                                                   xlabel = ' Day ',ylabel='Unit Sales')

Sales is slightly less in the monh end and 1st day of the month.
# In[31]:


plt.figure(figsize=(20,8))
df_train['store_nbr'].sort_values().value_counts(sort = False).plot.bar(figsize=(9, 7))
#value_counts sorts the values by descending frequencies by default. Disable sorting using sort=False:
plt.xlabel("Store Number", labelpad=14)
plt.ylabel("Unit_Sales", labelpad=14)
plt.title("Stores vs Sales Comparsion", y=1.02);

There are 54 stores in total and the store_number 44 has the highest sales wheres as the store number 52 has the lowest sales
# In[32]:


df_train.boxplot(column =['unit_sales'], grid = False,color='red')

There are few outliers in our dependent variable.
# In[33]:


df_train['onpromotion'].value_counts(dropna=False).plot.bar(color='aqua',xlabel='onpromotion',ylabel='count')

Most of the products are not on promotion and there are some missing values in this field as well.
# ## 3.2) Transactions

# In[34]:


#transactions
# month over month sales
df_transactions['date']=pd.to_datetime(df_transactions['date'])
temp=df_transactions.groupby(['date']).aggregate({'store_nbr':'count','transactions':np.sum})
temp=temp.reset_index()
temp_2013=temp[temp['date'].dt.year==2013].reset_index(drop=True)
temp_2014=temp[temp['date'].dt.year==2014].reset_index(drop=True)
temp_2015=temp[temp['date'].dt.year==2015].reset_index(drop=True)
temp_2016=temp[temp['date'].dt.year==2016].reset_index(drop=True)
temp_2017=temp[temp['date'].dt.year==2017].reset_index(drop=True)

sns.set(style="whitegrid", color_codes=True)
plt.figure(figsize=(12,8))
plt.subplot(211)
plt.plot(temp_2013['date'],temp_2013.iloc[:,1],label="2013")
plt.plot(temp_2014['date'],temp_2014.iloc[:,1],label="2014")
plt.plot(temp_2015['date'],temp_2015.iloc[:,1],label="2015")
plt.plot(temp_2016['date'],temp_2016.iloc[:,1],label="2016")
plt.plot(temp_2017['date'],temp_2017.iloc[:,1],label="2017")
plt.ylabel('Number of stores open', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.title('Number of stores open', fontsize=15)
plt.xticks(rotation='vertical')
plt.legend(['2013', '2014', '2015', '2016'], loc='lower right')


# In[35]:


plt.figure(figsize=(12,6))
plt.plot(df_transactions.rolling(window=30,center=False).mean(),label='Rolling Mean');
plt.plot(df_transactions.rolling(window=30,center=False).std(),label='Rolling sd');
plt.legend();


# In[36]:


plt.style.use('seaborn-white')
plt.figure(figsize=(12,6))
plt.plot(df_transactions.date.values, df_transactions.transactions.values, color='darkblue')
plt.ylim(-50, 10000)
plt.title("Distribution of transactions per day from 2013 till 2017")
plt.ylabel('transactions per day', fontsize= 16)
plt.xlabel('Date', fontsize= 16)
plt.show()

The bigger yearly periodic spike in transactions seem to occur at the end of the year in December.
Perhaps this is due to some sort of Christmas sale/discount that Corporacion Favorita holds every December.
# ## 3.3) Items

# In[37]:


df_items['perishable'].value_counts(sort = False).plot.bar(color=['blue'],edgecolor='green',
                                                           xlabel='Perishable Items',ylabel='count')

Perishable items are less in quantity compared to the non-perishable ones.
# In[38]:


figsize=(24,40)
df_items['family'].value_counts(sort = False).plot.bar(color=['red'],edgecolor='purple',
                                                       xlabel='Item Family',ylabel='count')


# ## 3.4) Stores

# In[39]:


plt.style.use('seaborn-white')
nbr_cluster = df_stores.groupby(['store_nbr','cluster']).size()
nbr_cluster.unstack().iloc[neworder].plot(kind='bar',stacked=True, colormap= 'tab20', figsize=(16,5),  grid=False)
plt.title('Store numbers and the clusters they are assigned to', fontsize=14)
plt.ylabel('Clusters')
plt.xlabel('Store number')
plt.show()

Cluster  is the biggest one
# In[40]:


plt.style.use('seaborn-white')
city_cluster = df_stores.groupby(['city','type']).store_nbr.size()
city_cluster.unstack().plot(kind='bar',stacked=True, colormap= 'viridis', figsize=(14,7),  grid=False)
plt.title('Stacked Barplot of Store types opened for each city')
plt.ylabel('Count of stores for a particular city')
plt.show()

# Quito City has the largest number of stores. It consist of store type A,B,C and D.Gauyaquil is the next state and it has all
types of cities.
Guayaquil and Quito are two cities that stand out in terms of the range of retail kinds available. These are unsurprising given that Quito is Ecuador's capital and Guayaquil is the country's largest and most populated metropolis. As a result, one might expect Corporacion Favorita to target these major cities with the most diverse store types, as evidenced by the highest counts of store nbrs attributed to those two cities.
# ## 3.5) Oil

# In[41]:


#df_oil['Year'] = pd.DatetimeIndex(df_oil['date']).year #— to get the year from the date.
#df_oil['Month'] = pd.DatetimeIndex(df_oil['date']).month # To get month from the date.
#df_oil['Day'] = pd.DatetimeIndex(df_oil['date']).day # To get month from the date.


# In[513]:


trace = go.Scatter(
    name='Oil prices',
    x=df_oil['date'],
    y=df_oil['dcoilwtico'].dropna(),
    mode='lines',
    line=dict(color='rgb(25, 15, 200, 0.5)'),
    #fillcolor='rgba(68, 68, 68, 0.3)',
    fillcolor='rgba(0, 1, 219, 0.4)',
    fill='tonexty' )

data = [trace]

layout = go.Layout(
    yaxis=dict(title='Daily Oil price'),
    title='Daily oil prices from Jan 2013 till July 2017',
    showlegend = False)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='pandas-time-series-error-bars')


# ## 3.6) Holiday

# In[43]:


#df_holiday['Year'] = pd.DatetimeIndex(df_holiday['date']).year #— to get the year from the date.
#df_holiday['Month'] = pd.DatetimeIndex(df_holiday['date']).month # To get month from the date.
#df_holiday['Day'] = pd.DatetimeIndex(df_holiday['date']).day # To get month from the date.


# In[44]:


plt.style.use('seaborn-white')
holiday_local_type = df_holiday.groupby(['locale_name', 'type']).size()
holiday_local_type.unstack().plot(kind='bar',stacked=True, colormap= 'magma_r', figsize=(12,6),  grid=False)
plt.title('Stacked Barplot of locale name against event type')
plt.ylabel('Count of entries')
plt.show()

Ecuador has all tpes of holidays wheres the other places have either holiday or additional leave.
# # 4) Creating new Dataframe with filtered data

# Lets Extract 3 years data and keep it in a dataframe and save it in a csv

# In[16]:


year_nbr = [2015,2016,2017]
df_trainyr = df_train[df_train['Year'].isin(year_nbr)]
print(df_trainyr.shape)


# ####  Year Dataframe
125 million records has been reduced to 86 million
# In[17]:


print(df_train.shape)

We can see that the count has been nearly half after filtering out the data from the original one.Let us seggregat for
thr other datasets as well.
# ## 4.1) Filtering based on conditions
# 

# In[18]:


df_trainyr.head()

To minimise the number of rows , we are going to filter the dataframe based on some conditions; We are going to take only
the job family Eggs
# In[19]:


item_num = df_items.loc[df_items['family']=='EGGS']['item_nbr'].tolist()
print(item_num)


# ####  Egg Dataframe

# In[20]:


df_egg = df_trainyr[df_trainyr['item_nbr'].isin(item_num)]


# In[21]:


print(df_egg.shape)

86 million records has been reduced to 9 lakh records. But we will further filter to one item
# In[22]:


df_egg['item_nbr'].value_counts(sort = False).plot.bar(color=['violet'],edgecolor='green',
                                                           xlabel='Egg Items',ylabel='count',figsize = (20,10))
plt.xticks(rotation=45)
plt.tight_layout()

Item number 208384 is the item with more count and the item 1974848 is the item with least count
# In[23]:


df_eggmin = df_egg[df_egg['item_nbr']==1974848]
df_eggmax = df_egg[df_egg['item_nbr']==208384]


# In[24]:


print(df_eggmin.shape)
print(df_eggmax.shape)


# so let us proceed with the dataframe df_eggmin

# In[25]:


df_eggmax.info()


# In[26]:


df_eggmax.head()


# In[27]:


df_eggmax['unit_sales'].plot()


# ## 4.2) Merging with other Dataframes

# In[28]:


# Left Join - Train & Items 
train_subset = pd.merge(df_eggmax, df_items, on = 'item_nbr', how = 'left')
train_subset.head()


# In[29]:


# Left Join - merged & Store 
train_subset = pd.merge(train_subset, df_stores, on = 'store_nbr', how = 'left')
train_subset.head()


# In[30]:


# Left Join - merged & Oil
train_subset = pd.merge(train_subset, df_oil, on = 'date', how = 'left')
train_subset.head()


# In[31]:


# Left Join - Train & Holiday
train_subset = pd.merge(train_subset, df_holiday, on = 'date', how = 'left')
train_subset.head()


# In[32]:


train_subset.columns


# In[33]:


train_subset = train_subset.drop(['locale', 'locale_name','description','transferred'], axis=1)


# In[34]:


train_subset = train_subset.rename(columns={"type_y": "day_type", "type_x": "type","dcoilwtico":"oil_price"})


# In[35]:


train_subset.describe()


# In[36]:


train_subset.info()


# In[37]:


train_subset


# In[38]:


train_subsetbkp = train_subset.copy()


# In[39]:


train_subset = train_subset.set_index('date')


# ## 4.3) Resampling to Business days

# In[40]:


train_subset_d = train_subset.resample('B').mean()
train_subset_d.shape


# In[41]:


train_subset_with = train_subset.resample('D').mean()
train_subset_with.shape
#Daily - with holidays- facing lot of Nans


# In[42]:


train_subset_dbkp = train_subset_d.copy()
#Back up in another dataframe 


# In[43]:


train_subset_d


# # 5) Exploratory Data Analysis
Converting the normal time into a time series format is very important. Lets do some EDA on our input dataframe named train_subset
# Basic Analysis

# In[44]:


print("Rows :",train_subset_d.shape[0])
print("Columns :",train_subset_d.shape[1])
print("\n Features :",train_subset_d.columns.tolist())
print("\n Missing Values :",train_subset_d.isnull().sum())
print("\n Unique Values :",train_subset_d.nunique()) 


# Its evident that the column item_nbr ,family , class and perishable have only one value so we can drop them. Also, the missing values are available in the fields oil_price and day_type
# 

# Indexing - this allows Querying of the date easily and also data retrieval works fine here.At the same time, time series needs date as the index

# Querying the date (We can pass the indexed value in the loc function for effective querying)

# In[45]:


train_subset_d.loc['2017']


# In[46]:


train_subset_d.loc['2016':'2017']


# ## 5.1 Plotting the Variables

# In[47]:


train_subset_data = train_subset_d['unit_sales']
train_subset_data.head()


# In[48]:


train_subset_data.plot(grid=True,xlabel='Date',ylabel='Unit_Sales')


# Year 2015 Plot

# In[49]:


train_2015 = train_subset['2015']
sales_train_2015 = train_2015['unit_sales']
sales_train_2015.plot(grid=True)


# Year 2016 Plot

# In[50]:


train_2016 = train_subset['2016']
sales_train_2016 = train_2016['unit_sales']
sales_train_2016.plot(grid=True)


# Year 2017 Plot

# In[51]:


train_2017 = train_subset['2017']
sales_train_2017 = train_2017['unit_sales']
sales_train_2017.plot(grid=True)


# Histogram

# In[52]:


train_subset_d[['unit_sales']].hist()

Density Plot
# In[53]:


train_subset_d[['unit_sales']].plot(kind='density')


# Lag Plot:
#       They will show the relationship between the current period and the lag period. Linear shape suggest that a Autoregressive can be applied. This is used to check linearity , randomness , Outliers and check autocorrelation as well

# First Order

# In[54]:


pd.plotting.lag_plot(train_subset_d['oil_price'],lag=1)


# In[55]:


pd.plotting.lag_plot(train_subset_d['oil_price'],lag=30)


# In[56]:


pd.plotting.lag_plot(train_subset_d['oil_price'],lag=365)


# ## 5.2) Missing Values
Handling missing values in time series is different from the normal data because we cannot drop the values 
as the order of the time series will be disturbed and also we cannot impute the values with global mean and
media as it affects the trend and seasonality components
# In[57]:


# Train Dataset
train_subset_d.isnull().sum()


# Displaying the affected records

# In[58]:


train_subset_d.query('oil_price != oil_price')


# ### 5.2.1) Oil_price Filling

# In[59]:


pd.plotting.lag_plot(train_subset['oil_price'],lag=24)


# As per the above plot, there is a lineraity with the previous lag for the current lag in oil_price so we can fill the values #
# with forward fill(take prev values and fill next value). we can also apply bfill inplace of ffill to fill backwards
# (takes future values and take). but backward is not preferrable because furure value will not be available in some business
# scenario.

# In[60]:


train_subset_d['oil_price'] = train_subset_d['oil_price'].fillna(method='ffill')

we can also fill missing value by taking prev 2 or 3 values of mean and substitute(not global mean).
the code for that is train_subset['oil_price'] = train_subset['oil_price'].rolling(window=2,min_periods=1).mean()
# In[61]:


train_subset_d.isnull().sum()


# ## 5.3) Anomaly Detection
Identifies the data points or observations that deviate from the normal behavior of the dataset. it indicates critical incident
in business that needs to be checked.

Two Types:
*********
    Global outlier - this point is completely outside and far away.Naked eye will show this. may be its recorded by mistake as well or genuine as well. 
    Contexual outlier - time series data outlier ; this will be available in the time series data but may be in trend or seasonal patterns.
    
# In[62]:


# Outlier Analysis
fig, axs = plt.subplots(1, figsize = (5,5))
plt1 = sns.boxplot(train_subset_d['unit_sales'],color='red')
#plt2 = sns.boxplot(advertising['Newspaper'], ax = axs[1])
#plt3 = sns.boxplot(advertising['Radio'], ax = axs[2])
plt.tight_layout()


# In[63]:


unit_sales = train_subset_d["unit_sales"].values
#print(unit_sales.shape)


# In[64]:


plt.scatter(x = range(unit_sales.shape[0]), y = np.sort(unit_sales))


# In[65]:


train_subset_z = train_subset_d.copy()


# In[66]:


train_subset_z['z_score'] = (train_subset_z['unit_sales'] - train_subset_z['unit_sales'].mean())/train_subset_z['unit_sales'].std(ddof=0)


# In[67]:


train_subset_z['z_score'].plot()


# In[68]:


train_subset_z[train_subset_z['z_score'] > 4]

solstice celebration ecuador festival on 20th june 2017
# In[69]:


#exclude the rows with z score more than 4
from scipy.stats import zscore
train_subset_z = train_subset_z[(np.abs(zscore(train_subset_z['z_score'])) < 2)]


# In[70]:


unit_sales_z = train_subset_z['z_score'].values


# In[71]:


plt.scatter(x = range(unit_sales_z.shape[0]), y = np.sort(unit_sales_z))


# In[306]:


train_subset_z.shape


# In[72]:


# Outlier Analysis
fig, axs = plt.subplots(1, figsize = (5,5))
plt1 = sns.boxplot(train_subset_z['unit_sales'],color='red')
plt.tight_layout()


# ### Standardisation

# In[73]:


train_subset_s = train_subset_z.copy()


# In[74]:


# histogram plot
plt.hist(train_subset_s['unit_sales'])
plt.show()


# In[75]:


train_subset_s[['unit_sales']].plot(kind='density')


# In[76]:


train_subset_ss = train_subset_s.copy()


# In[77]:


from sklearn.preprocessing import StandardScaler
scale= StandardScaler()
scaled_data = scale.fit_transform(train_subset_ss) 
print(scaled_data)


# In[78]:


train_subset_ss.drop(columns=['id','z_score'],inplace=True)


# In[79]:


train_subset_ss[['unit_sales']].plot(kind='density')


# ## 5.5) Correlation

# In[80]:


train_corr = train_subset_d[['unit_sales','cluster','oil_price']].corr(method='pearson')
train_corr
g = sns.heatmap(train_corr,vmax=6,center=0,
               square = True,linewidths=5,cbar_kws = {'shrink':.5},annot=True,fmt='.2f',cmap='twilight')
g.figure.set_size_inches(10,10)


# In[81]:


train_corr = train_subset_ss[['unit_sales','cluster','oil_price']].corr(method='pearson')
train_corr

Cluster and oil_price is the only pair with a correlation value of 0.20. others are having less than 1
# # 6) TimeSeries Components - train_subset_d
Time series has 4 main components-Trend,seasonality, stationarity and residuals
# ## 6.1) Decomposition
Decomposition(statistical technique) means removing the trend , seasonlity , Noise and cyclic behavior from the Time series data. They are the 
time series components. We can have only the stationarity component in the data to proceed he forecasting of the data.
Now, let us see whether we have any of those components and if we have them , we can remove them by using the appropriate
techniques inorder to do the better forecasting of the data.

Note : After removing the components , whatever is remaining is called as the residuals.

Decomposition should be Additive if the data is stationary and it should be multiplicative if the data is not stationary.
# In[82]:


df_decomp = train_subset_s.copy()


# In[83]:


df_decomp


# In[84]:


import statsmodels.api as sm
res = sm.tsa.seasonal_decompose(df_decomp['unit_sales'],
                                model='additive',period=2,extrapolate_trend='freq')

resplot = res.plot()

There is some upward trend and no seasonality value and some residuals:
HAndling Trend:
    We should divide it by observed value . that is called detrend in case of multiplicative value. pd.Dataframe(res.observed/res.trend).plot() - this graph will remove trend.
    We can subtract the actual value with observed value in case of additive model.
  
 Note : additive model is applied for linear model and multiplicative for the non-linear data.

# In[85]:


res.trend.plot()


# In[86]:


res.seasonal.plot()


# In[87]:


res.resid.plot()


# In[88]:


res.observed #ACTUAL UNIT_SALES Value


# In[89]:


print(res.trend)

NAn states that t-1 and t-2 analysis 
# In[90]:


print(res.seasonal)


# In[91]:


res.trend[1] + res.seasonal[1] + res.resid[1]

This answer is same as observed[1]
# ### 6.1.a) Detrend the data

# In[92]:


pd.DataFrame(res.observed-res.trend).plot()


# In[93]:


detrend = pd.DataFrame(res.observed/res.trend)


# In[94]:


#detrend.rename(columns = {0:'detrend_sales'},inplace=True)
detrend


# In[95]:


detrend.plot()


# In[96]:


train_subset_sbkp = train_subset_s.copy()


# In[97]:


train_subset_s = pd.concat([train_subset_s,detrend[0]],axis=1)


# In[98]:


train_subset_s = train_subset_s.rename(columns={0:'unit_sales_detrend'})


# In[99]:


train_subset_s[['unit_sales','unit_sales_detrend']].plot(figsize=(8,7))


# In[100]:


train_subset_s['unit_sales_detrend'].plot()


# In[101]:


#train_subset_s['unit_sales_detrend'] = train_subset_s['unit_sales_detrend'].fillna(method='bfill')


# In[102]:


train_subset_s.isnull().sum()


# ## 6.2) Handling Stationarity
NON-Stationary - Go for ARIMA model (I component makes TS stationary- differencing (Typical value(d) is 2))
Stationary - ARMA Model (no trend and seasonlity)
If the TS model has seasonality then, we can go for SARIMA model.
# ### 6.2.1) ADF
In ADF, null hypothesis states that TS is not Stationary and it possesses a unit-root
         Alternate Hypothesis is Ts is stationary
    p value less than 0.05 means, accept the alternate
# In[103]:


train_subset_s.isnull().sum()


# In[104]:


from statsmodels.tsa.stattools import adfuller
result = adfuller(train_subset_s['unit_sales_detrend'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

if result[0] < result[4]["5%"]:
    print ("Reject Ho - Time Series is Stationary")
else:
    print ("Failed to Reject Ho - Time Series is Non-Stationary")


# ### 6.2.2) KPSS Test
In KPSS, null hypothesis states that TS is Stationary 
         Alternat Hypothesis is Ts is not-stationary
    p value less than 0.05 means, accept the alternate
# In[105]:


from statsmodels.tsa.stattools import kpss
tstest = kpss(train_subset_s['unit_sales_detrend'],'ct')


# In[106]:


tstest


# ## 6.3) Handling Residuals by smoothening

# In[107]:


train_subset_s


# In[108]:


train_split = train_subset_s.loc['2015-01-01':'2017-08-08']


# In[109]:


test_split = train_subset_s.loc['2017-08-09':'2017-08-15']


# In[110]:


train_split


# In[111]:


test_split


# ### 6.1.1) Simple Moving Average

# In[112]:


df_sma = train_subset_s.copy()


# In[113]:


train_subset_s.drop


# In[114]:


#df_sma = df_sma[['unit_sales_detrend']]


# In[115]:


df_sma['ma_rolling_3'] = df_sma['unit_sales_detrend'].rolling(window=3).mean().shift(1)


# In[116]:


df_sma[['unit_sales_detrend','ma_rolling_3']].plot(color=['red','green'])


# ### 6.1.2 Weighted Moving average

# In[117]:


def wma(weights):
    def calc(x):
        return (weights*x).mean()
    return calc


# In[118]:


df_sma['wma_rolling_3'] = df_sma['unit_sales_detrend'].rolling(window=3).apply(wma(np.array([0.5,1,1.5]))).shift(1)


# In[119]:


df_sma[['unit_sales_detrend','wma_rolling_3']].plot(color=['blue','green'])


# ### 6.1.3) Exponential Weighted  Moving Average

# In[120]:


df_sma['ewm_window_3'] = df_sma['unit_sales_detrend'].ewm(span=3,adjust=False,min_periods=0).mean().shift(1)


# In[121]:


df_sma[['unit_sales_detrend','ewm_window_3']].plot(color=['green','red'])


# In[122]:


def rmse(x,y):
    return ((df_sma[x]-df_sma[y])**2).mean()**0.5
print('Moving Average')
print(rmse('unit_sales_detrend','ma_rolling_3'))
print('Weighted Moving Average')
print(rmse('unit_sales_detrend','wma_rolling_3'))
print('Exponential Weighted Moving Average')
print(rmse('unit_sales_detrend','ewm_window_3'))


# Moving average gives the best results

# In[123]:


df_sma[['unit_sales_detrend','ewm_window_3','wma_rolling_3','ma_rolling_3']].plot(color=['violet','red','green','blue'])


# # PROPHET

# In[183]:


#!pip install pystan fbprophet


# In[184]:


from prophet import Prophet
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pandas as pd
import plotly.express as px


# In[185]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pandas as pd
import plotly.express as px

mpl.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['axes.grid'] = False


# In[186]:


#train_subset_d


# In[187]:


df_pro = train_subset_d.copy()


# In[188]:


df_pro


# In[189]:


df_pro_a =df_pro.reset_index()[['date','unit_sales']].rename({'date':'ds','unit_sales':'y'}, axis='columns')


# In[190]:


df_pro_b = df_pro_a.copy()


# In[191]:


train_pro=df_pro_a[(df_pro_a['ds'] >= '2015-01-01') & (df_pro_a['ds'] <= '2017-08-09')]
test_pro=df_pro_a[(df_pro_a['ds'] > '2017-08-09')]


# ## Prophet1

# In[192]:


m = Prophet(interval_width=0.95)
m.fit(train_pro)

confidence interval by default is 80% so we are passing 95%
# In[193]:


#m.params


# In[194]:


future = m.make_future_dataframe(periods=369,freq='B') # 4 is test data shape
forecast1 = m.predict(future)
forecast1.head()


# In[195]:


prophet1 = forecast1['yhat']


# In[196]:


pd.concat([df_pro_a.set_index('ds')['y'],forecast1.set_index('ds')['yhat']],axis=1).plot()


# In[197]:


fig1 = m.plot(forecast1)


# In[198]:


#fig2 = m.plot_components(forecast)


# In[200]:


from prophet.plot import add_changepoints_to_plot
fig = m.plot(forecast1)
a = add_changepoints_to_plot(fig.gca(), m, forecast1)


# In[201]:


se = np.square(forecast1.loc[:, 'yhat'] - df_pro_a['y'])
mse1 = np.mean(se)
rmse1 = np.sqrt(mse1)
print(mse1)
print(rmse1)


# In[411]:


from sklearn.metrics import r2_score


# In[415]:


df_pro_a


# In[419]:


print('mse value is',((forecast1.loc[:, 'yhat'] - df_pro_a['y'])**2).mean()**0.5)
print('rmse value is',((forecast1.loc[:, 'yhat'] - df_pro_a['y']) ** 2).mean())
print('mape value is',np.mean(np.abs(((forecast1.loc[:, 'yhat'] - df_pro_a['y'])) / forecast1.loc[:, 'yhat'])) * 100)
print('r_squared is',r2_score(forecast1.loc[:683, 'yhat'], df_pro_a['y']))


# ## Prophet 2 - seasonality included

# In[202]:


pro_change= Prophet(interval_width=0.95,changepoint_range = 0.95, daily_seasonality=True)
forecast2 = pro_change.fit(train_pro).predict(future)
fig4 = pro_change.plot(forecast2);
b = add_changepoints_to_plot(fig4.gca(), pro_change, forecast2)


# In[203]:


ae = np.square(forecast2.loc[:, 'yhat'] - df_pro_b['y'])
mse2 = np.mean(ae)
rmse2 = np.sqrt(mse2)
print(mse2)
print(rmse2)


# In[420]:


print('mse value is',((forecast2.loc[:, 'yhat'] - df_pro_b['y'])**2).mean()**0.5)
print('rmse value is',((forecast2.loc[:, 'yhat'] - df_pro_b['y']) ** 2).mean())
print('mape value is',np.mean(np.abs(((forecast2.loc[:, 'yhat'] - df_pro_b['y'])) / forecast2.loc[:, 'yhat'])) * 100)
print('r_squared is',r2_score(forecast2.loc[:683, 'yhat'], df_pro_b['y']))


# ## Prophet 3 - seasonality included and holiday included

# In[204]:


holi = df_holiday.copy()


# In[205]:


holi = holi[['date','type']]


# In[206]:


holi = holi.reset_index()[['date','type']].rename({'date':'ds','type':'holiday'}, axis='columns')


# In[207]:


holi


# In[208]:


hol_change= Prophet(changepoint_range = 0.90, daily_seasonality=True,holidays=holi)
forecast3 = hol_change.fit(train_pro).predict(future)
fig5 = pro_change.plot(forecast3);
c = add_changepoints_to_plot(fig5.gca(), pro_change, forecast3)


# In[209]:


be = np.square(forecast3.loc[:, 'yhat'] - df_pro_b['y'])
mse3 = np.mean(be)
rmse3 = np.sqrt(mse3)
print(mse3)
print(rmse3)


# In[421]:


print('mse value is',((forecast3.loc[:, 'yhat'] - df_pro_b['y'])**2).mean()**0.5)
print('rmse value is',((forecast3.loc[:, 'yhat'] - df_pro_b['y']) ** 2).mean())
print('mape value is',np.mean(np.abs(((forecast3.loc[:, 'yhat'] - df_pro_b['y'])) / forecast2.loc[:, 'yhat'])) * 100)
print('r_squared is',r2_score(forecast3.loc[:683, 'yhat'], df_pro_b['y']))


# # ARIMA & SARIMA

# Auto correlation (ACF and PACF plots)

# In[427]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[468]:


df_arma = train_subset.resample('B').mean()


# In[469]:


fig = plt.figure(figsize=(15,8))
ax1= fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_arma['unit_sales'].iloc[13:],lags=40,ax=ax1)
ax2= fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_arma['unit_sales'].iloc[13:],lags=40,ax=ax2)


# In[470]:


type(df_arma)


# In[471]:


df = df_arma['unit_sales'].squeeze()


# In[472]:


type(df)


# In[473]:


X = df
train_size = int(len(X) * 0.75)
trainset, testset= X[0:train_size], X[train_size:]


# In[474]:


def performance(y_true, y_pred): 
    mse = ((y_pred - y_true) ** 2).mean()
    mape= np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r_squared = r2_score(y_true,y_pred)
    performance_data= {'MSE':round(mse, 2),
                      'RMSE':round(np.sqrt(mse), 2),
                       'MAPE':round(mape, 2),
                       'RSquared':round(r_squared, 2)
                                             }
    return performance_data

def performance2(y_true, y_pred): 
    #y_true, y_pred = np.array(y_true), np.array(y_pred)
    mse = ((y_pred - y_true) ** 2).mean()
    mape= np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return( print(' The MSE of forecasts is {}'.format(round(mse, 2))+
                  '\n The RMSE of forecasts is {}'.format(round(np.sqrt(mse), 2))+
                  '\n The MAPE of forecasts is {}'.format(round(mape, 2))))


# ## ARIMA

# In[490]:


df.values


# In[491]:


import warnings
from pandas import Series
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# evaluate an ARIMA model
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.75)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error

# evaluate the combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))


# In[492]:


# evaluate parameters
p_values = [0, 1, 1, 1,1]
d_values = range(1, 2)
q_values = range(1, 2)
warnings.filterwarnings("ignore")
evaluate_models(df.values, p_values, d_values, q_values)


# In[493]:


trainset.tail(12)


# In[494]:


from statsmodels.tsa.arima.model import ARIMA
model_arima = ARIMA(trainset, order = (10,0,8))
model_arima_fit = model_arima.fit()
arima_predict = model_arima_fit.predict(start=pd.to_datetime('2016-11-21'), end=pd.to_datetime('2017-08-25')
                                           ,dynamic=False)


# In[495]:


# One step ahead forecast
#observed plot
ax = df.plot(label='Observed',color='#2574BF')
#predicted plot
#rcParams['figure.figsize'] = 14, 7
arima_predict.plot(ax=ax, label='ARIMA (10,0,8) Prediction', alpha= 0.7, color='red') 
plt.title('ARIMA  sales forecasting')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()


# In[496]:


arima_results= performance(df[-200:],arima_predict)
arima_results


# In[497]:


# One step ahead forecast
#observed plot
ax = df.plot(label='Observed',color='#2574BF')
#predicted plot
#rcParams['figure.figsize'] = 14, 7
#arma_predict.plot(ax=ax,label='ARMA (1,1) Prediction', linestyle= '-.', alpha= 0.7, color='r')
arima_predict.plot(ax=ax, label='ARIMA Prediction', linestyle= "--" ,alpha= 0.7, color='g')
plt.title('ARMA(1,1) and ARIMA(10,0,8) sales forecasting comparison')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()


# In[498]:


arima_predict


# In[499]:


df[-12:]


# In[500]:


get_ipython().system('pip install pmdarima ')


# In[501]:


## Find optimal order
import pmdarima as pm
model_1 = pm.auto_arima(trainset,seasonal=True, m=12,d=0, D=1, max_p=2, max_q=2,
                       trace=True,error_action='ignore',suppress_warnings=True) 

# Print model summary
print(model_1.summary())

#best model is Fit ARIMA: ARIMA(2,0,2) and seasonal_order=(2, 1, 1, 12); AIC=2639,


# ## SARIMA

# In[502]:


#fitting model
sarima_model_1 = sm.tsa.statespace.SARIMAX(trainset,
                                order=(2, 0, 2),
                                seasonal_order=(2, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
sarima_fit_1 = sarima_model_1.fit()
print(sarima_fit_1.summary().tables[1])


# In[503]:


sarima_fit_1.plot_diagnostics(figsize=(16, 8))
plt.show()


# In[504]:


df.tail


# In[505]:


# One step ahead forecast
sarima_predict_1 = sarima_fit_1.get_prediction(start=pd.to_datetime('2016-10-31'), end=pd.to_datetime('2017-08-15')
                                           ,dynamic=False)
sarima_predict_conf_1 = sarima_predict_1.conf_int()
#observed plot
ax = df.plot(label='Observed',color='#2574BF')
#predicted plot
#rcParams['figure.figsize'] = 14, 7
sarima_predict_1.predicted_mean.plot(ax=ax, label='SARIMA (2, 0, 2)x(2, 1, 1, 12) Prediction', alpha= 0.7, color='red') 
ax.fill_between(sarima_predict_conf_1.index,
                #lower sales
                sarima_predict_conf_1.iloc[:, 0],
                #upper sales
                sarima_predict_conf_1.iloc[:, 1], color='k', alpha=0.1)
plt.title('Seasonal ARIMA  sales forecasting')
plt.xlabel('Date')
plt.ylabel('Unit Sales')
plt.legend()
plt.show()


# In[506]:


trainset.tail(12)


# In[507]:


sarima_results=performance(df[-207:],sarima_predict_1.predicted_mean)
sarima_results


# # Deep Learning

# In[124]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from statsmodels.graphics import tsaplots
import statsmodels.api as sm
from pylab import rcParams
import itertools
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from pandas import Series

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout,Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from statsmodels.tools.eval_measures import rmse

import warnings
warnings.filterwarnings("ignore")


# In[125]:


#pip install statsmodels
get_ipython().system('pip install sklearn')


# In[2]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install numpy')
get_ipython().system('pip install matplotlib')


# In[6]:


pip install seaborn


# In[7]:


pip install itertools


# In[142]:


df_lstm = train_subset.resample('B').mean()


# In[143]:


df1 = df_lstm['unit_sales'].squeeze()


# In[144]:


train1, test1 = np.array(df1[:-12]), np.array(df1[-12:])
train1= train1.reshape(-1,1)
test1= test1.reshape(-1,1)


# In[ ]:


#Scale train and test data to [-1, 1]
scaler = MinMaxScaler()
scaler.fit(train1)
train1 = scaler.transform(train1)
test1 = scaler.transform(test1)


# In[149]:


n_input = 671
# univariate
n_features = 1
#TimeseriesGenerator automatically transform a univariate time series dataset into a supervised learning problem.
generator = TimeseriesGenerator(train1, train1, length=n_input, batch_size=10)


# ## Vanilla LSTM

# In[215]:


######
#set the counter to repeat
n=2
store= np.zeros((671,n))
for i in range(n):
    model_vanilla = Sequential()
    model_vanilla.add(LSTM(50, activation='relu', input_shape=(671, 1)))
    #Add layer
    model_vanilla.add(Dense(100, activation='relu'))
    model_vanilla.add(Dense(100, activation='relu'))
    #Output
    model_vanilla.add(Dense(1))
    model_vanilla.compile(optimizer='adam', loss='mse')
    # 22
    model_vanilla.fit_generator(generator,epochs=10)
    
    pred_list = []

    batch = train1[-n_input:].reshape((1, n_input, n_features))

    for j in range(n_input):   
        pred_list.append(model_vanilla.predict(batch)[0]) 
        batch = np.append(batch[:,1:,:],[[pred_list[j]]],axis=1)

    df_predict_vanilla = pd.DataFrame(scaler.inverse_transform(pred_list),
                              index=df1[-n_input:].index, columns=['Prediction'])

    
    store[:,i]=df_predict_vanilla['Prediction']
print(store)


# In[216]:


final_vanilla= np.zeros((store.shape[0],1))

#final_vanilla= np.zeros((store.shape[0],1))
for i in range(store.shape[0]):
    
    final_vanilla[i]=np.mean(store[i,:])
final_vanilla=final_vanilla.reshape((671,))


# In[508]:


# report performance
rcParams['figure.figsize'] = 6, 4
# line plot of observed vs predicted
plt.plot(df1.index,df1,label="Observed",color='#2574BF')
plt.plot(df1[13:].index,final_vanilla,label="Vanilla LSTM Prediction")
plt.title('Vanilla LSTM  sales forecasting')
plt.xlabel('Date')
plt.ylabel(' Sales')
plt.legend()
plt.show()


# In[ ]:


def performance(y_true, y_pred): 
    mse = ((y_pred - y_true) ** 2).mean()
    mape= np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r_squared = r2_score(y_true,y_pred)
    performance_data= {'MSE':round(mse, 2),
                      'RMSE':round(np.sqrt(mse), 2),
                       'MAPE':round(mape, 2),
                       'RSquared':round(r_squared, 2)
                                             }
    return performance_data

def performance2(y_true, y_pred): 
    #y_true, y_pred = np.array(y_true), np.array(y_pred)
    mse = ((y_pred - y_true) ** 2).mean()
    mape= np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return( print(' The MSE of forecasts is {}'.format(round(mse, 2))+
                  '\n The RMSE of forecasts is {}'.format(round(np.sqrt(mse), 2))+
                  '\n The MAPE of forecasts is {}'.format(round(mape, 2))))


# In[456]:


vanilla_lstm= performance(df1[-671:],final_vanilla)
vanilla_lstm


# ## LSTM 4-LAYER

# In[330]:


df_input_lstm = train_subset[['unit_sales']]


# In[331]:


df_input_lstm 


# In[332]:


from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf


# In[333]:


scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_input_lstm)


# In[334]:


features=data_scaled
target=data_scaled[:,0]


# In[335]:


TimeseriesGenerator(features, target, length=2, sampling_rate=1, batch_size=1)[0]


# In[336]:


x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=123, shuffle = False)
x_train.shape


# In[337]:


win_length=2
batch_size=10
num_features=1
train_generator = TimeseriesGenerator(x_train, y_train, length=win_length, sampling_rate=1, batch_size=batch_size)
test_generator = TimeseriesGenerator(x_test, y_test, length=win_length, sampling_rate=1, batch_size=batch_size)


# In[338]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(128, input_shape= (win_length, num_features), return_sequences=True))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5)) 
model.add(tf.keras.layers.LSTM(128, input_shape= (win_length, num_features), return_sequences=True))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5)) 
model.add(tf.keras.layers.LSTM(128, return_sequences=True))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5)) 
model.add(tf.keras.layers.Dropout(0.3)) 
model.add(tf.keras.layers.LSTM(64, return_sequences=False))
model.add(tf.keras.layers.Dropout(0.3)) 
model.add(tf.keras.layers.Dense(1))


# In[339]:


model.summary()


# In[340]:


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=2,
                                                    mode='min')

model.compile(loss=tf.losses.MeanSquaredError(),
              optimizer=tf.optimizers.Adam(),
              metrics=[tf.metrics.MeanAbsoluteError()])

history = model.fit_generator(train_generator, epochs=10,
                    validation_data=test_generator,
                    shuffle=False,
                    callbacks=[early_stopping])


# In[341]:


model.evaluate_generator(test_generator, verbose=0) 


# In[342]:


predictions=model.predict_generator(test_generator)


# In[343]:


x_test[:,1:][win_length:]


# In[344]:


df_pred_lstm =pd.concat([pd.DataFrame(predictions), pd.DataFrame(x_test[:,1:][win_length:])],axis=1)


# In[345]:


rev_trans=scaler.inverse_transform(df_pred_lstm)


# In[346]:


df_final=df_input_lstm[predictions.shape[0]*-1:]


# In[347]:


df_final['App_Pred']=rev_trans[:,0]


# In[512]:


rcParams['figure.figsize'] = 6, 4
df_final[['unit_sales','App_Pred']].plot()


# In[460]:


def rmse(x,y):
    return ((df_final[x]-df_final[y])**2).mean()**0.5
def mse(x,y):
    return ((df_final[x]-df_final[y]) ** 2).mean()
def mape(x,y):
     return np.mean(np.abs(((df_final[x]-df_final[y])) / df_final[x])) * 100
def r_sq(x,y):
    return r2_score(df_final[x],df_final[y
                                       ])


# In[461]:


aa='unit_sales'
bb='App_Pred'

print('rmse is ',rmse(aa,bb))
print('mse is',mse(aa,bb))
print('mape is',mape(aa,bb))
print('R Squared is',r_sq(aa,bb))

