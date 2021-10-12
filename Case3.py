#!/usr/bin/env python
# coding: utf-8

# # DATASET LAADPAALDATA

# ### IMPORTING AND CLEANING DATA

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


# In[2]:


laadpaaldata = pd.read_csv('laadpaaldata.csv')
pd.set_option('display.max_columns', None)


# In[3]:


laadpaaldata.head()


# In[4]:


laadpaaldata.info()


# In[5]:


laadpaaldata.describe()


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


plt.hist(laadpaaldata['ConnectedTime'], bins = 50)


# In[8]:


plt.hist(laadpaaldata['TotalEnergy'], bins=50)


# In[9]:


plt.hist(laadpaaldata['ChargeTime'], bins=50)


# In[10]:


plt.hist(laadpaaldata['MaxPower'], bins=50)


# In[11]:


laadpaaldata1 = laadpaaldata[laadpaaldata['ChargeTime'] >= 0 ]


# In[12]:


laadpaaldata1.describe()


# In[13]:


plt.hist(laadpaaldata1['ChargeTime'], bins=50)


# In[14]:


laadpaaldata1 = laadpaaldata1[laadpaaldata1['ConnectedTime']<=50]


# In[15]:


laadpaaldata1 = laadpaaldata1[laadpaaldata1['ChargeTime']<=15]


# In[16]:


laadpaaldata1['Percentage opladen'] = laadpaaldata1['ChargeTime'] / laadpaaldata1['ConnectedTime']


# In[17]:


laadpaaldata1.head()


# In[18]:


laadpaaldata1.info()


# In[19]:


plt.scatter(y=laadpaaldata1['TotalEnergy'], x=laadpaaldata1['ChargeTime'])


# In[ ]:





# In[ ]:





# DATASET laadpaaldata1 is klaar voor gebruik, volledig clean

# In[20]:


import plotly.express as px
import plotly.graph_objects as go


# In[93]:


medianConTime = laadpaaldata1['ConnectedTime'].median()
medianChaTime = laadpaaldata1['ChargeTime'].median()
meanConTime = laadpaaldata1['ConnectedTime'].mean()
meanChaTime = laadpaaldata1['ChargeTime'].mean()


# In[94]:


fig = go.Figure()
for col in ['ConnectedTime', 'ChargeTime']:
    fig.add_trace(go.Histogram(x=laadpaaldata1[col]))


dropdown_buttons = [
    {'label': 'Connected Time', 'method': 'update',
    'args': [{'visible': [True, False]},
            {'title': 'Connected Time'}]},
    {'label': 'Charge Time', 'method': 'update',
    'args': [{'visible': [False, True]},
            {'title': 'Charge Time'}]}]

float_annotation = {'xref': 'paper', 'yref': 'paper',
                    'x': 0.5, 'y': 0.8,'showarrow': False,
                    'text': f'Median of Connecting Time is {medianConTime} hours',
                    'font' : {'size': 15,'color': 'black'}
                    }


fig.data[1].visible=False
fig.update_layout({'updatemenus':[{'type': "dropdown",'x': 1.3,'y': 0.5,'showactive': True,'active': 0,'buttons': dropdown_buttons}]})
fig.update_layout(xaxis_title='Time in hour',
                  yaxis_title="Number of observations")
fig.update_layout({'annotations': [float_annotation]})
fig.show()


# In[109]:


laadpaaldata1 = laadpaaldata1[laadpaaldata1['ConnectedTime']<=20]
laadpaaldata1 = laadpaaldata1[laadpaaldata1['ChargeTime']<=6]


# In[114]:


import plotly.figure_factory as ff

group_1 = laadpaaldata1['ConnectedTime']
group_2 = laadpaaldata1['ChargeTime']

hist_data = [group_1, group_2]
group_labels = ['Connected Time', 'Charge Time']

fig = ff.create_distplot(hist_data, group_labels, colors=['blue','red'])
fig.update_layout({'title': {'text':'Distplot of Charge and Connecting Time'},
                   'xaxis': {'title': {'text':'Time in hours'}}})
fig.show()


# In[22]:


laadpaaldata1['ConnectedTime'].mean()


# In[23]:


laadpaaldata1['ChargeTime'].mean()


# In[24]:


laadpaaldata1['Percentage opladen'].mean()


# - eventueel kleur aanpassen
# - toevoegen van gemiddelde, mediaan en kansdichtheid

# In[25]:


fig = go.Figure()
for col in ['ConnectedTime', 'ChargeTime']:
    fig.add_trace(go.Scatter(x=laadpaaldata1[col], y=laadpaaldata1['TotalEnergy'], mode='markers'))

my_buttons = [{'label': 'Connected Time', 'method': 'update',
    'args': [{'visible': [True, False, False]},
            {'title': 'Connected Time'}]},
    {'label': 'Charge Time', 'method': 'update',
    'args': [{'visible': [False, True, False]},
            {'title': 'Charge Time'}]},
    {'label': 'Combined', 'method': 'update',
    'args': [{'visible': [True, True, True]},
            {'title': 'Combined'}]}]

fig.update_layout({
    'updatemenus': [{
      'type':'buttons','direction': 'down',
      'x': 1.3,'y': 0.5,
      'showactive': True, 'active': 0,
      'buttons': my_buttons}]})    
fig.update_layout(xaxis_title='Time in hour',
                  yaxis_title="Total energy used in Wh")
fig.data[1].visible=False
fig.show()    


# - Legenda voor combined

# # DATASET API OPENCHARGEMAP

# ### IMPORTING AND CLEANING DATA

# In[147]:


import pandas as pd
import requests
import csv
import json
url = 'https://api.openchargemap.io/v3/poi/?output=json&countrycode=NL&maxresults=1000&key=74e5c90d-3e4f-4bbe-b506-233af06f55ca'
r = requests.get(url)
datatxt = r.text
datajson = json.loads(datatxt)
print(datajson)


# In[148]:


df = pd.json_normalize(datajson)
df.head()


# In[149]:


df['AddressInfo.Country.Title'].unique()


# In[150]:


pd.set_option('max_columns', None)


# In[151]:


labels = ['UserComments', 'PercentageSimilarity','MediaItems','IsRecentlyVerified','DateLastVerified',
         'UUID','ParentChargePointID','DataProviderID','DataProvidersReference','OperatorID',
         'OperatorsReference','UsageTypeID','GeneralComments','DatePlanned','DateLastConfirmed','MetadataValues',
         'SubmissionStatusTypeID','DataProvider.WebsiteURL','DataProvider.Comments','DataProvider.DataProviderStatusType.IsProviderEnabled',
         'DataProvider.DataProviderStatusType.ID','DataProvider.DataProviderStatusType.Title',
         'DataProvider.IsRestrictedEdit','DataProvider.IsOpenDataLicensed','DataProvider.IsApprovedImport',
         'DataProvider.License','DataProvider.DateLastImported','DataProvider.ID','DataProvider.Title',
         'OperatorInfo.Comments','OperatorInfo.PhonePrimaryContact','OperatorInfo.PhoneSecondaryContact',
         'OperatorInfo.IsPrivateIndividual','OperatorInfo.AddressInfo','OperatorInfo.BookingURL',
         'OperatorInfo.ContactEmail','OperatorInfo.FaultReportEmail','OperatorInfo.IsRestrictedEdit',
         'UsageType','OperatorInfo','AddressInfo.DistanceUnit','AddressInfo.Distance','AddressInfo.AccessComments',
         'AddressInfo.ContactEmail','AddressInfo.ContactTelephone2','AddressInfo.ContactTelephone1',
         'OperatorInfo.WebsiteURL','OperatorInfo.ID','UsageType.ID','StatusType.IsUserSelectable',
         'StatusType.ID','SubmissionStatus.IsLive','SubmissionStatus.ID','SubmissionStatus.Title',
         'AddressInfo.CountryID','AddressInfo.Country.ContinentCode','AddressInfo.Country.ID',
         'AddressInfo.Country.ISOCode','AddressInfo.RelatedURL','Connections']
df = df.drop(columns=labels)


# In[157]:


df.head(30)


# In[154]:


df['NumberOfPoints'].sum()


# In[156]:


fig = go.Figure()

fig.add_trace(go.Histogram(x=df['AddressInfo.StateOrProvince']))

fig.update_layout(xaxis_title='Time in hour',
                  yaxis_title="Number of observations")

fig.show()


# In[144]:


df['LAT'] = df['AddressInfo.Latitude']
df['LNG'] = df['AddressInfo.Longitude']


# In[115]:


import folium


# In[145]:


m = folium.Map(location = [52.0893191, 5.1101691], 
               zoom_start = 7)

for row in df.iterrows():
    row_values = row[1]
    location = [row_values['LAT'], row_values['LNG']]
    marker = folium.Marker(location = location,
                         popup = row_values['AddressInfo.AddressLine1'])
    marker.add_to(m)

m


# In[161]:


import geopandas as gpd
provincies = gpd.read_file('provinces.geojson')


# In[163]:


provincies.head(20)


# In[164]:


m = folium.Map(location = [52.0893191, 5.1101691], 
               zoom_start = 7)
m.choropleth(
    geo_data = provincies,
    name = 'geometry',
    data = provincies)


# # DATASET ELEKTRISCHE VOERTUIGEN

# ### IMPORTING AND CLEANING DATA

# In[33]:


Elektrisch = pd.read_csv('Elektrische_voertuigen.csv')


# In[34]:


Elektrisch.head()


# In[35]:


Elektrisch.info()


# In[36]:


Elektrisch.describe()


# In[37]:


plt.hist(Elektrisch['Massa rijklaar'], bins = 40)


# In[38]:


plt.scatter(y=Elektrisch['Massa rijklaar'], x=Elektrisch['Massa ledig voertuig'])


# In[39]:


Elektrisch1 = Elektrisch[Elektrisch['Massa rijklaar'] > 750]
Elektrisch1['Massa rijklaar'].hist(bins = 40)
plt.show()


# In[40]:


pd.isna(Elektrisch1['Catalogusprijs']).sum()


# In[41]:


Elektrisch1['Catalogusprijs'].fillna(Elektrisch['Catalogusprijs'].mean(), inplace=True)


# #### SELECTION OF COLUMNS TO USE

# In[42]:


data = ['Merk', 'Handelsbenaming', 'Inrichting', 'Eerste kleur', 'Massa rijklaar', 'Zuinigheidslabel', 'Catalogusprijs'] 
df = Elektrisch1[data]


# In[43]:


df.head()


# In[44]:


pd.isna(df['Catalogusprijs']).sum()


# In[45]:


df.info()


# In[46]:


df['Zuinigheidslabel'].fillna(('Onbekend'), inplace=True)


# In[47]:


df['Zuinigheidslabel'].value_counts().sort_values()


# In[48]:


del df['Zuinigheidslabel']


# In[49]:


df.info()


# In[50]:


df.describe()


# In[51]:


df['Catalogusprijs'].max()


# In[52]:


df1 = df[df['Catalogusprijs'] <= 200000]
df1.info()


# In[53]:


plt.hist(df1['Catalogusprijs'], bins=100)


# In[54]:


df1['Eerste kleur'].value_counts()


# In[55]:


df1['Inrichting'].value_counts()


# In[56]:


df1.groupby("Merk")['Handelsbenaming'].unique()


# In[ ]:





# In[57]:


df1["Merk"].unique()


# In[58]:


df1['Merk'].value_counts()


# In[59]:


from fuzzywuzzy import fuzz


# In[60]:


print(fuzz.ratio('TESLA','TESLA MOTORS'))
print(fuzz.partial_ratio('TESLA','TESLA MOTORS'))


# In[61]:


df1 = df1.assign(ratio = df1.apply(lambda x: fuzz.ratio(x['Merk'], 'TESLA'), axis= 1))
df1.loc[:, ['Merk', 'ratio']].drop_duplicates().head()


# In[62]:


df1 = df1.assign(Merk2 = np.where(df1['ratio'] >= 80,'TESLA', df1['Merk']))


# In[63]:


df1


# In[64]:


df1['Merk2'].unique()


# In[65]:


df1['Merk2'].value_counts()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




