#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import streamlit as st
import warnings
import requests
import joblib
warnings.filterwarnings("ignore")


# ## Best Model

# In[3]:


github_file_url = 'https://github.com/liyang6620/158222/raw/main/model.joblib'
response = requests.get(github_file_url)

with open('model.joblib', 'wb') as f:
    f.write(response.content)

model = joblib.load('model.joblib')


# ## Front-end

# target_encoded_value = 29
# target_data = Suburb_encoder.inverse_transform([target_encoded_value])[0]

# In[28]:


st.title("House Price Prediction")
Areas = ['Central Auckland', 'Central Suburbs', 'Eastern Suburbs','Franklin/Manukau Rural', 'North Shore', 'Pakuranga/Howick','Rodney', 'South Auckland', 'West Auckland']
Regions = st.selectbox('Regions', Areas)
Central_Auckland=['CityCentre']
Central_Suburbs=['Epsom', 'Greenlane', 'Hillsborough', 'Lynfield', 'MountEden',
       'MountRoskill', 'NewWindsor', 'Onehunga', 'RoyalOak',
       'Sandringham', 'ThreeKings', 'GreyLynn', 'MountAlbert',
       'PointChevalier', 'OneTreeHill', 'HerneBay', 'Morningside',
       'Avondale', 'BlockhouseBay', 'FreemansBay', 'Pt.Chevalier',
       'Westmere', 'Ponsonby', 'Newmarket', 'Waterview', 'Grafton',
       'Kingsland', 'StMarysBay']
Eastern_Suburbs=['Ellerslie', 'GlenInnes', 'Glendowie', 'MissionBay',
       'MountWellington', 'Panmure', 'Parnell', 'PointEngland', 'Remuera',
       'SaintHeliers', 'Kohimarama', 'Meadowbank', 'SaintJohns',
       'Stonefields', 'Orakei', 'PtEngland']
Franklin_Manukau_Rural=['Pukekohe', 'Tuakau', 'Waiuku', 'Pokeno', 'Karaka', 'Bombay',
       'Patumahoe', 'ClarksBeach', 'Glenbrook']
North_Shore=['Albany', 'Bayview', 'BeachHaven', 'Birkdale', 'Birkenhead',
       'BrownsBay', 'Devonport', 'ForrestHill', 'Glenfield', 'Hillcrest',
       'Milford', "Murray'sBay", 'Northcote', 'Sunnynook', 'Takapuna',
       'Torbay', 'UnsworthHeights', 'LongBay', 'CastorBay', 'Greenhithe',
       'MairangiBay', 'Pinehill', 'Belmont', 'TotaraVale', 'Northcross',
       'RothesayBay', 'DairyFlat', 'Bayswater']
Pakuranga_Howick=['Beachlands', 'BotanyDowns', 'BucklandsBeach', 'Dannemora',
       'FarmCove', 'FlatBush', 'HalfmoonBay', 'HighlandPark', 'Howick',
       'Northpark', 'Pakuranga', 'Burswood', 'CockleBay',
       'EastTamakiHeights', 'Somerville', 'PakurangaHeights', 'Maraetai',
       'ShellyPark', 'Sunnyhills', 'MellonsBay', 'Whitford']
Rodney=['GulfHarbour', 'Helensville', 'Manly', 'Orewa', 'Silverdale',
       'StanmoreBay', 'RedBeach', 'Warkworth', 'Kumeu', 'Stillwater',
       'ArmyBay', 'Kaukapakapa', 'ArklesBay', 'Waimauku', 'Wellsford',
       'SnellsBeach', 'Taupaki']
South_Auckland=['ClendonPark', 'CloverPark', 'ConiferGrove', 'Favona',
       'GoodwoodHeights', 'Mangere', 'MangereBridge', 'MangereEast',
       'Manurewa', 'Otahuhu', 'Otara', 'Papakura', 'Papatoetoe',
       'RandwickPark', 'Takanini', 'WattleDowns', 'Weymouth', 'Karaka',
       'TheGardens', 'Drury', 'FlatBush', 'HillPark', 'Opaheke',
       'EastTamaki', 'Manukau', 'Wiri', 'TotaraHeights', 'Alfriston']
West_Auckland=['Avondale', 'BlockhouseBay', 'GlenEden', 'Glendene', 'GreenBay',
       'Henderson', 'Hobsonville', 'Kelston', 'Massey', 'NewLynn',
       'Ranui', 'Sunnyvale', 'TeAtatuPeninsula', 'TeAtatuSouth',
       'Titirangi', 'WestHarbour', 'Whenuapai', 'Swanson',
       'HendersonHeights', 'Westgate', 'Laingholm', 'RoyalHeights',
       'Waitakere']


if Regions=='Central Auckland':
    suburb = st.selectbox('Suburbs', Central_Auckland)
elif Regions=='Central Suburbs': 
    suburb = st.selectbox('Suburbs', Central_Suburbs)
elif Regions=='Eastern Suburbs': 
    suburb = st.selectbox('Suburbs', Eastern_Suburbs)
elif Regions=='Franklin/Manukau Rural': 
    suburb = st.selectbox('Suburbs', Franklin_Manukau_Rural)
elif Regions=='North Shore': 
    suburb = st.selectbox('Suburbs', North_Shore)
elif Regions=='Pakuranga/Howick': 
    suburb = st.selectbox('Suburbs', Pakuranga_Howick)
elif Regions=='Rodney': 
    suburb = st.selectbox('Suburbs', Rodney)
elif Regions=='South Auckland': 
    suburb = st.selectbox('Suburbs', South_Auckland)
elif Regions=='West Auckland': 
    suburb = st.selectbox('Suburbs', West_Auckland)

year = st.selectbox('Year', range(2020, 2100))    
month = st.selectbox('Month', range(1, 13))


# In[ ]:


suburb_encoding_map = {
    'Albany': 0,
    'Alfriston': 1,
    'ArklesBay': 2,
    'ArmyBay': 3,
    'Avondale': 4,
    'Bayswater': 5,
    'Bayview': 6,
    'BeachHaven': 7,
    'Beachlands': 8,
    'Belmont': 9,
    'Birkdale': 10,
    'Birkenhead': 11,
    'BlockhouseBay': 12,
    'Bombay': 13,
    'BotanyDowns': 14,
    'BrownsBay': 15,
    'BucklandsBeach': 16,
    'Burswood': 17,
    'CastorBay': 18,
    'CityCentre': 19,
    'ClarksBeach': 20,
    'ClendonPark': 21,
    'CloverPark': 22,
    'CockleBay': 23,
    'ConiferGrove': 24,
    'DairyFlat': 25,
    'Dannemora': 26,
    'Devonport': 27,
    'Drury': 28,
    'EastTamaki': 29,
    'EastTamakiHeights': 30,
    'Ellerslie': 31,
    'Epsom': 32,
    'FarmCove': 33,
    'Favona': 34,
    'FlatBush': 35,
    'ForrestHill': 36,
    'FreemansBay': 37,
    'GlenEden': 38,
    'GlenInnes': 39,
    'Glenbrook': 40,
    'Glendene': 41,
    'Glendowie': 42,
    'Glenfield': 43,
    'GoodwoodHeights': 44,
    'Grafton': 45,
    'GreenBay': 46,
    'Greenhithe': 47,
    'Greenlane': 48,
    'GreyLynn': 49,
    'GulfHarbour': 50,
    'HalfmoonBay': 51,
    'Helensville': 52,
    'Henderson': 53,
    'HendersonHeights': 54,
    'HerneBay': 55,
    'HighlandPark': 56,
    'HillPark': 57,
    'Hillcrest': 58,
    'Hillsborough': 59,
    'Hobsonville': 60,
    'Howick': 61,
    'Karaka': 62,
    'Kaukapakapa': 63,
    'Kelston': 64,
    'Kingsland': 65,
    'Kohimarama': 66,
    'Kumeu': 67,
    'Laingholm': 68,
    'LongBay': 69,
    'Lynfield': 70,
    'MairangiBay': 71,
    'Mangere': 72,
    'MangereBridge': 73,
    'MangereEast': 74,
    'Manly': 75,
    'Manukau': 76,
    'Manurewa': 77,
    'Maraetai': 78,
    'Massey': 79,
    'Meadowbank': 80,
    'MellonsBay': 81,
    'Milford': 82,
    'MissionBay': 83,
    'Morningside': 84,
    'MountAlbert': 85,
    'MountEden': 86,
    'MountRoskill': 87,
    'MountWellington': 88,
    "Murray'sBay": 89,
    'NewLynn': 90,
    'NewWindsor': 91,
    'Newmarket': 92,
    'Northcote': 93,
    'Northcross': 94,
    'Northpark': 95,
    'OneTreeHill': 96,
    'Onehunga': 97,
    'Opaheke': 98,
    'Orakei': 99,
    'Orewa': 100,
    'Otahuhu': 101,
    'Otara': 102,
    'Pakuranga': 103,
    'PakurangaHeights': 104,
    'Panmure': 105,
    'Papakura': 106,
    'Papatoetoe': 107,
    'Parnell': 108,
    'Patumahoe': 109,
    'Pinehill': 110,
    'PointChevalier': 111,
    'PointEngland': 112,
    'Pokeno': 113,
    'Ponsonby': 114,
    'Pt.Chevalier': 115,
    'PtEngland': 116,
    'Pukekohe': 117,
    'RandwickPark': 118,
    'Ranui': 119,
    'RedBeach': 120,
    'Remuera': 121,
    'RothesayBay': 122,
    'RoyalHeights': 123,
    'RoyalOak': 124,
    'SaintHeliers': 125,
    'SaintJohns': 126,
    'Sandringham': 127,
    'ShellyPark': 128,
    'Silverdale': 129,
    'SnellsBeach': 130,
    'Somerville': 131,
    'StMarysBay': 132,
    'StanmoreBay': 133,
    'Stillwater': 134,
    'Stonefields': 135,
    'Sunnyhills': 136,
    'Sunnynook': 137,
    'Sunnyvale': 138,
    'Swanson': 139,
    'Takanini': 140,
    'Takapuna': 141,
    'Taupaki': 142,
    'TeAtatuPeninsula': 143,
    'TeAtatuSouth': 144,
    'TheGardens': 145,
    'ThreeKings': 146,
    'Titirangi': 147,    
    'Torbay': 148,
    'TotaraHeights': 149,
    'TotaraVale': 150,
    'Tuakau': 151,
    'UnsworthHeights': 152,
    'Waimauku': 153,
    'Waitakere': 154,
    'Waiuku': 155,
    'Warkworth': 156,
    'Waterview': 157,
    'WattleDowns': 158,
    'Wellsford': 159,
    'WestHarbour': 160,
    'Westgate': 161,
    'Westmere': 162,
    'Weymouth': 163,
    'Whenuapai': 164,
    'Whitford': 165,
    'Wiri': 166
}


area_encoding_map = {
    'Central Auckland': 0,
    'Central Suburbs': 1,
    'Eastern Suburbs': 2,
    'Franklin/Manukau Rural': 3,
    'North Shore': 4,
    'Pakuranga/Howick': 5,
    'Rodney': 6,
    'South Auckland': 7,
    'West Auckland': 8
}


# In[ ]:


suburb_num = suburb_encoding_map.get( suburb )
area_num = area_encoding_map.get(Regions)
prediction_data = pd.DataFrame({
    'Suburb_num': [suburb_num],
    'Month_num': [month],
    'Area_num': [area_num],
    'Year': [year]
})

prediction = float(model.predict(prediction_data))

button_result = st.button("Result", key="result")
if button_result:
    st.write(prediction)


# In[ ]:




