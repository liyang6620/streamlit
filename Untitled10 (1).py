#!/usr/bin/env python
# coding: utf-8

# In[23]:


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

# In[26]:


github_file_url = 'https://github.com/liyang6620/158222/raw/main/model.joblib'

response = requests.get(github_file_url)

with open('model.joblib', 'wb') as f:
    f.write(response.content)

model = joblib.load('model.joblib')


# In[27]:


prediction_data = pd.DataFrame({
    'Suburb_num': [32],
    'Month_num': [2],
    'Area_num': [1],
    'Year': [2021]
})

prediction = float(model.predict(prediction_data))
prediction


# ## Front-end

# target_encoded_value = 29
# target_data = Suburb_encoder.inverse_transform([target_encoded_value])[0]

# In[ ]:


button = st.button("show")
if button:
    st.text(prediction)



