#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import warnings
warnings.filterwarnings("ignore")


# ## Best Model

# In[5]:


HousePrice=pd.read_csv('G:/course/20231/ai/project3/data/all.csv')


# In[9]:


HousePrice.drop(columns=['Unnamed: 0'],inplace=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Front-end

# In[ ]:


option = st.selectbox(
    'which column do you like best?',
    ['+', '-'])

data1 = st.number_input('input a number', key='input1')
data2 = st.number_input('input a number', key='input2')
weight = st.select_slider('select a number', options = range(15))
if option == '+':
    result = weight*(data1 + data2)
else:
    result = weight*(data1 - data2)
button = st.button('calculate')
if button:
    st.write('Result: ', result )

