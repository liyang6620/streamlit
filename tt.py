import streamlit as st
import pandas as pd
import numpy as np


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