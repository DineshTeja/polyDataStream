import streamlit as st
import pickle

import numpy as np
import pandas as pd
import sklearn

from sklearn.metrics import accuracy_score,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error
from PIL import Image



image = Image.open('WebApp/images/polydatalogo.png')
image2 = Image.open('WebApp/images/ussenatetitle.png')
col1, col2, col3 = st.columns([1,6,1])

with col1:
    st.write("")

with col2:
    st.image(image,width=490)
    

with col3:
    st.write("")

st.image(image2,width=690)