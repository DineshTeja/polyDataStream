import streamlit as st
import numpy as np
import pandas as pd

#show_predict_page()
from PIL import Image


# Custom imports 
from multipage import MultiPage
#from pages import predict_page,explore_page,housepred

from pages.predict_page import show_predict_page
from pages.explore_page import show_explore_page
from pages.housepred import show_house_page

# Create an instance of the app 
app = MultiPage()

# Title of the main page
image = Image.open('WebApp/images/polydatalogo.png')

col1, col2, col3 = st.columns([1,6,1])

with col1:
    st.write("")

with col2:
    st.image(image,width=490)
        

with col3:
    st.write("")
    


# Add all your applications (pages) here
app.add_page("U.S. Presidential Prediction Model", show_predict_page)
app.add_page("U.S. Senate Prediction Model", show_explore_page)
app.add_page("U.S. House Prediction Model", show_house_page)


# The main app
app.run()
