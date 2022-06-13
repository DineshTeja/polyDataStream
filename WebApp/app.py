import streamlit as st
import numpy as np
import pandas as pd

#show_predict_page()
from PIL import Image


# Custom imports 
from multipage import MultiPage
from WebApp/pages import predict_page,explore_page,housepred

from predict_page import show_predict_page


# Create an instance of the app 
app = MultiPage()

# Title of the main page
image = Image.open('images/polydatalogo.png')

col1, col2, col3 = st.columns([1,6,1])

with col1:
    st.write("")

with col2:
    st.image(image,width=490)
        

with col3:
    st.write("")
    


# Add all your applications (pages) here
app.add_page("U.S. Presidential Prediction Model", predict_page.show_predict_page)
app.add_page("U.S. Senate Prediction Model", explore_page.show_explore_page)
app.add_page("U.S. House Prediction Model", housepred.show_house_page)


# The main app
app.run()
