import streamlit as st
import numpy as np
import pandas as pd

#show_predict_page()
from PIL import Image


# Custom imports 
#from multipage import MultiPage
#from pages import predict_page,explore_page,housepred



##import multipage
#from multipage import MultiPage
# Create an instance of the app 
#app = MultiPage()

# Title of the main page
image = Image.open('WebApp/images/polydatalogo.png')

#col1, col2, col3 = st.columns([1,6,1])

#img_style = {'width': '50%'}
#with col1:
#    st.write("")

#with col2:
#   st.image(image, width=490)
        

#with col3:
#    st.write("")


#image = Image.open('WebApp/images/predHeader.png')
    
#st.image(image,width=690)


# Add all your applications (pages) here
#app.add_page("U.S. Presidential Prediction Model", show_predict_page)
#app.add_page("U.S. Senate Prediction Model", show_explore_page)
#app.add_page("U.S. House Prediction Model", show_house_page)


# The main app
#app.run()

col1, col2 = st.columns(2)
with col1:

    st.image(image,width=290)
    image2 = Image.open('WebApp/images/polyDataPredHeaderNew.png')
        
    st.image(image2,width=310)
with col2:
    image3 = Image.open('WebApp/images/mitElection.png')
            
    st.image(image3,width=400)

    image4 = Image.open('WebApp/images/arkdems.png')
            
    st.image(image4,width=325)





    

#st.title(' Federal Election Prediction Model Demos')
#st.header(' Check Side Menu to Switch Demos')
