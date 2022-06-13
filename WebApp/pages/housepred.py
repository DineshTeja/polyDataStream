import streamlit as st
import pickle

import numpy as np
import pandas as pd
import sklearn

from sklearn.metrics import accuracy_score,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error
from PIL import Image

def show_house_page():
    titleImage = Image.open('images/ushousetitle.png')

    st.image(titleImage,width=690)