import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

st.title("TRUTHS")
st.write(
    "DIRTT: Data assimilation for Inverting Radiative Transfer models using Truths data"
)

st.image("TRUTHSsampling.png", caption="Sunrise by the mountains")

all_users = ["Alice", "Bob", "Charly"]
with st.container(border=True):
    users = st.multiselect("Users", all_users, default=all_users)
    rolling_average = st.toggle("Rolling average")

np.random.seed(42)
data = pd.DataFrame(np.random.randn(20, len(users)), columns=users)
if rolling_average:
    data = data.rolling(7).mean().dropna()

tab1, tab2 = st.tabs(["Chart", "Dataframe"])
tab1.line_chart(data, height=250)
tab2.dataframe(data, height=250, use_container_width=True)