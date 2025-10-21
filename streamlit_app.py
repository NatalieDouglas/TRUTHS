import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

st.html("""
    <style>
        .stMainBlockContainer {
            max-width:75rem;
        }
    </style>
    """
)

st.title(
    "DIRTT: Data assimilation for Inverting Radiative Transfer models using TRUTHS data"
)
st.write("""Accurately quantifying photosynthesis is essential for guiding climate mitigation and land-management policies 
         because it controls terrestrial carbon uptake and directly affects the global carbon budget. Reliable estimates of 
          photosynthetic activity improve future projections of atmospheric CO2 concentrations, enhance the predictive skill of 
           climate models, and support evidence-based decisions for limiting future climate impacts. The TRUTHS (Traceable Radiometry Underpinning 
          Terrestrial- and Helio-studies) mission, originally scheduled for launch in 2030, aims to provide high-precision solar and reflected radiance measurements, 
           establishing a gold standard for Earth observation calibration. By enabling more accurate satellite data, TRUTHS will enhance our ability 
            to monitor photosynthesis and its role in the carbon cycle, thereby infomring climate policy and improving climate predictions.""")

st.image("TRUTHSsampling.png", caption="Site locations for TRUTHS geometries", width="content")

tab1, tab2 = st.tabs(["GORT", "Semi-Discrete"])
tab1.image("gort_brf_timeseries.png", width='stretch')
tab1.image("gort_brf_scatter.png",width="stretch")
tab2.image("semid_brf_timeseries.png", width='stretch')
tab2.image("semid_brf_scatter.png", width='stretch')

#all_users = ["Alice", "Bob", "Charly"]
#with st.container(border=True):
#    users = st.multiselect("Users", all_users, default=all_users)
#    rolling_average = st.toggle("Rolling average")

#np.random.seed(42)
#data = pd.DataFrame(np.random.randn(20, len(users)), columns=users)
#if rolling_average:
#    data = data.rolling(7).mean().dropna()

#tab1, tab2 = st.tabs(["Chart", "Dataframe"])
#tab1.line_chart(data, height=250)
#tab2.dataframe(data, height=250, width='stretch')