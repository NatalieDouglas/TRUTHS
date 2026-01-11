import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

st.title(
    "DIRTT: Data assimilation for Inverting Radiative Transfer models using TRUTHS data"
)
st.write("""Accurately quantifying photosynthesis is essential for guiding climate mitigation and land-management policies 
         because it controls terrestrial carbon uptake and directly affects the global carbon budget. Reliable estimates of 
          photosynthetic activity improve future projections of atmospheric CO2 concentrations, enhance the predictive skill of 
           climate models, and support evidence-based decisions for limiting future climate impacts. The TRUTHS (Traceable Radiometry Underpinning 
          Terrestrial- and Helio-studies) mission, originally scheduled for launch in 2030, aims to provide high-precision solar and reflected radiance measurements, 
           establishing a gold standard for Earth observation calibration. By enabling more accurate satellite data, TRUTHS will enhance our ability 
            to monitor photosynthesis and its role in the carbon cycle, thereby informing climate policy and improving climate predictions.""")

st.image("slope.png", caption="Site locations for TRUTHS geometries", width="content")
st.image("Images/TRUTHSsampling.png", caption="Site locations for TRUTHS geometries", width="content")