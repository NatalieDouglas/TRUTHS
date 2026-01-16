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
#st.image("Images/TRUTHSsampling.png", caption="Site locations for TRUTHS geometries", width="content")
st.divider()
st.subheader("Sentinel-2 uncertainty settings (fixed)")
st.markdown(
    """
    The Sentinel-2 observation uncertainties used in this app are specified following:

    👉 **Source:** [Sentinel-2 Radiometric Performance Documentation](https://ieeexplore.ieee.org/document/10613854)

    These values are **fixed** and do not change with the UI settings.
    """
)
# Example structure — replace values with YOUR spec
s2_sigma_rows = [
    {"Band": "B2",  "λ (nm)": 492.4,  "Std dev (reflectance)": 5.95},
    {"Band":"B3",  "λ (nm)": 559.8,  "Std dev (reflectance)": 4.13},
    {"Band": "B4", "λ (nm)": 664.6,  "Std dev (reflectance)": 3.49},
    {"Band":"B5", "λ (nm)": 704.1,  "Std dev (reflectance)": 3.77},
    {"Band":"B6",  "λ (nm)": 740.5,  "Std dev (reflectance)": 3.56},
    {"Band": "B7", "λ (nm)": 782.8,  "Std dev (reflectance)": 3.35},
    {"Band":"B8", "λ (nm)": 832.8,  "Std dev (reflectance)": 3.32},
    {"Band": "B8A","λ (nm)": 864.7,  "Std dev (reflectance)": 3.35},
    {"Band":"B9", "λ (nm)": 945.1, "Std dev (reflectance)": 31.5},
    {"Band":"B11", "λ (nm)": 1613.7, "Std dev (reflectance)": 3.55},
    {"Band":"B12", "λ (nm)": 2202.4, "Std dev (reflectance)": 3.57},
]

s2_sigma_df = pd.DataFrame(s2_sigma_rows)
s2_sigma_df = s2_sigma_df.reset_index(drop=True)
# Make it look tidy
s2_sigma_df["λ (nm)"] = s2_sigma_df["λ (nm)"].map(lambda x: f"{x:.1f}")
s2_sigma_df["Std dev (reflectance)"] = s2_sigma_df["Std dev (reflectance)"].map(lambda x: f"{x:.2f}")

# Static table (won’t change/scroll like dataframe)
#s2_sigma_df = s2_sigma_df.reset_index(drop=True)
st.table(s2_sigma_df)
