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

st.image("slope.png", caption="Global plot showing the sensitivity of gross primary productivity (GPP) to a unit increase in leaf area index (LAI) indicating the extent of the increase in carbon sequestered during photosynthesis for a particular plant functional type. Green = increase, pink = decrease, measured in kg C m$^{-2}$ s$^{-1}$", width="content")
#st.image("Images/TRUTHSsampling.png", caption="Site locations for TRUTHS geometries", width="content")
st.divider()
st.write("""In this project, we inverted radiative transfer models against hypothetical TRUTHS reflectance data at different global locations 
         and at different sampling geometries to retrieve the albedo and biophysical traits (such as leaf area index) of the canopy scene. In 
         phase 1 of the project, we produced synthetic observations of bidirectional reflectance factors (BRFs) given specific illumination, 
         viewing geometries and known canopy characteristics (e.g. leaf area index and projected crown cover) using the GORT radiative transfer model. 
         In this phase, we inverted the BRFs using the linear Ross-Thick Li-Sparse kernel model to assess the ability to retrieve reflectances and model black sky surface albedos. 
         In phase 2 of the project, we generated synthetic BRFs using the Semi-Discrete model and inverted against the same nonlinear model using the 4DEnVar data assimilation technique. 
         In each phase, we quantified the additional benefits of having TRUTHS' data, specifically with regard to characteristics such as instrument accuracy and additional geometries, 
         over those offered by other missions alone (such as Sentinel-2). This study illustrates the capacity, although beyond the lifetime of this project, to assimilate TRUTHS data 
         into full complex land surface models such as JULES.""")
st.divider()
st.write("""We produced this app to demonstrate the results from Phase 1 of the project. Click on the results tab to vary the parameters of our retrievals.""")
st.divider()
st.subheader("Sentinel-2 uncertainty settings (fixed)")
st.markdown(
    """
    The default Sentinel-2 observation uncertainties used in this app are specified following:

    👉 **Source:** [Sentinel-2 Radiometric Performance Documentation](https://ieeexplore.ieee.org/document/10613854)

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
