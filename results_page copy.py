import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

st.title("Results")

tab1, tab2 = st.tabs(["GORT", "Semi-Discrete"])
tab1.image("gort_brf_timeseries.png", width='stretch')
tab1.image("gort_brf_scatter.png",width="stretch")
tab2.image("semid_brf_timeseries.png", width='stretch')
tab2.image("semid_brf_scatter.png", width='stretch')
