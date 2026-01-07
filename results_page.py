import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.title("Results")

SITES = [
    {"site": "Site01", "lat": 0.0, "lon": 10.0, "slug": "site01"},
    {"site": "Site02", "lat": 60.0, "lon": 30.0, "slug": "site02"},
]
DATA_DIR = Path("data")