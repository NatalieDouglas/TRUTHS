import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def files_for_site(site_slug: str,LAI,PCC):
    site_dir = DATA_DIR / site_slug
    timestamps_csv = site_dir / (site_slug+"_geometries.csv")
    albedos_csv = site_dir / (site_slug+"LAI"+str(LAI)+"PCC"+str(PCC)+"_albedos.csv")
    return timestamps_csv, albedos_csv

def parse_timestamp(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    return pd.to_datetime(s, errors="coerce")

@st.cache_data(show_spinner=False)
def load_site_data(timestamps_csv: Path, albedos_csv: Path):
    # ---- Load timestamps ----
    ts = pd.read_csv(timestamps_csv)

    if "datetime" not in ts.columns:
        raise ValueError(f"'datetime' column not found in {timestamps_csv}")
    if "mission" not in ts.columns:
        raise ValueError(f"'mission' column not found in {timestamps_csv}")

    ts["datetime"] = parse_timestamp(ts["datetime"])

    # ---- Load albedos ----
    alb = pd.read_csv(albedos_csv)
    if len(alb) != len(ts):
        raise ValueError(
            f"Row mismatch: albedos({len(alb)}) vs timestamps({len(ts)})"
        )
    alb = alb.drop(columns=["mission",'Unnamed: 0'])
    df = pd.concat([ts.reset_index(drop=True), alb.reset_index(drop=True)], axis=1)

    # Clean + index
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()

    # Ensure numeric albedo columns
    for c in df.columns:
        if c != "mission":
            df[c] = pd.to_numeric(df[c], errors="ignore")

    return df

def get_wavelength_columns(df: pd.DataFrame):
    wl_cols = []
    for c in df.columns:
        if c == "mission":
            continue
        try:
            float(str(c))
            if pd.api.types.is_numeric_dtype(df[c]):
                wl_cols.append(c)
        except ValueError:
            pass

    return sorted(wl_cols, key=lambda x: float(str(x)))

def plot_missions(df: pd.DataFrame, wl_col, show_lines=True):
    truths = df.loc[df["mission"] == "TRUTHS", wl_col]
    s2     = df.loc[df["mission"] == "Sentinel2", wl_col]

    fig, ax = plt.subplots()

    if show_lines:
        ax.plot(truths.index, truths.values, "-o", label="TRUTHS")
        ax.plot(s2.index,     s2.values,     "-s", label="Sentinel-2")
    else:
        ax.plot(truths.index, truths.values, "o", label="TRUTHS")
        ax.plot(s2.index,     s2.values,     "s", label="Sentinel-2")

    ax.set_title(f"Black Sky Spectral Albedo at {wl_col} nm")
    ax.set_xlabel("Time")
    ax.set_ylabel("Albedo")
    ax.legend()
    fig.autofmt_xdate()

    return fig

st.title("Results")

SITES = [
    {"site": "LAT: 40.0, LON: -120.0", "lat": 40.0, "lon": -120.0, "slug": "LAT40.0LON-120.0"},
    {"site": "LAT: 50.0, LON: -120.0", "lat": 50.0, "lon": -120.0, "slug": "LAT50.0LON-120.0"},
    {"site": "LAT: 60.0, LON: -120.0", "lat": 60.0, "lon": -120.0, "slug": "LAT60.0LON-120.0"},
    {"site": "LAT: 70.0, LON: -120.0", "lat": 70.0, "lon": -120.0, "slug": "LAT70.0LON-120.0"},
    {"site": "LAT: -30.0, LON: -60.0", "lat": -30.0, "lon": -60.0, "slug": "LAT-30.0LON-60.0"},
    {"site": "LAT: -20.0, LON: -60.0", "lat": -20.0, "lon": -60.0, "slug": "LAT-20.0LON-60.0"},
    {"site": "LAT: -10.0, LON: -60.0", "lat": -10.0, "lon": -60.0, "slug": "LAT-10.0LON-60.0"},
     {"site": "LAT: 0.0, LON: -60.0", "lat": 0.0, "lon": -60.0, "slug": "LAT0.0LON-60.0"},
    {"site": "LAT: 10.0, LON: 0.0", "lat": 10.0, "lon": 0.0, "slug": "LAT10.0LON0.0"},
    {"site": "LAT: 20.0, LON: 0.0", "lat": 20.0, "lon": 0.0, "slug": "LAT20.0LON0.0"},
    {"site": "LAT: 30.0, LON: 0.0", "lat": 30.0, "lon": 0.0, "slug": "LAT30.0LON0.0"},
    {"site": "LAT: 40.0, LON: 0.0", "lat": 40.0, "lon": 0.0, "slug": "LAT40.0LON0.0"},
    {"site": "LAT: 50.0, LON: 0.0", "lat": 50.0, "lon": 0.0, "slug": "LAT50.0LON0.0"},
    {"site": "LAT: 30.0, LON: 60.0", "lat": 30.0, "lon": 60.0, "slug": "LAT30.0LON60.0"},
    {"site": "LAT: 40.0, LON: 60.0", "lat": 40.0, "lon": 60.0, "slug": "LAT40.0LON60.0"},
    {"site": "LAT: 50.0, LON: 60.0", "lat": 50.0, "lon": 60.0, "slug": "LAT50.0LON60.0"},
    {"site": "LAT: 60.0, LON: 60.0", "lat": 60.0, "lon": 60.0, "slug": "LAT60.0LON60.0"},
    {"site": "LAT: 30.0, LON: 120.0", "lat": 30.0, "lon": 120.0, "slug": "LAT30.0LON120.0"},
    {"site": "LAT: 40.0, LON: 120.0", "lat": 40.0, "lon": 120.0, "slug": "LAT40.0LON120.0"},
    {"site": "LAT: 50.0, LON: 120.0", "lat": 50.0, "lon": 120.0, "slug": "LAT50.0LON120.0"},
    {"site": "LAT: 60.0, LON: 120.0", "lat": 60.0, "lon": 120.0, "slug": "LAT60.0LON120.0"},
    {"site": "LAT: 70.0, LON: 120.0", "lat": 70.0, "lon": 120.0, "slug": "LAT70.0LON120.0"}
]

DATA_DIR = Path("Data")

with st.sidebar:
    st.header("Controls")

    site_names = [s["site"] for s in SITES]
    selected_site_name = st.selectbox("Select site", site_names)

    site = next(s for s in SITES if s["site"] == selected_site_name)

    st.caption(f"Lat/Lon: {site['lat']:.1f}, {site['lon']:.1f}")

    show_lines = st.toggle("Connect points with lines", value=True)

    LAIs=["low","high"]
    selected_LAI = st.selectbox("Select LAI", LAIs)

    PCCs=["low","high"]
    selected_PCC = st.selectbox("Select canopy coverage", PCCs)

if selected_LAI == "low":
    LAI=1.0
else:
    LAI=5.0

if selected_PCC == "low":
    PCC=0.3
else:
    PCC=0.7

# Resolve files
timestamps_csv, albedos_csv = files_for_site(site["slug"],LAI,PCC)

# Load data
try:
    df = load_site_data(timestamps_csv, albedos_csv)
except Exception as e:
    st.error(f"Failed to load site data: {e}")
    st.stop()

wl_cols = get_wavelength_columns(df)
if not wl_cols:
    st.error("No wavelength columns found to plot.")
    st.write("Columns found:", list(df.columns))
    st.stop()

with st.sidebar:
    wl_choice = st.selectbox("Wavelength", wl_cols)

# Plot
fig = plot_missions(df, wl_choice, show_lines=show_lines)
st.pyplot(fig, clear_figure=True)

# Optional table
with st.expander("Show data"):
    st.dataframe(df[["mission", wl_choice]].dropna().sort_index())