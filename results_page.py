import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import kernels
from kernels import kernelBRDF
from dataclasses import dataclass
from PIL import Image

@dataclass
class geom:
    vza: float
    vaa: float
    sza: float
    saa: float

def add_obs_brfs_to_kernelBRDF(brfs,k):
    for i in range( k.nAngles ):
        for j in range( k.nWavelengths ):
            k.brfObs[i][j]=brfs[i][j]

    return k

def geom_list_from_brdfFile(b):
    geom_list = []
    for i in range(b.nAngles):
        geom_list.append(geom(vza=b.vza_arr[i], vaa=b.vaa_arr[i], sza=b.sza_arr[i], saa=b.saa_arr[i]))
    return geom_list 

def files_for_site(site_slug: str,LAI,PCC):
    site_dir = DATA_DIR / site_slug
    timestamps_csv = site_dir / (site_slug+"_geometries.csv")
    albedos_csv = site_dir / (site_slug+"LAI"+str(LAI)+"PCC"+str(PCC)+"_albedos.csv")
    brfs_csv = site_dir / (site_slug+"LAI"+str(LAI)+"PCC"+str(PCC)+"_BRFs.csv")
    return timestamps_csv, albedos_csv, brfs_csv

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
    #df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    df = df.dropna(subset=["datetime"]).set_index("datetime")

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

def get_pred_albs(brfs_csv,k,ret_sel,rel_err_Sentinel,rel_err_TRUTHS):
    BRFs = pd.read_csv(brfs_csv)
    if ret_sel == "TRUTHS": 
        BRFs_mission=BRFs.loc[BRFs["mission"] == "TRUTHS"]
        BRFs_mission = BRFs_mission.drop(columns=["mission",'Unnamed: 0'])
        sigma_arr = rel_err_TRUTHS * np.maximum(BRFs_mission.values, eps)
    elif ret_sel == "Sentinel2":
        BRFs_mission=BRFs.loc[BRFs["mission"] == "Sentinel2"]
        BRFs_mission = BRFs_mission.drop(columns=["mission",'Unnamed: 0'])
        sigma_arr = rel_err_Sentinel * np.maximum(BRFs_mission.values, eps)
    else:
        BRFs_mission = BRFs.copy()
        BRFs_mission = BRFs_mission.drop(columns=["mission",'Unnamed: 0'])
        sigma_arr = np.zeros_like(BRFs_mission.values, dtype=float)
        n_truths = (BRFs["mission"] == "TRUTHS").sum()
        n_truths = (BRFs["mission"] == "Sentinel2").sum()
        sigma_arr[:n_truths] = rel_err_TRUTHS * np.maximum(BRFs_mission.values[:n_truths], eps)
        sigma_arr[n_truths:] = rel_err_Sentinel * np.maximum(BRFs_mission.values[n_truths:], eps)
    
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, sigma_arr)
    #BRFs_data = BRFs_mission.drop(columns=["mission",'Unnamed: 0'])
    BRFs_data=BRFs_mission.copy()
    band_cols = BRFs_data.columns
    BRFs_data.loc[:, band_cols] = BRFs_mission.values+noise
    
    k=add_obs_brfs_to_kernelBRDF(BRFs_data.values,k)
    weights=k.solveKernelBRDF()
    k.predict_brfs(weights)
    #pred_BRFs.loc[:, band_cols] = k.brf
    #st.write(pred_BRFs)
    #alb = pd.read_csv(albedos_csv) 
    #alb=alb.loc[alb["mission"] == "TRUTHS"]
    #st.write(alb)
       
    kbs=[]
    for i in range(len(k.sza_arr)):
        kbs.append(k.predictBSAlbedoRTkLSp(weights,k.sza_arr[i]))
    pred_alb=BRFs_mission.copy()
    pred_alb.loc[:, band_cols] = kbs
    #st.write(pred_alb)
    return pred_alb

def make_plots(df: pd.DataFrame, wl_col, all_wl,pred_alb,LAI,PCC,IMG_DIR, LAT,LON,show_lines=True):
    if wl_col == "ALL":
        wl=all_wl
        msize=2
    else:
        wl=[wl_col]
        msize=5
    
    truths = df.loc[df["mission"] == "TRUTHS", wl]
    s2     = df.loc[df["mission"] == "Sentinel2", wl]
    both = df[wl]
    pred_alb_wl = pred_alb[wl]
    
    #st.write(pred_alb_wl.columns)
    fig, axes = plt.subplots(2,2, figsize=(12, 10))

    axes[0,0].imshow(Image.open(IMG_DIR / ('mapLAT'+str(int(LAT))+'LON'+str(LON)+'.png')))
    axes[0,0].axis("off")
    axes[0,1].imshow(Image.open(IMG_DIR / ('ImageLAI'+str(LAI)+'PCC'+str(PCC)+'.png')),aspect="auto")
    axes[0,1].set_aspect("auto")
    axes[0,1].axis("off")
    axes[0,1].margins(0)

    colors = ["blue","orange","green","red","purple","brown","pink","olive","gray","cyan","gold"]
    if show_lines:
        if wl_col == "ALL":
            for i,w in enumerate(all_wl):
                axes[1,0].plot(truths.index, truths[w], "-o", label="TRUTHS",markersize=2,color=colors[i])
                axes[1,0].plot(s2.index,     s2[w],     "-s", label="Sentinel-2",markersize=2,color=colors[i])
        else:
            w=wl_col
            i = wl.index(w)
            axes[1,0].plot(truths.index, truths[w], "-o", label="TRUTHS",markersize=5,color=colors[i])
            axes[1,0].plot(s2.index,     s2[w],     "-s", label="Sentinel-2",markersize=5,color=colors[i])        
    else:
        if wl_col == "ALL":
            for i,w in enumerate(all_wl):
                axes[1,0].plot(truths.index, truths[w], "o", label="TRUTHS",markersize=2,color=colors[i])
                axes[1,0].plot(s2.index,     s2[w],     "s", label="Sentinel-2",markersize=2,color=colors[i])
        else:
            w=wl_col
            i = wl.index(w)
            axes[1,0].plot(truths.index, truths[w], "o", label="TRUTHS",markersize=5,color=colors[i])
            axes[1,0].plot(s2.index,     s2[w],     "s", label="Sentinel-2",markersize=5,color=colors[i])        
   
    axes[1,0].set_title(f"Black Sky Spectral Albedo at {wl_col} nm")
    axes[1,0].set_xlabel("Time")
    axes[1,0].set_ylabel("Albedo")
    leg = axes[1,0].legend()
    if wl_col == "ALL":
        leg.remove()

    #colors = [plt.cm.tab20(np.linspace(0, 1, len(wl)-1)),"gold"]
    colors = ["blue","orange","green","red","purple","brown","pink","olive","gray","cyan","gold"]
    for i,w in enumerate(wl):
        if ret_sel == "TRUTHS":
            axes[1,1].scatter(df.loc[df["mission"] == "TRUTHS", w],pred_alb[w],label=w,color=colors[i])
        elif ret_sel == "Sentinel2":
            axes[1,1].scatter(df.loc[df["mission"] == "Sentinel2", w],pred_alb[w],label=w,color=colors[i])
        else:
            axes[1,1].scatter(df[w],pred_alb[w],label=w,color=colors[i])
    xlims=axes[1,1].get_xlim()
    ylims=axes[1,1].get_ylim()
    axes[1,1].plot([0,1],[0,1],"--")
    axes[1,1].set_xlim(xlims)
    axes[1,1].set_ylim(ylims)
    axes[1,1].set_title(f"Predicted versus observed BS albedo at {wl_col} nm")
    axes[1,1].set_xlabel("Observed")
    axes[1,1].set_ylabel("Predicted")
    axes[1,1].legend()

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
IMG_DIR = Path("Images")

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
timestamps_csv, albedos_csv, brfs_csv = files_for_site(site["slug"],LAI,PCC)

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
    wl_choice = st.selectbox("Wavelength", ["ALL"]+wl_cols)

    retrievals=["TRUTHS","Sentinel2","TRUTHS+Sentinel2"]
    ret_sel=st.selectbox("Retrieve with:", retrievals)

eps = 1e-3# Avoid σ=0 when reflectance is 0
rel_err_Sentinel=0.03 # THIS NEEDS TO BE WAVELENGHT DEPENDENT!
rel_err_TRUTHS=0.003

# Inversion
if ret_sel == "TRUTHS":
    BRDF_filename= DATA_DIR / ('BRDF_files/TRUTHS/TRUTHSgeometries/TRUTHSgeomsLAT'+str(site["lat"])+'LON'+str(site["lon"])+'.brdf' )
elif ret_sel == "Sentinel2":
    BRDF_filename= DATA_DIR / ('BRDF_files/Sentinel/SentinelGeometries/SentinelGeomsLAT'+str(site["lat"])+'LON'+str(site["lon"])+'.brdf' )
else:
    BRDF_filename= DATA_DIR / ('BRDF_files/Sentinel+TRUTHS/Sentinel+TRUTHSGeometries/Sentinel+TRUTHSGeomsLAT'+str(site["lat"])+'LON'+str(site["lon"])+'.brdf' )  

k=kernelBRDF( )
k.readBRDF(BRDF_filename)
geom_list=geom_list_from_brdfFile(k)

predicted_albedos=get_pred_albs(brfs_csv,k,ret_sel,rel_err_Sentinel,rel_err_TRUTHS)

# Plot
fig = make_plots(df, wl_choice, wl_cols, predicted_albedos, LAI,PCC,IMG_DIR,site["lat"],site["lon"],show_lines=show_lines)
st.pyplot(fig, clear_figure=True)

# Optional table
with st.expander("Show data"):
    if wl_choice == "ALL":
        st.dataframe(df.drop(columns='Unnamed: 0').sort_index())
    else:
        dontkeep = [w for w in wl_cols if w != wl_choice]
        st.dataframe(df.drop(columns=dontkeep+['Unnamed: 0']).dropna().sort_index())