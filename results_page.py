import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import kernels
from kernels import kernelBRDF
from dataclasses import dataclass
from PIL import Image
from scipy.linalg import block_diag

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
    n_truths = (BRFs["mission"] == "TRUTHS").sum()
    n_sent = (BRFs["mission"] == "Sentinel2").sum()
    R_TRUTHS=[]
    R_Sentinel=[]
    for i in range(0,11):
        R_TRUTHS.append(np.diag(rel_err_TRUTHS[i]**2*np.ones(n_truths)))
        R_Sentinel.append(np.diag(rel_err_Sentinel[i]**2*np.ones(n_sent)))
    if ret_sel == "TRUTHS": 
        BRFs_mission=BRFs.loc[BRFs["mission"] == "TRUTHS"]
        BRFs_mission = BRFs_mission.drop(columns=["mission",'Unnamed: 0'])
        sigma_arr = rel_err_TRUTHS * np.maximum(BRFs_mission.values, eps)
        R=R_TRUTHS.copy()
    elif ret_sel == "Sentinel2":
        BRFs_mission=BRFs.loc[BRFs["mission"] == "Sentinel2"]
        BRFs_mission = BRFs_mission.drop(columns=["mission",'Unnamed: 0'])
        sigma_arr = rel_err_Sentinel * np.maximum(BRFs_mission.values, eps)
        R=R_Sentinel.copy()
    else:
        BRFs_mission = BRFs.copy()
        BRFs_mission = BRFs_mission.drop(columns=["mission",'Unnamed: 0'])
        sigma_arr = np.zeros_like(BRFs_mission.values, dtype=float)
        sigma_arr[:n_sent] = rel_err_Sentinel * np.maximum(BRFs_mission.values[:n_sent], eps)
        sigma_arr[n_sent:] = rel_err_TRUTHS * np.maximum(BRFs_mission.values[n_sent:], eps)
        R=[]
        for i in range(0,11):
            R.append(block_diag(R_TRUTHS[i],R_Sentinel[i]))
        
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, sigma_arr)
    #BRFs_data = BRFs_mission.drop(columns=["mission",'Unnamed: 0'])
    BRFs_data=BRFs_mission.copy()
    band_cols = BRFs_data.columns
    BRFs_data.loc[:, band_cols] = BRFs_mission.values+noise
    
    k=add_obs_brfs_to_kernelBRDF(BRFs_data.values,k)
    weights=k.solveKernelBRDF(R)
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

def make_plots(df: pd.DataFrame, wl_col, all_wl,pred_alb,LAI,PCC,IMG_DIR, LAT,LON):
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
    

    #fig, axes = plt.subplots(2,2, figsize=(12, 10))
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    ax1.imshow(Image.open(IMG_DIR / ('mapLAT'+str(int(LAT))+'LON'+str(LON)+'.png')))
    ax1.axis("off")
    
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.imshow(Image.open(IMG_DIR / ('ImageLAI'+str(LAI)+'PCC'+str(PCC)+'.png')),aspect="auto")
    ax2.set_aspect("auto")
    ax2.axis("off")
    ax2.margins(0)

    fig3, ax3 = plt.subplots(figsize=(4, 4))
    colors = ["blue","orange","green","red","purple","brown","pink","olive","gray","cyan","gold"]

    if wl_col == "ALL":
        for i,w in enumerate(all_wl):
            ax3.plot(truths.index, truths[w], "-o", label="TRUTHS",markersize=2,color=colors[i])
            ax3.plot(s2.index,     s2[w],     "-s", label="Sentinel-2",markersize=2,color=colors[i])
    else:
        w=wl_col
        i = wl.index(w)
        ax3.plot(truths.index, truths[w], "-o", label="TRUTHS",markersize=5,color=colors[i])
        ax3.plot(s2.index,     s2[w],     "-s", label="Sentinel-2",markersize=5,color=colors[i])        

    ax3.set_title(f"Black Sky Spectral Albedo at {wl_col} nm")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Albedo")
    leg = ax3.legend()
    if wl_col == "ALL":
        leg.remove()

    fig3.autofmt_xdate()
    
    fig4, ax4 = plt.subplots(figsize=(4, 4))
    for i,w in enumerate(wl):
        if ret_sel == "TRUTHS":
            ax4.scatter(df.loc[df["mission"] == "TRUTHS", w],pred_alb[w],label=w,color=colors[i])
        elif ret_sel == "Sentinel2":
            ax4.scatter(df.loc[df["mission"] == "Sentinel2", w],pred_alb[w],label=w,color=colors[i])
        else:
            ax4.scatter(df[w],pred_alb[w],label=w,color=colors[i])
    xlims=ax4.get_xlim()
    ylims=ax4.get_ylim()
    ax4.plot([0,1],[0,1],"--")
    ax4.set_xlim(xlims)
    ax4.set_ylim(ylims)
    ax4.set_title(f"Retrieved versus simulated BS albedo at {wl_col} nm")
    ax4.set_xlabel("Simulated 'truth'")
    ax4.set_ylabel("Retrieved")
    ax4.legend()

    fig5, ax5 = plt.subplots(figsize=(4, 4))
    for i,w in enumerate(wl):
        if ret_sel == "TRUTHS":
            ax5.scatter(df.loc[df["mission"] == "TRUTHS", w],pred_alb[w],label=w,color=colors[i])
        elif ret_sel == "Sentinel2":
            ax5.scatter(df.loc[df["mission"] == "Sentinel2", w],pred_alb[w],label=w,color=colors[i])
        else:
            ax5.scatter(df[w],pred_alb[w],label=w,color=colors[i])
    xlims=ax5.get_xlim()
    ylims=ax5.get_ylim()
    ax5.plot([0,1],[0,1],"--")
    ax5.set_xlim(xlims)
    ax5.set_ylim(ylims)
    ax5.set_title(f"Retrieved versus simulated BS albedo at {wl_col} nm")
    ax5.set_xlabel("Simulated 'truth'")
    ax5.set_ylabel("Retrieved")
    ax5.legend()

    fig6, ax6 = plt.subplots(figsize=(4, 4))
    for i,w in enumerate(wl):
        if ret_sel == "TRUTHS":
            ax6.scatter(df.loc[df["mission"] == "TRUTHS", w],pred_alb[w],label=w,color=colors[i])
        elif ret_sel == "Sentinel2":
            ax6.scatter(df.loc[df["mission"] == "Sentinel2", w],pred_alb[w],label=w,color=colors[i])
        else:
            ax6.scatter(df[w],pred_alb[w],label=w,color=colors[i])
    xlims=ax6.get_xlim()
    ylims=ax6.get_ylim()
    ax6.plot([0,1],[0,1],"--")
    ax6.set_xlim(xlims)
    ax6.set_ylim(ylims)
    ax6.set_title(f"Retrieved versus simulated BS albedo at {wl_col} nm")
    ax6.set_xlabel("Simulated 'truth'")
    ax6.set_ylabel("Retrieved")
    ax6.legend()
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌍 Site Location")
        st.write("This image shows the site you have selected. Latitude = "+str(LAT)+", Longitude = "+str(LON))
        st.pyplot(fig1)

    with col2:
        st.markdown("### 🌳 Canopy Characteristics ")
        st.write("This is how the GORT radiative transfer model views the canopy with LAI="+str(LAI)+" and PCC="+str(PCC))
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### 📋 Albedo Timeseries")
        st.write("The black sky spectral albedos (calculated by GORT) for the available TRUTHS and Sentinel-2 solar and viewing geometries.")
        st.pyplot(fig3)

    with col4:
        st.markdown("### 📈 Inversion ")
        st.write("Simulated GORT black sky albedos versus the alebdos retrived from the inversion of the Ross-Thick Li-Sparse linear kernels model.")
        st.pyplot(fig4)
    return 

    col5, col6 = st.columns(2)

    with col5:
        st.markdown("### 📈 Inversion ")
        st.write("Simulated GORT black sky albedos versus the alebdos retrived from the inversion of the Ross-Thick Li-Sparse linear kernels model.")
        st.pyplot(fig5)

    with col6:
        st.markdown("### 📈 Albedos ")
        st.write("Simulated GORT black sky albedos versus the alebdos retrived from the inversion of the Ross-Thick Li-Sparse linear kernels model.")
        st.pyplot(fig6)
    return 

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

    alpha_options = {
    "10 × better (0.1)":  0.1,
    "5 × better (0.2)":   0.2,
    "2 × better (0.5)":   0.5,
    "Same as Sentinel-2 (1.0)": 1.0,
}

    label = st.selectbox(
        "Assumed TRUTHS accuracy relative to Sentinel-2",
        list(alpha_options.keys()),
        index=2
    )

    alpha = alpha_options[label]

eps = 1e-3# Avoid σ=0 when reflectance is 0
rel_err_Sentinel=np.array([0.0595,0.0413,0.0349,0.0377,0.0356,0.0335,0.0332,0.0335,0.315,0.0355,0.0357])
rel_err_TRUTHS=alpha*rel_err_Sentinel

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
make_plots(df, wl_choice, wl_cols, predicted_albedos, LAI,PCC,IMG_DIR,site["lat"],site["lon"])
#st.pyplot(fig, clear_figure=True)

# Optional table
with st.expander("Show data"):
    if wl_choice == "ALL":
        st.dataframe(df.drop(columns='Unnamed: 0').sort_index())
    else:
        dontkeep = [w for w in wl_cols if w != wl_choice]
        st.dataframe(df.drop(columns=dontkeep+['Unnamed: 0']).dropna().sort_index())