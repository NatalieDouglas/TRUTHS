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
def load_site_data(timestamps_csv: Path, albedos_csv: Path, reflectances_csv: Path):
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

    # ---- Load reflectances ----
    ref = pd.read_csv(reflectances_csv)

    if len(ref) != len(ts):
        raise ValueError(
            f"Row mismatch: reflectances({len(ref)}) vs timestamps({len(ts)})"
        )
    ref = ref.drop(columns=["mission",'Unnamed: 0'])
    
    df = pd.concat([ts.reset_index(drop=True), alb.reset_index(drop=True)], axis=1)
    df_ref = pd.concat([ts.reset_index(drop=True), ref.reset_index(drop=True)], axis=1)
    
    # Clean + index
    #df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    df = df.dropna(subset=["datetime"]).set_index("datetime")
    df_ref = df_ref.dropna(subset=["datetime"]).set_index("datetime")

    # Ensure numeric albedo columns
    for c in df.columns:
        if c != "mission":
            df[c] = pd.to_numeric(df[c], errors="ignore")
            df_ref[c] = pd.to_numeric(df_ref[c], errors="ignore")

    return df, df_ref

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
    weights,K=k.solveKernelBRDF(R)
    k.predict_brfs(weights)
    
    # get standard deviations
    ref_std=[]
    for i in range(0,11):
        work0=np.dot(K.T,np.dot(np.linalg.inv(R[i]),K))
        ref_cov=np.dot(np.dot(K,np.linalg.inv(work0)),K.T)
        ref_std.append(np.sqrt(np.diag(ref_cov)))
    
    kbs=[]
    for i in range(len(k.sza_arr)):
        kbs.append(k.predictBSAlbedoRTkLSp(weights,k.sza_arr[i]))
    pred_alb=BRFs_mission.copy()
    pred_alb.loc[:, band_cols] = kbs
    #st.write(pred_alb)
    return BRFs_data, pred_alb, ref_std

def make_plots(df: pd.DataFrame, df1: pd.DataFrame, wl_col, all_wl,pred_ref,pred_alb,ref_std,LAI,PCC,IMG_DIR, LAT,LON,hide):
    
    colors = ["blue","orange","green","red","purple","brown","pink","olive","gray","cyan","gold"]

    if hide:
        hide_wl = [w for w in all_wl if w != "945.1"]
        colors_hide = ["blue","orange","green","red","purple","brown","pink","olive","cyan","gold"]

    else:
        hide_wl=all_wl
        colors_hide = ["blue","orange","green","red","purple","brown","pink","olive","gray","cyan","gold"]
    
    
    if wl_col == "ALL":
        wl=all_wl
    #    msize=2
    else:
        wl=[wl_col]
    #    msize=5
    
    #truths = df.loc[df["mission"] == "TRUTHS"]
    #s2     = df.loc[df["mission"] == "Sentinel2"]
    truths = df.loc[df["mission"] == "TRUTHS", wl]
    s2     = df.loc[df["mission"] == "Sentinel2", wl]
    #both = df[wl]
    #pred_alb_wl = pred_alb[wl]
    

    #fig, axes = plt.subplots(2,2, figsize=(12, 10))
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    ax1.imshow(Image.open(IMG_DIR / ('mapLAT'+str(int(LAT))+'LON'+str(LON)+'.png')))
    ax1.axis("off")
    
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.imshow(Image.open(IMG_DIR / ('Canopies/ImageLAI'+str(LAI)+'PCC'+str(PCC)+'.png')),aspect="auto")
    ax2.set_aspect("auto")
    ax2.axis("off")
    ax2.margins(0)
    
    fig3, ax3 = plt.subplots(figsize=(4, 4))
    ax3.imshow(Image.open(IMG_DIR / ('polar_plots/'+'angular_sampling_LAT'+str(LAT)+'LON'+str(LON)+'.png')))
    ax3.axis("off")

    fig4, ax4 = plt.subplots(figsize=(4, 4))

    if wl_col == "ALL":
        for i,w in enumerate(wl):
            ax4.plot(truths.index, truths[w], "-o", label="TRUTHS",markersize=3,color=colors[i])
            ax4.plot(s2.index,     s2[w],     "-s", label="Sentinel-2",markersize=3,color=colors[i])
    else:
        w=[wl_col]
        i = all_wl.index(w[0])
        ax4.plot(truths.index, truths[w], "-o", label="TRUTHS",markersize=5,color=colors[i])
        ax4.plot(s2.index,     s2[w],     "-s", label="Sentinel-2",markersize=5,color=colors[i])        

    ax4.set_title(f"Black Sky Spectral Albedo at {wl_col} nm")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Albedo")
    leg = ax4.legend()
    if wl_col == "ALL":
        leg.remove()

    fig4.autofmt_xdate()

    fig5, ax5 = plt.subplots(figsize=(4, 4))
    if wl_col == "ALL":
        for i,w in enumerate(hide_wl):
            if ret_sel == "TRUTHS":
                #ax5.scatter(df1.loc[df1["mission"] == "TRUTHS", w],pred_ref[w],label=w,color=colors_hide[i])
                ax5.errorbar(x=df1.loc[df1["mission"] == "TRUTHS", w],y=pred_ref[w],yerr=ref_std[i],fmt="o",capsize=1,elinewidth=0.5,color=colors_hide[i],label=float(w[0]))
            elif ret_sel == "Sentinel2":
                #ax5.scatter(df1.loc[df1["mission"] == "Sentinel2", w],pred_ref[w],label=w,color=colors_hide[i])
                ax5.errorbar(x=df1.loc[df1["mission"] == "Sentinel2", w],y=pred_ref[w],yerr=ref_std[i],fmt="o",capsize=1,elinewidth=0.5,color=colors_hide[i],label=float(w[0]))
            else:
                #ax5.scatter(df1[w],pred_ref[w],label=w,color=colors_hide[i])
                ax5.errorbar(x=df1[w],y=pred_ref[w],yerr=ref_std[i],fmt="o",capsize=1,elinewidth=0.5,color=colors_hide[i],label=float(w[0]))

    else:
        w=[wl_col]
        i = all_wl.index(w[0])
        if ret_sel == "TRUTHS":
            #ax5.scatter(df1.loc[df1["mission"] == "TRUTHS", w],pred_ref[w],label=float(w[0]),color=colors[i])
            ax5.errorbar(x=df1.loc[df1["mission"] == "TRUTHS", w],y=pred_ref[w],yerr=ref_std[i],fmt="o",capsize=1,elinewidth=0.5,color=colors[i],label=float(w[0]))
        elif ret_sel == "Sentinel2":
            #ax5.scatter(df1.loc[df1["mission"] == "Sentinel2", w],pred_ref[w],label=float(w[0]),color=colors[i])
            ax5.errorbar(x=df1.loc[df1["mission"] == "Sentinel2", w],y=pred_ref[w],yerr=ref_std[i],fmt="o",capsize=1,elinewidth=0.5,color=colors[i],label=float(w[0]))

        else:
            #ax5.scatter(df1[w],pred_ref[w],label=float(w[0]),color=colors[i])
            ax5.errorbar(x=df1[w],y=pred_ref[w],yerr=ref_std[i],fmt="o",capsize=1,elinewidth=0.5,color=colors[i],label=float(w[0]))

    xlims=ax5.get_xlim()
    ylims=ax5.get_ylim()
    ax5.plot([0,1],[0,1],"--")
    ax5.set_xlim(xlims)
    ax5.set_ylim(ylims)
    ax5.set_title(f"Retrieved versus simulated reflectance at {wl_col} nm")
    ax5.set_xlabel("Simulated 'truth'")
    ax5.set_ylabel("Retrieved")
    ax5.legend()

    fig6, ax6 = plt.subplots(figsize=(4, 4))
    if wl_col == "ALL":
        for i,w in enumerate(hide_wl):
            if ret_sel == "TRUTHS":
                ax6.scatter(df.loc[df["mission"] == "TRUTHS", w],pred_alb[w],label=w,color=colors_hide[i])
            elif ret_sel == "Sentinel2":
                ax6.scatter(df.loc[df["mission"] == "Sentinel2", w],pred_alb[w],label=w,color=colors_hide[i])
            else:
                ax6.scatter(df[w],pred_alb[w],label=w,color=colors_hide[i])
    else:
        w=[wl_col]
        i = all_wl.index(w[0])
        if ret_sel == "TRUTHS":
            ax6.scatter(df.loc[df["mission"] == "TRUTHS", w],pred_alb[w],label=float(w[0]),color=colors[i])
        elif ret_sel == "Sentinel2":
            ax6.scatter(df.loc[df["mission"] == "Sentinel2", w],pred_alb[w],label=float(w[0]),color=colors[i])
        else:
            ax6.scatter(df[w],pred_alb[w],label=float(w[0]),color=colors[i])
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
        st.markdown("### ☀️ Solar Zenith Angular Sampling")
        st.write("The distribution of solar zenith angles for the available TRUTHS and Sentinel-2 solar and viewing geometries.")
        st.pyplot(fig3)

    with col4:
        st.markdown("### 📋 Albedo Timeseries")
        st.write("The black sky spectral albedos (calculated by GORT) for the available TRUTHS and Sentinel-2 solar and viewing geometries. No noise is added at this stage.")
        st.pyplot(fig4)

    col5, col6 = st.columns(2)

    with col5:
        st.markdown("### 📈 Inversion ")
        st.write("Simulated GORT reflectances versus the reflectances retrieved from the inversion of the Ross-Thick Li-Sparse linear kernels model.")
        st.pyplot(fig5)

    with col6:
        st.markdown("### ✨ Albedos ")
        st.write("Simulated GORT black sky albedos versus simulated albedos using the Ross-Thick Li-Sparse linear kernels model.")
        st.pyplot(fig6)
    return 

st.title("Results")

SITES = [
    {"site": "LAT: 0.0, LON: -60.0", "lat": 0.0, "lon": -60.0, "slug": "LAT0.0LON-60.0"},
    {"site": "LAT: 40.0, LON: -120.0", "lat": 40.0, "lon": -120.0, "slug": "LAT40.0LON-120.0"},
    {"site": "LAT: 50.0, LON: -120.0", "lat": 50.0, "lon": -120.0, "slug": "LAT50.0LON-120.0"},
    {"site": "LAT: 60.0, LON: -120.0", "lat": 60.0, "lon": -120.0, "slug": "LAT60.0LON-120.0"},
    {"site": "LAT: 70.0, LON: -120.0", "lat": 70.0, "lon": -120.0, "slug": "LAT70.0LON-120.0"},
    {"site": "LAT: -30.0, LON: -60.0", "lat": -30.0, "lon": -60.0, "slug": "LAT-30.0LON-60.0"},
    {"site": "LAT: -20.0, LON: -60.0", "lat": -20.0, "lon": -60.0, "slug": "LAT-20.0LON-60.0"},
    {"site": "LAT: -10.0, LON: -60.0", "lat": -10.0, "lon": -60.0, "slug": "LAT-10.0LON-60.0"},
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
    df, df_ref = load_site_data(timestamps_csv, albedos_csv, brfs_csv)
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
    hide_bad_wl = False
    if wl_choice == "ALL":
        hide_bad_wl = st.checkbox("Hide 945.1?",value=False)

    retrievals=["TRUTHS","Sentinel2","TRUTHS+Sentinel2"]
    ret_sel=st.selectbox("Retrieve with:", retrievals)

    truths_acc = st.slider(
    "TRUTHS radiometric accuracy (%)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1
    )
 
    alpha = st.slider(
    "Improvement to Sentinel (X)",
    min_value=1.0,
    max_value=10.0,
    value=1.0,
    step=0.5
    )

eps = 1e-3# Avoid σ=0 when reflectance is 0
rel_err_Sentinel=(1/alpha)*np.array([0.0595,0.0413,0.0349,0.0377,0.0356,0.0335,0.0332,0.0335,0.315,0.0355,0.0357])
#rel_err_TRUTHS=alpha*rel_err_Sentinel
rel_err_TRUTHS=(truths_acc/100)*np.ones(11)

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

predicted_refs, predicted_albedos, ref_std=get_pred_albs(brfs_csv,k,ret_sel,rel_err_Sentinel,rel_err_TRUTHS)

# Plot
make_plots(df,df_ref, wl_choice, wl_cols, predicted_refs, predicted_albedos, ref_std, LAI,PCC,IMG_DIR,site["lat"],site["lon"],hide_bad_wl)
#st.pyplot(fig, clear_figure=True)

# Optional table
with st.expander("Show data"):
    if wl_choice == "ALL":
        st.dataframe(df.drop(columns='Unnamed: 0').sort_index())
    else:
        dontkeep = [w for w in wl_cols if w != wl_choice]
        st.dataframe(df.drop(columns=dontkeep+['Unnamed: 0']).dropna().sort_index())


st.divider()
st.subheader("Sentinel-2 uncertainty settings")

# Example structure — replace values with YOUR spec
s2_sigma_rows = [
    {"Band": "B2",  "λ (nm)": 492.4,  "Std dev (default)": 5.95, "Std dev (updated)": (1.0/alpha)*5.95},
    {"Band":"B3",  "λ (nm)": 559.8,  "Std dev (default)": 4.13, "Std dev (updated)": (1.0/alpha)*4.13},
    {"Band": "B4", "λ (nm)": 664.6,  "Std dev (default)": 3.49, "Std dev (updated)": (1.0/alpha)*3.48},
    {"Band":"B5", "λ (nm)": 704.1,  "Std dev (default)": 3.77, "Std dev (updated)": (1.0/alpha)*3.77},
    {"Band":"B6",  "λ (nm)": 740.5,  "Std dev (default)": 3.56, "Std dev (updated)": (1.0/alpha)*3.56},
    {"Band": "B7", "λ (nm)": 782.8,  "Std dev (default)": 3.35, "Std dev (updated)": (1.0/alpha)*3.35},
    {"Band":"B8", "λ (nm)": 832.8,  "Std dev (default)": 3.32, "Std dev (updated)": (1.0/alpha)*3.32},
    {"Band": "B8A","λ (nm)": 864.7,  "Std dev (default)": 3.35, "Std dev (updated)": (1.0/alpha)*3.35},
    {"Band":"B9", "λ (nm)": 945.1, "Std dev (default)": 31.5, "Std dev (updated)": (1.0/alpha)*31.5},
    {"Band":"B11", "λ (nm)": 1613.7, "Std dev (default)": 3.55, "Std dev (updated)": (1.0/alpha)*3.55},
    {"Band":"B12", "λ (nm)": 2202.4, "Std dev (default)": 3.57, "Std dev (updated)": (1.0/alpha)*3.57}
]

s2_sigma_df = pd.DataFrame(s2_sigma_rows)
s2_sigma_df = s2_sigma_df.reset_index(drop=True)
# Make it look tidy
s2_sigma_df["λ (nm)"] = s2_sigma_df["λ (nm)"].map(lambda x: f"{x:.1f}")
s2_sigma_df["Std dev (default)"] = s2_sigma_df["Std dev (default)"].map(lambda x: f"{x:.2f}")
s2_sigma_df["Std dev (updated)"] = s2_sigma_df["Std dev (updated)"].map(lambda x: f"{x:.3f}")

# Static table (won’t change/scroll like dataframe)
#s2_sigma_df = s2_sigma_df.reset_index(drop=True)
st.table(s2_sigma_df)
