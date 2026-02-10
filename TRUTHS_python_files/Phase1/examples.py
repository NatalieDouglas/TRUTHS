import numpy as np
from copy import copy

import importlib
import brdfFile
import gort
import kernels
importlib.reload(brdfFile)
importlib.reload(gort)
importlib.reload(kernels)
from brdfFile import brdfFile
from gort import gort, geom
from kernels import kernelBRDF
import pandas as pd

from matplotlib import pyplot as plt 

def geom_list_from_brdfFile(b):
    geom_list = []
    for i in range(b.nAngles):
        geom_list.append(geom(vza=b.vza_arr[i], vaa=b.vaa_arr[i], sza=b.sza_arr[i], saa=b.saa_arr[i]))
    return geom_list

def add_obs_brfs_to_kernelBRDF(brfs,k):
    """note - it is entirely on the user to make sure the 
    number of samples in the brfs matches the wavelengths
    and angles in the brdfFile object
    """

    for i in range( k.nAngles ):
        for j in range( k.nWavelengths ):
            k.brfObs[i][j]=brfs[i][j]

    return k


def compare_bs_albedos(sza,g,k):

    weights=k.solveKernelBRDF()

    geom_list=[]
    geom_list.append(geom(vza=0, vaa=0, sza=sza, saa=saa))
    brfs, comp, enrg = g.run(geom_list, verbose=False)

    #get the kernel albedo predictions
    kbs=k.predictBSAlbedoRTkLSp(weights,sza)
    
    gort_alb=[]
        
    ##print the gort albedos:
    for i in range(len(g.wavl)):
        gort_alb.append(enrg[0,i*3])
        #print(enrg[0,i*3],kbs[i])
    
    for w in range( k.nWavelengths ):
        plt.plot(gort_alb[w],kbs[w],"o",label="%f"%k.wavelengths[w])
    #draw a 1:1 line
    ax=plt.gca()
    xlims=ax.get_xlim()
    ylims=ax.get_ylim()
    plt.plot([0,1],[0,1],"--")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
        
    plt.ylabel("RTk LSp bs albedo")
    plt.xlabel("GORT bs albedo")
    plt.legend()
    plt.show()

def compare_bs_albedos_nat(gort_alb,k):

    #get the kernel albedo predictions

    kbs = np.empty_like(gort_alb)
    
    for w in range( k.nAngles ):
        kbs[w,:]=k.predictBSAlbedoRTkLSp(weights,np.deg2rad(k.sza_arr[w]))
    for j in range( k.nWavelengths ):
        plt.plot(np.transpose(gort_alb)[j],np.transpose(kbs)[j],"o",label="%f"%k.wavelengths[j])

    print('kbs',kbs,np.shape(kbs))
    #draw a 1:1 line
    ax=plt.gca()
    xlims=ax.get_xlim()
    ylims=ax.get_ylim()
    plt.plot([0,1],[0,1],"--")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
        
    plt.ylabel("RTk LSp bs albedo")
    plt.xlabel("GORT bs albedo")
    plt.legend()
    plt.show()

def compare_brfs(k):
    
    obs_brf=np.asarray(k.brfObs)
    opt_brf=np.asarray(k.brf)
    
    for w in range( k.nWavelengths ):
        plt.plot(opt_brf[:,w],obs_brf[:,w],"o",label="%f"%k.wavelengths[w])
    #draw a 1:1 line
    ax=plt.gca()
    xlims=ax.get_xlim()
    ylims=ax.get_ylim()
    plt.plot([0,1],[0,1],"--")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
        
    plt.xlabel("RTLS optimised BRF")
    plt.ylabel("GORT observed BRF")
    plt.legend()
    
    #plt.savefig('/home/users/ndouglas/TRUTHS/images/brf_scatter.png')
    plt.show()

def plot_brfs(TRUTHS_times,Sentinel2_times,TRUTHS_albedos,Sentinel2_albedos,wl):
    
    df_TRUTHS = pd.DataFrame(TRUTHS_albedos, columns=["albedo"])
    df_TRUTHS["time"] = TRUTHS_times#pd.to_datetime(times)
    df_TRUTHS = df_TRUTHS.set_index("time")

    df_Sentinel2 = pd.DataFrame(Sentinel2_albedos, columns=["albedo"])
    df_Sentinel2["time"] = Sentinel2_times#pd.to_datetime(times)
    df_Sentinel2 = df_Sentinel2.set_index("time")
    
    colors=['r','k','b','g','k','k','k','k','k','k','k']
    markers=['o','x','o','o','o','1','2','o','o','o','o']

    fig, ax = plt.subplots()
    ax.plot(df_TRUTHS.index, df_TRUTHS.values, linestyle="-", marker="o", label="TRUTHS")
    ax.plot(df_Sentinel2.index, df_Sentinel2.values, linestyle="-", marker="s", label="Sentinel-2")
    ax.set_title(f"Albedo time series at {wl} nm")
    ax.set_xlabel("Time")
    ax.set_ylabel("Albedo")
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    #plt.savefig('/home/users/ndouglas/TRUTHS/images/brf_timeseries.png')
    plt.show()
    

if __name__=="__main__":

    filename='/home/users/ndouglas/TRUTHS/Geometries/TRUTHS/TRUTHSgeometries/TRUTHSgeomsLAT0.0LON-60.0.brdf'

    k=kernelBRDF( )
    k.readBRDF(filename)
    geom_list=geom_list_from_brdfFile(k)
    
    g=gort() 
    g.params["LAI"]=5.0
    g.params["PCC"]=0.7
    g.params["HB"]=2.0
    g.params["BR"]=1.0
    g.wavl=copy(k.wavelengths)    
    gort_brfs, comp, enrg = g.run(geom_list, verbose=False)

    gort_alb=enrg[:,::3]
    
    sza=25.

    k=add_obs_brfs_to_kernelBRDF(gort_brfs,k)   
    weights=k.solveKernelBRDF()

    k.predict_brfs(weights)


    # Get datetime for associated lat/lon choice
    lat=0.0
    lon=-60.0

    geometries_filepath='/home/users/ndouglas/TRUTHS/albedos/LAT'+str(lat)+'LON'+str(lon)+'/LAT'+str(lat)+'LON'+str(lon)+'_geometries.csv'
    albedos_filepath='/home/users/ndouglas/TRUTHS/albedos/LAT'+str(lat)+'LON'+str(lon)+'/LAT'+str(lat)+'LON'+str(lon)+'_albedos.csv'

    df_geom = pd.read_csv(geometries_filepath, dtype={"datetime": str})
    df_geom["datetime"] = pd.to_datetime(df_geom["datetime"].str.strip(), errors="raise")
    TRUTHS_times = (df_geom.loc[df_geom["mission"].eq("TRUTHS"), "datetime"].tolist())
    Sentinel2_times = (df_geom.loc[df_geom["mission"].eq("Sentinel2"), "datetime"].tolist())    
    
    wavelength="492.4"
    df_alb = pd.read_csv(albedos_filepath)
    TRUTHS_albedos= (df_alb.loc[df_alb["mission"].eq("TRUTHS"), wavelength].tolist())
    Sentinel2_albedos = (df_alb.loc[df_alb["mission"].eq("Sentinel2"), wavelength].tolist())

    plot_brfs(TRUTHS_times,Sentinel2_times,TRUTHS_albedos,Sentinel2_albedos,wavelength)

