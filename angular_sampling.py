import numpy as np
import matplotlib.pyplot as plt

from brdfFile import brdfFile
from gort import gort, geom
from kernels import kernelBRDF
from utils import *


def mk_angluar_sampling_fig(geom_tr,geom_s2,latlon,fn=None):
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={'projection': 'polar'},
                            layout='constrained')
    
    
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rmax(80)
    
    ax.set_rlim(0,80)
    ax.set_rticks([20, 40, 60, 80])  # Fewer radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)

    #label_position=ax.get_rlabel_position()
    #ax.text(np.radians(label_position+10),ax.get_rmax()/1.5,'Solar Zenith Angle',
    #    rotation=292.5,ha='center',va='center')

    #checking:
    #ax.plot(np.deg2rad(45),60,"g*")

    label="TRUTHS"
    for g in geom_tr:
        raa=np.deg2rad(g.saa-g.vaa)
        ax.plot(raa,g.sza,"ro",label=label)
        label=None

    label="S2"
    for g in geom_s2:
        raa=np.deg2rad(g.saa-g.vaa)
        ax.plot(raa,g.sza,"bo",label=label)
        label=None

    ax.legend()
    ax.set_title("Angular sampling at %s"%latlon, va='bottom')

    if fn is None:
        plt.show()
    else:
        plt.savefig(fn)
    
if __name__ == "__main__":
    
    from glob import glob
        
    file_list=glob("brdf_files/TRUTHS/TRUTHSgeoms*.brdf")
    
    for f in file_list:
        
        f=f.replace(".brdf","")
        latlon=f.replace("brdf_files/TRUTHS/TRUTHSgeoms","")
        
        outfn="polar_plots/angular_sampling_%s.png"%latlon
                     
        filename_tr="brdf_files/TRUTHS/TRUTHSgeoms%s.brdf"%latlon    
        filename_s2="brdf_files/Sentinel2/SentinelGeoms%s.brdf"%latlon    

        #truths
        k=kernelBRDF( )
        k.readBRDF(filename_tr)
        geom_tr=geom_list_from_brdfFile(k)
        
        #sentinel 2
        k=kernelBRDF( )
        k.readBRDF(filename_s2)
        geom_s2=geom_list_from_brdfFile(k)
        
        latlon=latlon.replace("LON"," LON=")
        latlon=latlon.replace("LAT","LAT=")
        mk_angluar_sampling_fig(geom_tr,geom_s2,latlon,outfn)

