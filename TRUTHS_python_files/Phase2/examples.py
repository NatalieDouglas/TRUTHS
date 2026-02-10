import numpy as np
from copy import copy

import importlib
import brdfFile
import gort
import seaborn as sns
import semidisc
import kernels
import fourDEnVar_engine_gamma
importlib.reload(brdfFile)
importlib.reload(gort)
importlib.reload(semidisc)
importlib.reload(kernels)
importlib.reload(fourDEnVar_engine_gamma)
from brdfFile import brdfFile
from gort import gort, geom
from semidisc import semiD
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
    
    #black sky albedo
    sza=np.deg2rad(sza)
    saa=np.deg2rad(0.)

    #have to run gort at the specific solar angle 
    geom_list=[]
    geom_list.append(geom(vza=0, vaa=0, sza=sza, saa=saa))
    brfs, comp, enrg = g.run(geom_list, verbose=False)

    #get the kernel albedo predictions
    kbs=k.predictBSAlbedoRTkLSp(weights,sza)
        
    gort_alb=[]
        
    #print the gort albedos:
    for i in range(len(g.wavl)):
        gort_alb.append(enrg[0,i*3])
        print(enrg[0,i*3],kbs[i])
    
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



def compare_brfs(k,xb,Xb,xa,Xa):
    
    obs_brf=np.asarray(k.brfObs)
    opt_brf=np.asarray(k.brf)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(15, 5))
    for w in range( k.nWavelengths ):
        ax1.plot(obs_brf[:,w],opt_brf[:,w],"o",label="%f"%k.wavelengths[w])
    #draw a 1:1 line
    #ax1=plt.gca()
    xlims=ax1.get_xlim()
    ylims=ax1.get_ylim()
    ax1.plot([0,1],[0,1],"--")
    ax1.set_xlim(xlims)
    ax1.set_ylim(ylims)
        
    ax1.set_ylabel("retrieved BRF")
    ax1.set_xlabel("simulated BRF")
    ax1.legend()

    weights = np.ones(20) / float(20)
    palette=sns.color_palette("colorblind", 11)
    ax2.axvline(xb[0], color=palette[6], linestyle='--',label='prior mean')
    ax2.axvline(5.0, color='k', linestyle='--',label='truth') 
    sns.histplot(Xb[0,:], kde=True, color=palette[6], stat='density',kde_kws=dict(cut=3),edgecolor=(1, 1, 1, .4),bins=4,label='prior dist.',ax=ax2)
    sns.histplot(Xa[0,:], kde=True, color=palette[2], stat='density',kde_kws=dict(cut=3),edgecolor=(1, 1, 1, .4),bins=4,label='post. dist.',ax=ax2)
    ax2.axvline(xa[0], color=palette[2], linestyle='--',label='posterior')
    ax2.set_xlabel('LAI')
    ax2.legend()

    ax3.axvline(xb[1], color=palette[6], linestyle='--',label='prior mean')
    ax3.axvline(80.0, color='k', linestyle='--',label='truth') 
    sns.histplot(Xb[1,:], kde=True, color=palette[6], stat='density',kde_kws=dict(cut=3),edgecolor=(1, 1, 1, .4),bins=4,label='prior dist.',ax=ax3)
    sns.histplot(Xa[1,:], kde=True, color=palette[2], stat='density',kde_kws=dict(cut=3),edgecolor=(1, 1, 1, .4),bins=4,label='post. dist.',ax=ax3)
    ax3.axvline(xa[1], color=palette[2], linestyle='--',label='posterior')
    ax3.set_xlabel(r'$C_{ab}$')
    ax3.legend()
    #plt.savefig('/home/users/ndouglas/TRUTHS/images/brf_scatter.png')
    fig.tight_layout()
    plt.show()

def plot_brfs(k,obs_model,opt_model,list_datetime):
    
    obs_brf=np.asarray(k.brfObs)
    opt_brf=np.asarray(k.brf)
   
    print(" info", np.shape(obs_brf),obs_brf)
    print(" info", np.shape(opt_brf),opt_brf)

    times=list_datetime
    colors=['r','k','b','g','k','k','k']
    markers=['o','x','o','o','o','1','2']
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12, 6))
    for w in range( k.nWavelengths ):
        ax1.plot(times,obs_brf[:,w],marker=markers[w],label="%d"%int(k.wavelengths[w]),color=colors[w])
    ax1.set_ylabel(obs_model+" BRF")
    ax1.set_xlabel("Date")
    ax1.tick_params(axis='x', labelrotation=60)
    ax1.legend()
    
    for w in range( k.nWavelengths ):   
        ax2.plot(times,opt_brf[:,w],marker=markers[w],label="%d"%int(k.wavelengths[w]),color=colors[w])
        
    ax2.set_ylabel(" BRFs")
    ax2.set_xlabel("Date")
    ax2.tick_params(axis='x', labelrotation=60)
    ax2.legend()
    plt.tight_layout()
    
    #plt.savefig('/home/users/ndouglas/TRUTHS/images/brf_timeseries.png')
    plt.show()
    
def fourDEnVar(k,geom_list,sigma_array):

    obs_brfs=np.asarray(k.brfObs)

    #DA_brfs=np.zeros(np.shape(obs_brfs)) 
    gamma=1
    #sigma=0.1
    xb=np.load('/home/users/ndouglas/TRUTHS/Data_Assimilation_files/xb_semid_20.npy')
    Xb=np.load('/home/users/ndouglas/TRUTHS/Data_Assimilation_files/Xb_semid_20.npy')
    print('std dev of Xb',np.std(Xb[:, 0], ddof=1))

    #print('k.brfObs',k.brfObs, np.shape(k.brfObs))
    y=np.concatenate(np.array(k.brfObs))
    #y = np.array(k.brfObs).T.reshape(-1)
    #print('y',y,np.shape(y))
    #obs_stddev=sigma*y
    #R=np.diag(np.square(obs_stddev))
    R=np.diag(np.concatenate(np.square(sigma_array)))
    #R=np.diag(np.array(sigma_array).T.reshape(-1))
    #print(np.shape(y))
    
    sd=semiD()    
    sd.params['LAI']=xb[0]
    sd.params['cab']=xb[1]
    sd.wvl=copy(k.wavelengths)    
    hxbar = np.concatenate(sd.run(geom_list)) # e.g. 12 by 7, 12 geometries, 7 wavelengths

    hX=[]
    for i in range(0,np.shape(Xb)[1]):  
        sd.params['LAI']=Xb[0,i]
        sd.params['cab']=Xb[1,i]   
        hX.append(np.concatenate(sd.run(geom_list)))
    hX=np.transpose(hX)

    xDA=fourDEnVar_engine_gamma.fourDEnVar_engine_gamma(Xb, hX, y, R, hxbar, gamma)
    sd.params['LAI']=xDA.xa[0]
    sd.params['cab']=xDA.xa[1]
    DA_brfs=sd.run(geom_list)
    print('std dev of Xa',np.std(xDA.Xa[:, 0], ddof=1))

    return DA_brfs, xb, Xb, xDA.xa, xDA.Xa

if __name__=="__main__":
    
    latitude=0.0#40.0
    longitude=-60.0#0.0
    add_noise=True
    eps = 1e-3# Avoid σ=0 when reflectance is 0

    rng = np.random.default_rng(42)
    filepath='/home/users/ndouglas/TRUTHS/Geometries/'
    experiment = 'Sentinel+TRUTHS' # 'TRUTHS','Sentinel', 'Sentinel+TRUTHS' or 'test'
    if experiment == 'Sentinel':
        filename=filepath+'Sentinel/SentinelGeometries/SentinelGeomsLAT'+str(latitude)+'LON'+str(longitude)+'.brdf'
    elif experiment == 'TRUTHS':
        filename=filepath+'TRUTHS/TRUTHSgeometries/TRUTHSgeomsLAT'+str(latitude)+'LON'+str(longitude)+'.brdf'
    elif experiment == 'Sentinel+TRUTHS':
        filename=filepath+'Sentinel+TRUTHS/Sentinel+TRUTHSGeometries/Sentinel+TRUTHSGeomsLAT'+str(latitude)+'LON'+str(longitude)+'.brdf'
    else:
        filename="./HEMItest.brdf"

    k=kernelBRDF( )
    k.readBRDF(filename)
    geom_list=geom_list_from_brdfFile(k)

    sd=semiD()    
    sd.params['LAI']=5.0
    sd.params['cab']=80.0
    sd.wvl=copy(k.wavelengths)    
    semid_brfs = sd.run(geom_list)
    
    Sent_factor=0.2
    TRUTHS_acc=0.003
    rel_err_Sentinel=Sent_factor*np.array([0.0595,0.0413,0.0349,0.0377,0.0356,0.0335,0.0332,0.0335,0.315,0.0355,0.0357])
    rel_err_TRUTHS=TRUTHS_acc*np.ones(11)
    eps = 1e-3
    if experiment == 'Sentinel':
        sigma_arr = rel_err_Sentinel * np.maximum(semid_brfs, eps)
    elif experiment == 'TRUTHS':
        sigma_arr = rel_err_TRUTHS * np.maximum(semid_brfs, eps)
    elif experiment == 'Sentinel+TRUTHS':
        sigma_arr = np.zeros_like(semid_brfs, dtype=float)
        sigma_arr[:33] = rel_err_Sentinel * np.maximum(semid_brfs[:33], eps) # need to update for number of sentinel geoms for each location
        sigma_arr[33:] = rel_err_TRUTHS * np.maximum(semid_brfs[33:], eps) # need to update for number of sentinel geoms for each location
    else:
        sigma_arr=0.0

    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, sigma_arr)
    semid_brfs_noise=semid_brfs+noise
    
    k=add_obs_brfs_to_kernelBRDF(semid_brfs_noise,k) 
    k.brf, xb, Xb, xa, Xa = fourDEnVar(k,geom_list,sigma_arr)

    compare_brfs(k,xb,Xb,xa,Xa)
