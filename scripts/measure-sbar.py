import sys 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
import pickle

home='/Users/zhutchens/Documents/Research/4HS/'
sys.path.insert(0,home+'g3code/')
from survey_volume import comoving_volume_shell, solid_angle_str as solid_angle

mock_filename = home+'catalogs/fibre_incomplete_mocks.parquet'
output_file = home+'catalogs/sbar_model.pkl'
H0 = 70. 
Om0 = 0.3 
Ode0 = 0.7

def completeness_model(x,a,b,c,d):
    return a*np.log10(b*x + c)+d
    #return (a / np.sqrt(b*x + c)) + d
    #return a*np.exp(-np.abs(b)*x + c)+d

if __name__ == '__main__':
    df_full = pd.read_parquet(mock_filename).set_index('id_galaxy_sky')

    # flag footprints
    in_footprint1 = (df_full.loc[:,'ra'] < 60)
    in_footprint2 = (df_full.loc[:,'ra'] > 60) & (df_full.loc[:,'ra'] < 300)
    in_footprint3 = (df_full.loc[:,'ra'] > 300)
    footprint = 1*in_footprint1 + 2*in_footprint2 + 3*in_footprint3
    df_full.loc[:,'footprint'] = footprint
    
    # Select objects for modeling
    lowz = (df_full.redshift_observed <= 0.2)
    observed = df_full.observed & ~df_full.ghosted & (df_full.mag_abs_r_SDSS > -99)
    detected_wide = (df_full.mag_Z_VISTA < 21.1) & (df_full.footprint <= 2)
    detected_deep = (df_full.mag_Z_VISTA < 21.25) & (df_full.footprint == 3)
    sel = lowz & observed & (detected_wide | detected_deep)
    df = df_full[sel]

    zz = df.redshift_observed.to_numpy()
    Mr = df.mag_abs_r_SDSS.to_numpy()

    # calculate solid angles
    wide1 = df[df.footprint==1]    
    wide2 = df[df.footprint==2]    
    deep = df[df.footprint==3]

    wide1_SA = solid_angle(wide1.ra, wide1.dec)
    wide2_SA = solid_angle(wide2.ra, wide2.dec)
    deep_SA = solid_angle(deep.ra, deep.dec)
    SA = wide1_SA + wide2_SA + deep_SA

    # set up redshift bins and find volumes
    zbin = np.linspace(0.005,0.2,10)
    
    zmax = zbin[1:]
    zmin = zbin[:-1]
    zbinc = 0.5*(zmin+zmax)
    buff = 0.002

    vol = np.zeros(len(zmin))
    floor = np.zeros(len(zmin))
    ndens = np.zeros(len(zmin))
    for i in range(len(zmin)):
        vol[i] = SA * comoving_volume_shell(zmin[i],zmax[i],H0,Om0,Ode0)
        floor[i] = np.percentile(Mr[(zz > (zmax[i]-buff)) & (zz < (zmax[i]+buff))],98)
        sel = (zz > zmin[i]) & (zz < zmax[i]) & (Mr <= floor[i])
        ndens[i] = np.sum(sel) / vol[i]
    sbar = ndens**(-1/3.)

    # make model for s(z)
    popt = np.polyfit(zbinc, sbar, 1)
    def smodel(x):
         return np.polyval(popt, x)

    popt2 = np.polyfit(floor, sbar, 1)
    def smodel2(x):
         return np.polyval(popt2, x)

    # plot everything
    tx = np.linspace(0.005,0.8,1000)
    fig,axs=plt.subplots(ncols=3,figsize=(11,4))
    axs[0].scatter(zz, Mr, s=1, alpha=0.5)
    axs[0].plot(zmax, floor, 'o',color='k',markersize=4)
    axs[0].invert_yaxis()
    axs[0].set_xlabel("$z$")
    axs[0].set_ylabel("$M_r$")
   
    axs[1].plot(floor, sbar, 'ko')
    axs[1].plot(floor, smodel2(floor),color='tab:blue')
    axs[1].set_xlabel(r'$M_r$ Floor')
    axs[1].set_ylabel(r'$\bar{s}$ [Mpc]')
    axs[1].invert_xaxis()

    zlin=np.linspace(0,0.8,10) 
    axs[2].plot(zbinc, sbar, 'ko')
    axs[2].plot(zlin, smodel(zlin),color='tab:blue')
    axs[2].set_xlabel('$z$')
    axs[2].set_ylabel(r'$\bar{s}$ [Mpc]')
    plt.tight_layout()
    plt.savefig(home+"figures/sbarmodel.png",dpi=300)
    plt.show()

    print(popt2)

