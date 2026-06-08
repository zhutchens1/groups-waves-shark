import numpy as np
import sys 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from copy import deepcopy
from astropy.table import Table
from astropy.io import fits
import pickle

from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(70,0.3,0.7)

home = '/Users/zhutchens/Documents/Research/4HS/'
sys.path.insert(0,home+'g3code/')
from kdfof import kdFOF
from purity_compl import *
from g3misc import multiplicity_function

def optimize(df, Mlim, meansep):
    bperp = np.arange(0.02,0.1,0.01)
    blos = np.arange(0.5,1.2,0.1)
    bperp_grid = np.zeros((bperp.size, blos.size))
    blos_grid = np.zeros((bperp.size, blos.size))
    pg_grid = np.zeros((bperp.size, blos.size))
    cg_grid = np.zeros((bperp.size, blos.size))
    ph_grid = np.zeros((bperp.size, blos.size))
    ch_grid = np.zeros((bperp.size, blos.size))
    perform = np.zeros((bperp.size, blos.size))

    nval=np.arange(1,31)
    print('niter :', bperp.size * blos.size)
    df = df[df.mag_abs_r_SDSS <= Mlim]
    for ii,bp in enumerate(bperp):
        for jj,bl in enumerate(blos):
            fofid = kdFOF(df.ra, df.dec, df.redshift_observed, bp, bl, meansep, cosmo)
            df.loc[:,'fofid']=fofid
            df.loc[:,'fofn'] = multiplicity_function(fofid,True)
            df.loc[:,'halon'] = multiplicity_function(df.haloid,True)
            df = df[df.halon>=2]
            pg, cg, _ = get_metrics_by_group(df.fofid, df.haloid, df.mag_abs_r_SDSS)
            ph, ch, _ = get_metrics_by_halo(df.fofid, df.haloid, df.mag_abs_r_SDSS)
            df.loc[:,'pg']=pg
            df.loc[:,'cg']=cg
            df.loc[:,'ph']=ph
            df.loc[:,'ch']=ch
            pg_grid[ii,jj] = np.average(pg, weights=df.fofn)
            cg_grid[ii,jj] = np.average(cg, weights=df.fofn)
            ph_grid[ii,jj] = np.average(ph, weights=df.halon)
            ch_grid[ii,jj] = np.average(ch, weights=df.halon)
            bperp_grid[ii,jj] = bp
            blos_grid[ii,jj] = bl
    total =  ph_grid * ch_grid #* pg_grid * cg_grid 
    optimal = np.unravel_index(np.argmax(total, axis=None), total.shape)
    best_bperp = bperp_grid[optimal]
    best_blos = blos_grid[optimal]
    print('bp: ', best_bperp)
    print('bl: ', best_blos)

# ------------------------------------------------------------- #
# ------------------------------------------------------------- #
if __name__ == '__main__':
    fname = home+'catalogs/giantFOF_optim_data.pkl'
    inputs = pickle.load(open(fname,'rb'))
    frames = inputs['frames']
    floor = inputs['floors']
    sbar = inputs['sbar']

    for ii in range(len(frames)):
        print(ii)
        optimize(frames[ii],floor[ii],sbar[ii])
        print('---------')

   
