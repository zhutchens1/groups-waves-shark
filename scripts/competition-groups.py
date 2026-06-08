import sys 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from copy import deepcopy
from astropy.table import Table
from astropy.io import fits
import pickle

home = '/Users/zhutchens/Documents/Research/4HS/'
sys.path.insert(0,home+'g3code/')
from g3group import g3groupfinder as g3gf
from survey_volume import comoving_volume_shell, solid_angle_str as solid_angle
from purity_compl import get_metrics_by_halo, get_metrics_by_group

mock_filename = home+'catalogs/shark_galaxies_comp.parquet'
output_file = home+'competition/waves-competition-groups.fits'
galaxytable = home+'competition/evaluation_galaxytable_g3.parquet'
grouptable = home+'competition/evaluation_grouptable_g3.parquet'
H0 = 70. 
Om0 = 0.3 
Ode0 = 0.7 
showplots = True

# ------------------------------------------------------------- #
# ------------------------------------------------------------- #
def createEvalCatalogs(df, group_savepath, galaxy_savepath):
    id_group = deepcopy(df.loc[:,'grpid'].to_numpy())
    id_group[(df.grpn == 1) | (df.grpid<0)] = -1
    df.loc[:,'id_group'] = id_group
    galaxy_cat = df[['id_group']]
    galaxy_cat.index.name = 'id_galaxy'
    #galaxy_cat.rename({'id_galaxy_sky' : 'id_galaxy'}, axis=1, inplace=True)

    group_cat = df[df.id_group>0].groupby('id_group').first()
    group_cat = group_cat[['grpn','grpra','grpdec','grpz']]
    group_cat.rename({'grpn' : 'number_members'},axis=1,inplace=True)
    group_cat.to_parquet(group_savepath, engine='pyarrow', index=True)
    galaxy_cat.to_parquet(galaxy_savepath, engine='pyarrow', index=True)
    return galaxy_cat, group_cat

def getCompleteness(zz, Mr):
    zz=np.asarray(zz)
    Mr=np.asarray(Mr)
    return np.percentile(Mr[(zz > 0.98*zz.max())],99)

def g3(catalog, sbar, floor, params, gd_fit_bins=None, starting_id=None, plot=False, handle=None):
    bperp = params['bperp']    
    blos = params['blos']    
    rproj_mult = params['rproj_mult']    
    vproj_mult = params['vproj_mult']    
    vproj_offs = params['vproj_offs']    
    gd_rproj_mult = params['gd_rproj_mult']    
    gd_vproj_mult = params['gd_vproj_mult']    
    gd_vproj_offs = params['gd_vproj_offs']    

    catalog_g3obj = g3gf(catalog.ra, catalog.dec, catalog.redshift_observed, catalog.mag_abs_r_SDSS, floor, H0, Om0, Ode0, precision=np.float64)
    fofid = catalog_g3obj.giantOnlyFOF(bperp, blos, fof_sep=sbar)
    catalog_g3obj.deriveGiantCalibrations(rproj_fit_multiplier=rproj_mult, vproj_fit_multiplier=vproj_mult, vproj_fit_offset=vproj_offs, n_bootstraps=1000)
    if plot: 
        catalog_g3obj.plotGiantGroupBoundaries(show=True, rproj_ylim=1.5, vproj_ylim=1500, savepath=handle+'-giantcal.png')
    catalog_g3obj.giantOnlyMerging()
    catalog_g3obj.dwarfAssoc()
    if gd_fit_bins is None:
        gd_fit_bins=np.arange(-24,-19,0.25)
    catalog_g3obj.deriveDwarfBoundaries(gd_rproj_fit_multiplier=gd_rproj_mult, gd_vproj_fit_multiplier=gd_vproj_mult, gd_vproj_fit_offs=gd_vproj_offs,\
         gd_fit_bins=gd_fit_bins)
    if plot: catalog_g3obj.plotDwarfBoundaries(show=True, rproj_xlim=(-17,-24), rproj_ylim=(0,0.8), vproj_xlim=(-17,-24), vproj_ylim=(0,800),\
                                               savepath = handle+'dwarfcal.png')
    catalog_g3obj.findDwarfOnlyGroups()
    if starting_id is not None:
         catalog_g3obj.g3grpid = catalog_g3obj.g3grpid + float(starting_id)
    catalogcat = pd.DataFrame(catalog_g3obj.getCatalog(by='galaxy'), columns=['grpid','grpra','grpdec','grpz','grpn','grpabsrmag']).set_index(catalog.index)
    return catalogcat

# ------------------------------------------------------------- #
# ------------------------------------------------------------- #

if __name__ == "__main__":
    # Read catalog, set haloID
    df_full = pd.read_parquet(mock_filename).set_index('id_galaxy_sky')

    # Flag footprints
    in_footprint1 = (df_full.loc[:,'ra'] < 60) 
    in_footprint2 = (df_full.loc[:,'ra'] > 60) & (df_full.loc[:,'ra'] < 300)
    in_footprint3 = (df_full.loc[:,'ra'] > 300)
    footprint = 1*in_footprint1 + 2*in_footprint2 + 3*in_footprint3
    df_full.loc[:,'footprint'] = footprint
    
    # Select objects for group-finding
    observed = df_full.observed & ~df_full.ghosted & (df_full.mag_abs_r_SDSS > -99)
    detected_wide = (df_full.mag_Z_VISTA < 21.1) & (df_full.footprint <= 2)
    detected_deep = (df_full.mag_Z_VISTA < 21.25) & (df_full.footprint == 3) & (df_full.dec < -30) & (df_full.dec > -35) & (df_full.ra > 339) & (df_full.ra < 351)
    sel = observed & (detected_wide | detected_deep)
    df = df_full[sel]

    # Construct dataframes representing WAVES-Wide and WAVES-Deep subvolumes
    waves_wide_sel = (df.footprint != 3) & (df.redshift_observed <= 0.2)
    waves_wide = df[waves_wide_sel]

    deep_sv1_sel = (df.footprint == 3) & (df.redshift_observed <=0.2)
    deep_sv2_sel = (df.footprint == 3) & (df.redshift_observed > 0.2) & (df.redshift_observed <= 0.4)
    deep_sv3_sel = (df.footprint == 3) & (df.redshift_observed > 0.4) & (df.redshift_observed <= 0.6)
    deep_sv4_sel = (df.footprint == 3) & (df.redshift_observed > 0.6) & (df.redshift_observed <= 0.8)
    deep_sv1 = df[deep_sv1_sel]
    deep_sv2 = df[deep_sv2_sel]
    deep_sv3 = df[deep_sv3_sel]
    deep_sv4 = df[deep_sv4_sel]

    # Determine Mr completeness limits
    waves_wide_floor = getCompleteness(waves_wide.redshift_observed, waves_wide.mag_abs_r_SDSS)
    deep_sv1_floor = getCompleteness(deep_sv1.redshift_observed, deep_sv1.mag_abs_r_SDSS)
    deep_sv2_floor = getCompleteness(deep_sv2.redshift_observed, deep_sv2.mag_abs_r_SDSS)
    deep_sv3_floor = getCompleteness(deep_sv3.redshift_observed, deep_sv3.mag_abs_r_SDSS)
    deep_sv4_floor = getCompleteness(deep_sv4.redshift_observed, deep_sv4.mag_abs_r_SDSS)

    dwarfgiant = -19.5
    waves_wide_floor = waves_wide_floor if (waves_wide_floor <= dwarfgiant) else dwarfgiant
    deep_sv1_floor = deep_sv1_floor if (deep_sv1_floor <= dwarfgiant) else dwarfgiant
    deep_sv2_floor = deep_sv2_floor if (deep_sv2_floor <= dwarfgiant) else dwarfgiant
    deep_sv3_floor = deep_sv3_floor if (deep_sv3_floor <= dwarfgiant) else dwarfgiant
    deep_sv4_floor = deep_sv4_floor if (deep_sv4_floor <= dwarfgiant) else dwarfgiant


    # Determine volumes
    wide1 = waves_wide[waves_wide.footprint==1]    
    wide2 = waves_wide[waves_wide.footprint==2]        
    
    wide1SA = solid_angle(wide1.ra, wide1.dec)
    wide2SA = solid_angle(wide2.ra, wide2.dec)
    wideSA = wide1SA + wide2SA
    deepSA = solid_angle(deep_sv1.ra, deep_sv2.ra)

    waves_wide_vol = wideSA * comoving_volume_shell(waves_wide.redshift_observed.min(),waves_wide.redshift_observed.max(),H0,Om0,Ode0)
    deep_sv1_vol = deepSA * comoving_volume_shell(deep_sv1.redshift_observed.min(),deep_sv1.redshift_observed.max(),H0,Om0,Ode0)
    deep_sv2_vol = deepSA * comoving_volume_shell(deep_sv2.redshift_observed.min(),deep_sv2.redshift_observed.max(),H0,Om0,Ode0)
    deep_sv3_vol = deepSA * comoving_volume_shell(deep_sv3.redshift_observed.min(),deep_sv3.redshift_observed.max(),H0,Om0,Ode0)
    deep_sv4_vol = deepSA * comoving_volume_shell(deep_sv4.redshift_observed.min(),deep_sv4.redshift_observed.max(),H0,Om0,Ode0)

    # Determine number densities and mean separations
    waves_wide_n = len(waves_wide[waves_wide.mag_abs_r_SDSS <= waves_wide_floor]) / waves_wide_vol
    deep_sv1_n = len(deep_sv1[deep_sv1.mag_abs_r_SDSS <= deep_sv1_floor]) / deep_sv1_vol
    deep_sv2_n = len(deep_sv2[deep_sv2.mag_abs_r_SDSS <= deep_sv2_floor]) / deep_sv2_vol
    deep_sv3_n = len(deep_sv3[deep_sv3.mag_abs_r_SDSS <= deep_sv3_floor]) / deep_sv3_vol
    deep_sv4_n = len(deep_sv4[deep_sv4.mag_abs_r_SDSS <= deep_sv4_floor]) / deep_sv4_vol

    waves_wide_s = waves_wide_n ** (-1/3.)
    deep_sv2_s = deep_sv2_n ** (-1/3.)
    deep_sv3_s = deep_sv3_n ** (-1/3.)
    deep_sv4_s = deep_sv4_n ** (-1/3.)

    # Run group-finding
    wide_sv1_params = dict(bperp=0.08, blos=1.2, rproj_mult=3., vproj_mult=4, vproj_offs=200, gd_rproj_mult=4, gd_vproj_mult=4, gd_vproj_offs=100)
    waves_wide_groups = g3(waves_wide, waves_wide_s, waves_wide_floor, wide_sv1_params, np.arange(-25,deep_sv1_floor,0.5), None, showplots, home+'competition/figs/waves_wide')
    deep_sv1_groups = g3(deep_sv1, waves_wide_s, deep_sv1_floor, wide_sv1_params, np.arange(-25,deep_sv1_floor,0.5), 1+waves_wide_groups.grpid.max(), showplots, home+'competition/figs/deep_sv1')

    deep_sv2_params = dict(bperp=0.04, blos=1.0, rproj_mult=3., vproj_mult=4, vproj_offs=200, gd_rproj_mult=4, gd_vproj_mult=4, gd_vproj_offs=100)
    deep_sv2_groups = g3(deep_sv2, deep_sv2_s, deep_sv2_floor, deep_sv2_params, np.arange(-25,deep_sv2_floor,0.5), 1+deep_sv1_groups.grpid.max(), showplots, home+'competition/figs/deep_sv2')

    deep_sv3_params = dict(bperp=0.04, blos=1.0, rproj_mult=3, vproj_mult=2, vproj_offs=500, gd_rproj_mult=4, gd_vproj_mult=4, gd_vproj_offs=100)
    deep_sv3_groups = g3(deep_sv3, deep_sv3_s, deep_sv3_floor,deep_sv3_params, np.arange(-25,deep_sv3_floor,0.5), 1+deep_sv2_groups.grpid.max(), showplots, home+'competition/figs/deep_sv3')

    deep_sv4_params = dict(bperp=0.03, blos=1, rproj_mult=3, vproj_mult=2, vproj_offs=500, gd_rproj_mult=4, gd_vproj_mult=4, gd_vproj_offs=100)
    deep_sv4_groups = g3(deep_sv4, deep_sv4_s, deep_sv4_floor, deep_sv4_params, np.arange(-25,deep_sv4_floor,0.5), 1+deep_sv3_groups.grpid.max(), showplots, home+'competition/figs/deep_sv4')

    waves_wide_groups.loc[:,'subvol'] = 'wide'
    deep_sv1_groups.loc[:,'subvol'] = 'deep_sv1'
    deep_sv2_groups.loc[:,'subvol'] = 'deep_sv2'
    deep_sv3_groups.loc[:,'subvol'] = 'deep_sv3'
    deep_sv4_groups.loc[:,'subvol'] = 'deep_sv4'

    # Merge group catalogs
    all_groups=pd.concat([waves_wide_groups, deep_sv1_groups, deep_sv2_groups, deep_sv3_groups, deep_sv4_groups], axis=0)
    assert len(all_groups) == (len(waves_wide_groups) + len(deep_sv1_groups) + len(deep_sv2_groups) + len(deep_sv3_groups) + len(deep_sv4_groups))

    # Recombine with inputs
    df_full.loc[all_groups.index,all_groups.keys()] = all_groups
    df_full.loc[df_full.subvol.isna(),'subvol'] = 'na'
    df_full.fillna(-1,inplace=True)

    # Save master file
    table=Table.from_pandas(df_full)
    table.write(output_file,format='fits',overwrite=True)
   
    # Save galaxy and group tables for evaluation
    createEvalCatalogs(df_full, grouptable, galaxytable)
