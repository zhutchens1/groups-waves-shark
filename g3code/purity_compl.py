import numpy as np
from scipy.stats import mode as scipy_mode
from tqdm import tqdm

def get_metrics_by_halo(grpid, haloid, galproperty):
    """
    For a group catalog constructed from a mock galaxy catalog,
    compute galaxy-wise purity and completeness metrics using
    true halo IDs for comparison. This function computes metrics
    on a halo-by-halo basis.

    Parameters
    ---------------------------
    groupid : iterable
        Group ID numbers after applying group-finding algorithm, length = # galaxies.
        The group ID should be set to a negative value if the galaxy was not included
        in group-finding, e.g. due to incompleteness.
    haloid : iterable
        Halo ID numbers extracted from mock catalog halos, length = # galaxies = len(groupid).
    galproperty : iterable
        Group property by which to determine the central galaxy in the group. If all values are
        negative, galproperty is assumed to be a abs. magnitude, and the central will be the
        brightest galaxy. If all values are positive, this value is assumed to be mass,
        and the central will be selected by the maximum.

    Returns
    ---------------------------
    Suppose we map halos to groups, and define
        - N_g as the number of galaxies in the group
        - N_h as the number of galaxies in the corresponding true halo
        - N_s as the number of galaxies in the group that are correctly classified as
            members of the corresponding halo
        - N_i as the number of interlopers in the group; galaxies classified to the group
            but that do not belong to the true halo.
    
    purity : np.array
        At index `i`, purity of the group to which galaxy `i` belongs (duplicated for every
        group member). Purity is defined as the percentage of the number of galaxies in the group
        that are correctly identified as part of the halo, N_s/N_g. Because N_g = N_s + N_i,
        the contamination fraction is given by 1 - purity = (N_g - N_s)/N_g = (N_i)/N_g. 
    completeness : np.array
        At index `i`, completeness of the group to which galaxy `i` belongs (duplicated for every
        group member). Completeness is defined as the percentage of galaxies in the halo that are 
        correctly identified as part of the group, N_s/N_h. 
    """
    grpid = np.asarray(grpid)
    haloid = np.asarray(haloid)
    galproperty = np.asarray(galproperty)
    assert len(grpid)==len(haloid) and len(haloid)==len(galproperty),\
        "Inputs must be equivalent length."

    if (galproperty>-30).all() and (galproperty<0).all():
        galproperty = -1 * galproperty # abs mag
    elif (galproperty>0).all():
        pass # mass
    else:
        raise ValueError("Could not determine a sorting factor based on `galproperty` inputs.")

    unique_haloIDs, haloidx, Nh = np.unique(haloid, return_inverse=True, return_counts=True)
    unique_grpIDs, grpidx = np.unique(grpid, return_inverse=True)
    gal_inds = np.arange(len(haloid))
    order_h = np.argsort(haloidx)
    halo_gals = np.split(gal_inds[order_h], np.cumsum(Nh)[:-1])
    
    Ng_all = np.bincount(grpidx)
    order_g = np.argsort(grpidx)
    group_gals = np.split(gal_inds[order_g], np.cumsum(Ng_all)[:-1])

    Ns = np.zeros(unique_haloIDs.size)
    Ng = np.ones(unique_haloIDs.size) * np.nan
    matches = np.zeros(unique_haloIDs.size) - 99.
    for ii, gals in tqdm(enumerate(halo_gals),total=len(halo_gals)):
        matched_grpid = grpid[gals]
        valid = (matched_grpid >= 0)
        if not valid.any():
            continue
        gg = mode(matched_grpid[valid],galproperty[gals][valid])
        gg_idx = np.searchsorted(unique_grpIDs, gg)
        Ns[ii] = np.sum(matched_grpid == gg)
        Ng[ii] = group_gals[gg_idx].size
        matches[ii] = gg
    purity = (Ns / Ng)[haloidx]
    compl = (Ns / Nh)[haloidx]
    matches = matches[haloidx]
    return purity, compl, matches

def get_metrics_by_group(grpid, haloid, galproperty):
    """
    For a group catalog constructed from a mock galaxy catalog,
    compute galaxy-wise purity and completeness metrics using
    true halo IDs for comparison. This function computes metrics
    on a group-by-group basis.

    Parameters
    ---------------------------
    groupid : iterable
        Group ID numbers after applying group-finding algorithm, length = # galaxies.
        The group ID should be set to a negative value if the galaxy was not included
        in group-finding, e.g. due to incompleteness.
    haloid : iterable
        Halo ID numbers extracted from mock catalog halos, length = # galaxies = len(groupid).
    galproperty : iterable
        Group property by which to determine the central galaxy in the group. If all values are
        negative, galproperty is assumed to be a abs. magnitude, and the central will be the
        brightest galaxy. If all values are positive, this value is assumed to be mass,
        and the central will be selected by the maximum.

    Returns
    ---------------------------
    Suppose we map groups to halos, and define
        - N_g as the number of galaxies in the group
        - N_h as the number of galaxies in the corresponding true halo
        - N_s as the number of galaxies in the group that are correctly classified as
            members of the corresponding halo
        - N_i as the number of interlopers in the group; galaxies classified to the group
            but that do not belong to the true halo.
    
    purity : np.array
        At index `i`, purity of the group to which galaxy `i` belongs (duplicated for every
        group member). Purity is defined as the percentage of the number of galaxies in the group
        that are correctly identified as part of the halo, N_s/N_g. Because N_g = N_s + N_i,
        the contamination fraction is given by 1 - purity = (N_g - N_s)/N_g = (N_i)/N_g. 
    completeness : np.array
        At index `i`, completeness of the group to which galaxy `i` belongs (duplicated for every
        group member). Completeness is defined as the percentage of galaxies in the halo that are 
        correctly identified as part of the group, N_s/N_h. 
    """
    grpid = np.asarray(grpid)
    haloid = np.asarray(haloid)
    galproperty = np.asarray(galproperty)
    assert len(grpid)==len(haloid) and len(haloid)==len(galproperty),\
        "Inputs must be equivalent length."

    if (galproperty>-30).all() and (galproperty<0).all():
        galproperty = -1 * galproperty # abs mag
    elif (galproperty>0).all():
        pass # mass
    else:
        raise ValueError("Could not determine a sorting factor based on `galproperty` inputs.")

    unique_grpIDs, grpidx, Ng = np.unique(grpid, return_inverse=True, return_counts=True)
    unique_haloIDs, haloidx = np.unique(haloid, return_inverse=True)
    gal_inds = np.arange(len(grpid))

    order_g = np.argsort(grpidx)
    group_gals = np.split(gal_inds[order_g], np.cumsum(Ng)[:-1])
    
    Nh_all = np.bincount(haloidx)
    order_h = np.argsort(haloidx)
    halo_gals = np.split(gal_inds[order_h], np.cumsum(Nh_all)[:-1])

    Ns = np.zeros(unique_grpIDs.size)
    Nh = np.ones(unique_grpIDs.size)
    matches = np.zeros(unique_grpIDs.size) - 99.
    for ii, gals in tqdm(enumerate(group_gals),total=len(group_gals)):
        gg = unique_grpIDs[ii]
        if gg < 0:
            continue 
        matched_haloid = haloid[gals]
        hh = mode(matched_haloid,galproperty[gals])
        hh_idx = np.searchsorted(unique_haloIDs, hh)
        Ns[ii] = np.sum(matched_haloid == hh)
        Nh[ii] = halo_gals[hh_idx].size
        matches[ii] = hh
    purity = (Ns / Ng)[grpidx]
    compl = (Ns / Nh)[grpidx]
    matches = matches[grpidx]
    return purity, compl, matches
   
def mode(arr, weights):
    return scipy_mode(arr[np.argsort(weights)]).mode

if __name__=='__main__':
    import pandas as pd
    cat = pd.read_csv('~/Documents/Research/4HS/ECO_cat_0_Planck_memb_cat.csv')

    from h23_group_purity import get_metrics_by_group as gmg
    pp, cc = gmg(cat.g3grp_l,cat.haloid,cat.M_r)
    pp1, cc1 = get_metrics_by_group(cat.g3grp_l,cat.haloid,cat.M_r)
    print(pp)
    print(pp1)
    print(cc)
    print(cc1)
    print((pp==pp1).all())
    print((cc==cc1).all())
