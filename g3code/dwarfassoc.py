import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from g3misc import *
SPEED_OF_LIGHT = 3e5

def dwarfAssocRoutine(dwarfra, dwarfdec, dwarfz, grpra, grpdec, grpz, grpid, radius_boundary, velocity_boundary, cosmo):
    """ 
    Associate galaxies to a group catalog based on given radius and velocity boundaries, based on a method
    similar to that presented in Eckert+ 2016. As used in Hutchens+2023 

    Parameters
    ----------
    dwarfra : iterable
        Right-ascension of dwarf galaxies in degrees.
    dwarfdec : iterable
        Declination of dwarf galaxies in degrees.
    dwarfcz : iterable
        Redhshift velocities of dwarf galaxies in km/s.
    grpra : iterable
        Right-ascension of group centers in degrees.
    grpdec : iterable
        Declination of group centers in degrees. Length matches `grpra`.
    grpcz : iterable
        Redshift velocity of group center in km/s. Length matches `grpra`.
    grpid : iterable
        group ID of each FoF group (i.e., from `foftools.fast_fof`.) Length matches `grpra`.
    radius_boundary : iterable
        Radius within which to search for dwarf galaxies around FoF groups. Length matches `grpra`.
    velocity_boundary : iterable
        Velocity from group center within which to search for dwarf galaxies around FoF groups. Length matches `grpra`.
    cosmo : astropy.cosmology object
        Astropy cosmology to specify cosmological distances.

    Returns
    -------
    assoc_grpid : iterable
        group ID of every dwarf galaxy. Length matches `dwarfra`.
    assoc_flag : iterable
        association flag for every galaxy (see function description). Length matches `dwarfra`.
    """
    dwarfra = np.asarray(dwarfra)
    dwarfdec = np.asarray(dwarfdec)
    dwarfz = np.asarray(dwarfz)
    grpra = np.asarray(grpra)
    grpdec = np.asarray(grpdec)
    grpz = np.asarray(grpz)
    grpid = np.asarray(grpid)
    velocity_boundary=np.asarray(velocity_boundary)
    radius_boundary=np.asarray(radius_boundary)

    Ndwarf = len(dwarfra)
    dwarf_cmvg = cosmo.comoving_transverse_distance(dwarfz).value
    assoc_grpid = np.zeros(Ndwarf, dtype=np.int32) - 1
    assoc_flag = np.zeros(Ndwarf, dtype=bool)
    r2plusv2 = np.zeros(Ndwarf)

    # resize group coordinates to be the # of groups, not # galaxies
    junk, uniqind = np.unique(grpid, return_index=True)
    grpra = grpra[uniqind]
    grpdec = grpdec[uniqind]
    grpz = grpz[uniqind]
    grpid = grpid[uniqind]
    grp_cmvg = cosmo.comoving_transverse_distance(grpz).value
    velocity_boundary=velocity_boundary[uniqind]
    radius_boundary=radius_boundary[uniqind]

    # get cartesian positions of groups, form a kd tree, then find closest dwarf neighbors 
    # using a query ball point.
    grpX, grpY, grpZ = comoving_cartesian_from_spherical(grpra, grpdec, grpz, cosmo)
    dwarfX, dwarfY, dwarfZ = comoving_cartesian_from_spherical(dwarfra, dwarfdec, dwarfz, cosmo)
    rmax = 16 * np.max(radius_boundary)
    grp_tree = cKDTree(np.array([grpX,grpY,grpZ]).T)
    dwarf_tree = cKDTree(np.array([dwarfX, dwarfY, dwarfZ]).T)
    nnind = grp_tree.query_ball_tree(dwarf_tree, r=rmax)
    N_G = len(nnind)
    for grp_i in tqdm(range(N_G), total=N_G):
        dwarf_i = np.array(nnind[grp_i])
        if len(dwarf_i) == 0:
            continue
       
        alpha_ij = angular_separation(grpra[grp_i], grpdec[grp_i], dwarfra[dwarf_i], dwarfdec[dwarf_i]) 
        Rp = 0.5 * (grp_cmvg[grp_i] + dwarf_cmvg[dwarf_i]) * alpha_ij 
        vpec = SPEED_OF_LIGHT * np.abs(grpz[grp_i] - dwarfz[dwarf_i]) / (1 + grpz[grp_i])
        assoc_condition = (Rp < radius_boundary[grp_i]) & (vpec < velocity_boundary[grp_i])
        assoc_sep = (Rp*Rp)/(radius_boundary[grp_i]*radius_boundary[grp_i]) + (vpec*vpec)/(velocity_boundary[grp_i]*velocity_boundary[grp_i])
      
        # `associate` is a bool: it is true when Rproj and dv_proj requirements are met (`assoc_condition`) and
        # either (i) the dwarf was never previously associated or (ii) the new giant-only group is a better fit. 
        associate = assoc_condition & (~assoc_flag[dwarf_i] | (assoc_flag[dwarf_i] & (assoc_sep < r2plusv2[dwarf_i])))
        assoc_grpid[dwarf_i[associate]] = grpid[grp_i]
        assoc_flag[dwarf_i[associate]] = True
        r2plusv2[dwarf_i[associate]] = assoc_sep[associate]
    
    still_isolated = (assoc_grpid < 0)
    maxgrpid = np.max(grpid)
    assoc_grpid[still_isolated] = np.arange(maxgrpid+1, maxgrpid+np.sum(still_isolated)+1)
    return assoc_grpid, assoc_flag
