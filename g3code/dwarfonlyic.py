import numpy as np
from scipy.spatial import cKDTree
from copy import deepcopy
from g3misc import *
SPEED_OF_LIGHT = 3.0e5

#
# Python code for the dwarf-only group-finding routine
# from Hutchens et al. 2023 / 2023ApJ...956...51H
#

def dwarfOnlyICRoutine(galaxyra, galaxydec, galaxyz, galaxyabsmag, rprojboundary, vprojboundary, starting_id, cosmo):
    """
    Construct dwarf-only groups via iterative combination.

    Parameters
    --------------
    galaxyra : array_like
        RA of galaxies in decimal degrees.
    galaxydec : array_like
        Dec of galaxies in decimal degrees.
    galaxyz : array_like
        Redshift of galaxies.
    galaxyabsmag : array_like
        Absolute magnitudes of galaxies.
    rprojboundary : callable
        Search boundary to apply on-sky. Should be callable function of group integrated luminosity..
        Units: Mpc consistent with `cosmo`.
    vprojboundary : callable
        Search boundary to apply in line-of-sight. Should be callable function of group integrated luminosity.
        Units: km/s
    starting_id : int
        Group ID to start at (to avoid overwriting existing group IDs for giant-hosting systems).
    cosmo : astropy.cosmology object
       Astropy cosmology for computing cosmological distances.
    
    Returns
    --------------
    dwarfgroupid : np.array
        Array of group ID numbers following iterative combination.
    """
    galaxyra=np.asarray(galaxyra)
    galaxydec=np.asarray(galaxydec)
    galaxyz=np.asarray(galaxyz)
    galaxyabsmag=np.asarray(galaxyabsmag)
    assert callable(rprojboundary),"Argument `rprojboundary` must callable function of Mr_int."
    assert callable(vprojboundary),"Argument `vprojboundary` must callable function of Mr_int."

    converged=False
    groupid = np.arange(starting_id, starting_id+len(galaxyra))
    niter=0
    while (not converged):
        print(f"Dwarf-only iterative combination {niter+1} in progress...")
        oldgroupid = groupid
        groupid = nearest_neighbor_assign_dw(galaxyra,galaxydec,galaxyz,galaxyabsmag, oldgroupid,\
                        rprojboundary,vprojboundary,cosmo)
        converged = np.array_equal(oldgroupid,groupid)
        niter+=1
    print("Dwarf-only iterative combination complete.")
    return groupid

def nearest_neighbor_assign_dw(galaxyra,galaxydec,galaxyz,galaxyabsmag,grpid,rprojboundary,vprojboundary,cosmo):
    """
    Refine input group ID by merging nearest-neighbor groups subject to boundary constraints.
    For info on arguments, see "dwarfOnlyICRoutine"

    Returns
    --------------
    refinedgrpid : np.array
        Refined group ID numbers based on nearest-neighbor merging.
    """
    # Prepare output array
    groupra, groupdec, groupz = group_skycoords(galaxyra, galaxydec, galaxyz, grpid)
    groupMint = get_int_mag(galaxyabsmag, grpid)
    gX, gY, gZ = cartesian_from_spherical_z(galaxyra, galaxydec, galaxyz)

    # Get unique potential seed groups
    uniqgrpid, uniqind, galaxyidx, seedN = np.unique(grpid, return_index=True, return_inverse=True, return_counts=True)
    seedra, seeddec, seedz = groupra[uniqind], groupdec[uniqind], groupz[uniqind]
    seedMint = groupMint[uniqind]
    seeddm = cosmo.comoving_transverse_distance(seedz).value
    seedX, seedY, seedZ = cartesian_from_spherical_z(seedra, seeddec, seedz)
    xyz = np.array([seedX, seedY, seedZ]).T
    kdt = cKDTree(xyz)
    nndist, nnind = kdt.query(xyz,k=2)
    nndist=nndist[:,1] # ignore self match
    nnind=nnind[:,1]
 
    # check whether, supposing the two seed groups were combined into a tentative
    # larger group, if both seed groups would satisfy the Rproj and Vproj requirements
    # when calculated from the center of the larger tentative group.
    n_tent = seedN + seedN[nnind]
    Mr_tent = -2.5*np.log10(10**(-0.4*seedMint) + 10**(-0.4*seedMint[nnind]))
    G = len(uniqgrpid)
    sumX = np.zeros(G)
    sumY = np.zeros(G)
    sumZ = np.zeros(G)
    np.add.at(sumX, galaxyidx, gX)
    np.add.at(sumY, galaxyidx, gY)
    np.add.at(sumZ, galaxyidx, gZ)
    tent_Xcen = (sumX + sumX[nnind]) / n_tent
    tent_Ycen = (sumY + sumY[nnind]) / n_tent
    tent_Zcen = (sumZ + sumZ[nnind]) / n_tent
    tent_z = np.sqrt(tent_Xcen*tent_Xcen + tent_Ycen*tent_Ycen + tent_Zcen*tent_Zcen)
    tent_ra = (np.degrees(np.arctan2(tent_Ycen,tent_Xcen))+360.) % 360.0
    tent_dec = np.degrees(np.arcsin(tent_Zcen / tent_z))
    tent_dm = cosmo.comoving_transverse_distance(tent_z).value

    alpha_i = angular_separation(seedra, seeddec, tent_ra, tent_dec)
    alpha_j = angular_separation(seedra[nnind], seeddec[nnind], tent_ra, tent_dec)
    dperp_i = 0.5 * (seeddm + tent_dm) * alpha_i
    dperp_j = 0.5 * (seeddm[nnind] + tent_dm) * alpha_j
    one_plus_z = (1 + tent_z)
    vpec_i = SPEED_OF_LIGHT*np.abs(tent_z - seedz) / one_plus_z
    vpec_j = SPEED_OF_LIGHT*np.abs(tent_z - seedz[nnind]) / one_plus_z 

    seedi_in = (dperp_i < rprojboundary(Mr_tent)) & (vpec_i < vprojboundary(Mr_tent))
    seedj_in = (dperp_j < rprojboundary(Mr_tent)) & (vpec_j < vprojboundary(Mr_tent))
    recip = nnind[nnind] == np.arange(len(nnind)) # are they each other's own nearest neighbor?
    merge = seedi_in & seedj_in & recip
  
    ii=np.where(merge)[0]
    jj=nnind[ii]
    keep = (ii<jj)
    ii, jj = ii[keep], jj[keep]
    revisedid = np.minimum(uniqgrpid[ii],uniqgrpid[jj])
    uniqgrpid[ii] = revisedid
    uniqgrpid[jj] = revisedid
    refinedgrpid = uniqgrpid[galaxyidx]
    return refinedgrpid
