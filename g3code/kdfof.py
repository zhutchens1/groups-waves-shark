import numpy as np
import time
from scipy.spatial import cKDTree
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

def kdFOF(ra, dec, zz, bperp, blos, meansep, cosmo):
    """ 
    -----------
    Compute friends-of-friends galaxy groups based on the method described in Hutchens+2023
    [2023ApJ...956...51H] which is based on Berlind+2006 [2006ApJS..167....1B]. This algorithm
    uses a k-d tree to find neighbor candidates to reduce the number of FoF pair calculations.
    Note that results may be sensitive to 64-bit vs. 32-bit precision for inputs. The code has 
    been validated to match the H23 code exactly using 64-bit inputs.
    
    Arguments:
        ra (iterable): list of right-ascesnsion coordinates of galaxies in decimal degrees.
        dec (iterable): list of declination coordinates of galaxies in decimal degrees.
        zz (iterable): redshift of input galaxies. 
        bperp (scalar): linking proportion for the on-sky plane (use 0.07 for RESOLVE/ECO)
        blos (scalar): linking proportion for line-of-sight component (use 1.1 for RESOLVE/ECO)
        meansep (scalar): mean separation of galaxies above floor in volume-limited catalog.
        cosmo : astropy.cosmology object
    Returns:
        fofid (np.array): list containing unique group ID numbers for each target in the input coordinates.
                The list will have shape len(ra).
    -----------
    """
    t1 = time.time()
    Ngalaxies = len(ra)
    ra = np.asarray(ra)
    dec = np.asarray(dec)
    zz = np.asarray(zz)
    assert (len(ra)==len(dec) and len(dec)==len(zz)),"RA/Dec/zz arrays must equivalent length."

    perpLL = bperp*meansep
    losLL = blos*meansep
    phi = np.deg2rad(ra)
    theta = np.pi/2. - np.deg2rad(dec) 
    dc = cosmo.comoving_distance(zz).value
    dm = cosmo.comoving_transverse_distance(zz).value
    xyz = np.zeros((Ngalaxies,3))
    xyz[:,0] = dc * np.sin(theta) * np.cos(phi)
    xyz[:,1] = dc * np.sin(theta) * np.sin(phi)
    xyz[:,2] = dc * np.cos(theta)
    tree = cKDTree(xyz)
    pairs = tree.query_pairs(r=np.sqrt(perpLL*perpLL + losLL*losLL), output_type='ndarray')
    i_idx, j_idx = pairs[:,0], pairs[:,1]
    ri = xyz[i_idx]
    rj = xyz[j_idx]
    dot = (ri * rj).sum(axis=1)
    mu = dot / (dc[i_idx] * dc[j_idx]) #mu=cos(alpha_ij)
    if (mu>1).any():
        mu[mu>1] = 1.
    dperp = (dm[i_idx]+dm[j_idx]) * np.sqrt((1-mu)/2.) # Eq 1. H23, but approximating sin(alpha_ij/2) ~ alpha_ij/2.
    dlos = np.abs(dc[i_idx]-dc[j_idx]) # Eq 2. H23
    friendship = (dlos <= losLL) & (dperp <= perpLL)
    friendship = sp.coo_array((friendship[friendship], (i_idx[friendship],j_idx[friendship])), shape=(Ngalaxies,Ngalaxies))
    fofID = 1+connected_components(friendship)[1] # 1+ gets rid of groupID=0
    print(f'FoF completed in {time.time()-t1:0.3f} s.')
    return fofID
 
if __name__=='__main__':
    import pandas as pd
    from astropy.cosmology import LambdaCDM
    cosmo = LambdaCDM(70,0.3,0.7)
    df = pd.read_csv("/srv/one/zhutchen/g3groupfinder/resolve_and_eco/ECOdata_G3catalog_luminosity.csv")
    df = df[df.absrmag<=-19.5]
    fofid = kdFOF(np.float32(df.radeg), np.float32(df.dedeg), np.float32(df.cz)/3e5, 0.07, 1.1, 4.84, cosmo)
    print(fofid)
    grpn, count = np.unique(fofid, return_counts=True)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.title('kd fof')
    hist=plt.hist(count, bins=np.arange(0.5,400.5,1))
    print(hist[0][0:10])
    plt.yscale('log')
    plt.xlim(0,40)
    plt.show()
