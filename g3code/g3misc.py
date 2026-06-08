import numpy as np
from scipy.stats import binned_statistic
from sklearn.utils import check_random_state

# ------------------------------------------------------------------------------ #
# Supporting group-metric functions (group center, multiplicity, etc.) 
# ------------------------------------------------------------------------------ #
def group_skycoords(galaxyra, galaxydec, galaxyz, galaxygrpid):
    """ 
    -----
    Obtain a list of group centers (RA/Dec/cz) given a list of galaxy coordinates (equatorial)
    and their corresponding group ID numbers.
    
    Inputs (all same length)
       galaxyra : 1D iterable,  list of galaxy RA values in decimal degrees
       galaxydec : 1D iterable, list of galaxy dec values in decimal degrees
       galaxycz : 1D iterable, list of galaxy cz values in km/s
       galaxygrpid : 1D iterable, group ID number for every galaxy in previous arguments.
    
    Outputs (all shape match `galaxyra`)
       groupra : RA in decimal degrees of galaxy i's group center.
       groupdec : Declination in decimal degrees of galaxy i's group center.
       groupz : Redshift of galaxy i's group center.
    """
    galaxyra = np.asarray(galaxyra)
    galaxydec = np.asarray(galaxydec)
    galaxyz = np.asarray(galaxyz)
    galaxygrpid = np.asarray(galaxygrpid)
    galaxyX, galaxyY, galaxyZ = cartesian_from_spherical_z(galaxyra, galaxydec, galaxyz)
    uniqgrpid, galaxyidx = np.unique(galaxygrpid, return_inverse=True)
    Ngroups = len(uniqgrpid)
    nmembers = np.bincount(galaxyidx, minlength=Ngroups)
    Xcen = np.bincount(galaxyidx, weights=galaxyX, minlength=Ngroups) / nmembers
    Ycen = np.bincount(galaxyidx, weights=galaxyY, minlength=Ngroups) / nmembers
    Zcen = np.bincount(galaxyidx, weights=galaxyZ, minlength=Ngroups) / nmembers
    zcen = np.sqrt(Xcen*Xcen + Ycen*Ycen + Zcen*Zcen)
    racen = (np.degrees(np.arctan2(Ycen,Xcen))+360.) % 360.0
    deccen = np.degrees(np.arcsin(Zcen / zcen))
    groupra = racen[galaxyidx]
    groupdec = deccen[galaxyidx]
    groupz = zcen[galaxyidx]
    return groupra, groupdec, groupz

def angular_separation(ra1,dec1,ra2,dec2):
    """
    Compute the angular separation between two lists of galaxies using the Haversine formula.
    
    Parameters
    ------------
    ra1, dec1, ra2, dec2 : array-like
       Lists of right-ascension and declination values for input targets, in decimal degrees. 
    
    Returns
    ------------
    angle : np.array
       Array containing the angular separations between coordinates in list #1 and list #2, as above.
       Return value expressed in radians, NOT decimal degrees.
    """
    phi1 = np.deg2rad(ra1)
    phi2 = np.deg2rad(ra2)
    theta1 = np.pi/2. - np.deg2rad(dec1)
    theta2 = np.pi/2. - np.deg2rad(dec2)
    sin_dt = np.sin((theta2-theta1)/2.0)
    sin_dp = np.sin((phi2 - phi1)/2.0)
    return 2*np.arcsin(np.sqrt(sin_dt*sin_dt + np.sin(theta1)*np.sin(theta2) * (sin_dp*sin_dp)))

def multiplicity_function(grpids, return_by_galaxy=False):
    """
    Obtain the number of galaxies in each host group.

    Parameters
    ----------
    grpids : iterable
        List of group ID numbers. Length must match # galaxies.
    Returns
    -------
    occurences : list
        Number of galaxies in each galaxy group (length matches # groups).
    """
    grpids=np.asarray(grpids)
    uniqid, inv, counts = np.unique(grpids, return_counts=True, return_inverse=True)
    if return_by_galaxy:
        return counts[inv]
    else:
        return counts

def cartesian_from_spherical_z(ra, dec, redshift):
    """
    Convert RA/Dec/z to comoving (x,y,z).

    Parameters
    ----------------
    ra : array_like
        RA in decimal degrees.
    dec : array_like
        Decl. in decimal degrees.
    redshift : array_like
        Redshift (z).
    cosmo:
        Astropy cosmology object (specifies
        comoving distance formulation).

    Returns
    ----------------
    XX, YY, ZZ : array_like
        Cartesian coordinates in comoving Mpc
        according to the input cosmology.
    """
    phi = np.deg2rad(ra)
    theta = np.pi/2. - np.deg2rad(dec) 
    XX = redshift * np.sin(theta) * np.cos(phi)
    YY = redshift * np.sin(theta) * np.sin(phi)
    ZZ = redshift * np.cos(theta)
    return XX, YY, ZZ

def comoving_cartesian_from_spherical(ra, dec, redshift, cosmo):
    """
    Convert RA/Dec/z to comoving (x,y,z).

    Parameters
    ----------------
    ra : array_like
        RA in decimal degrees.
    dec : array_like
        Decl. in decimal degrees.
    redshift : array_like
        Redshift (z).
    cosmo:
        Astropy cosmology object (specifies
        comoving distance formulation).

    Returns
    ----------------
    XX, YY, ZZ : array_like
        Cartesian coordinates in comoving Mpc
        according to the input cosmology.
    """
    phi = np.deg2rad(ra)
    theta = np.pi/2. - np.deg2rad(dec) 
    dc = cosmo.comoving_distance(redshift).value
    XX = dc * np.sin(theta) * np.cos(phi)
    YY = dc * np.sin(theta) * np.sin(phi)
    ZZ = dc * np.cos(theta)
    return XX, YY, ZZ

def get_int_mag(galmag, grpid):
    """
    Given a list of galaxy absolute magnitudes and group ID numbers,
    compute group-integrated total magnitudes.

    Parameters
    ------------
    galmag : iterable
       List of absolute magnitudes for every galaxy (SDSS r-band).
    grpid : iterable
       List of group ID numbers for every galaxy.

    Returns
    ------------
    Mint_grp : np array
       Array containing group-integrated magnitudes for each galaxy. Length matches `galmag`.
    """
    galmag=np.asarray(galmag)
    grpid=np.asarray(grpid)
    grpmags = np.zeros(len(galmag))
    uniqgrpid, galaxyidx = np.unique(grpid, return_inverse=True)
    Ngroups = len(uniqgrpid)
    # Mint_grp = -2.5 * log10(Sum[10 ^ -0.4M]) for galaxy abs mag M.
    Mi_terms = np.power(10, -0.4*galmag)
    sum_terms = np.bincount(galaxyidx, weights=Mi_terms, minlength=Ngroups) 
    Mint_grp = -2.5 * np.log10(sum_terms)
    return Mint_grp[galaxyidx]

# ------------------------------------------------------------------------------ #
# Misc. supporting functions
# ------------------------------------------------------------------------------ #
def center_binned_stats(*args, **kwargs):
    """ 
     Same as scipy.stats.binned_statistic, but returns
     the bin centers (matching length of `statistic`)
     instead of the binedges.

     See docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html
    """
    stat, binedges, binnumber = binned_statistic(*args,**kwargs)
    bincenters = (binedges[:-1]+binedges[1:])/2.
    return stat, bincenters, binedges, binnumber

def sigmarange(x):
    q84, q16 = np.percentile(x, [84 ,16])
    return (q84-q16)/2.

def giantmodel(x, a, b): 
    return np.abs(a)*np.log10(np.abs(b)*x+1)

def decayexp(x, a, b): 
    return np.abs(a)*np.exp(-1*np.abs(b)*x)

def smoothedbootstrap(data, n_bootstraps, user_statistic, kwargs=None, random_state=None):
    """Compute smoothed bootstrapped statistics of a data set.
    Parameters
    ----------
    data : array_like
        A 1-dimensional data array of size n_samples
    n_bootstraps : integer
        the number of bootstrap samples to compute.  Note that internally,
        two arrays of size (n_bootstraps, n_samples) will be allocated.
        For very large numbers of bootstraps, this can cause memory issues.
    user_statistic : function
        The statistic to be computed.  This should take an array of data
        of size (n_bootstraps, n_samples) and return the row-wise statistics
        of the data.
    kwargs : dictionary (optional)
        A dictionary of keyword arguments to be passed to the
        user_statistic function.
    random_state: RandomState or an int seed (0 by default)
        A random number generator instance
    Returns
    -------
    distribution : ndarray
        the bootstrapped distribution of statistics (length = n_bootstraps)
    """
    # we don't set kwargs={} by default in the argument list, because using
    # a mutable type as a default argument can lead to strange results
    if kwargs is None:
        kwargs = {}

    rng = check_random_state(random_state)

    data = np.asarray(data)
    n_datapts = data.size

    if data.ndim != 1:
        raise ValueError("bootstrap expects 1-dimensional data")

    # Generate random indices with repetition
    ind = rng.randint(n_datapts, size=(n_bootstraps, n_datapts))
    
    # smoothing noise
    noisemean = 0.
    noisesigma = np.std(data,ddof=1) / np.sqrt(n_datapts)
    noise = np.random.normal(noisemean,noisesigma,(n_bootstraps, n_datapts))
    databroadcast = data[ind] + noise

    # Call the function
    stat_bootstrap = user_statistic(databroadcast, **kwargs)

    # compute the statistic on the data
    return stat_bootstrap
