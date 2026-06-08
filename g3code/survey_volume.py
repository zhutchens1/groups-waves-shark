from astropy.cosmology import LambdaCDM
from scipy.stats import binned_statistic
import numpy as np

def comoving_volume(ra1,ra2,dec1,dec2,z1,z2,H0,Om0,Ode0):
    """
    Compute the comoving volume of a lat/lon rectangle on-sky,
    bounded by (ra1,dec1)->(ra2,dec2), and extended over the 
    redshift range (z1,z2).

    Parameters
    -----------------
    ra1 : scalar
        Lower RA in decimal degrees.
    ra2 : scalar
        Upper RA in decimal degrees.
    dec1 : scalar
        Lower declination in decimal degrees.
    dec2 : scalar
        Upper declination in decimal degrees.
    z1 : scalar
        Inner redshift.
    z2 : scalar
        Outer redshift.
    H0 : scalar
        Hubble constant for LambdaCDM cosmology.
    Om0 : scalar
        Density of non-relativistic matter in units of critical
        density at z=0.
    Ode0 : scalar
        Density of dark matter at z=0 in units of critical density.

    Returns
    ------------------
    volume : scalar
        Volume of survey spanned by input coordinates.
        Units of Mpc^3.
    """
    cosmo = LambdaCDM(H0,Om0,Ode0)
    dtor=np.pi/180.
    if ra2>=ra1:
        delta_ra_rad = (ra2-ra1)*dtor
    else:
        delta_ra_rad = ((360.-ra1)+ra2)*dtor
    delta_dec_rad = np.sin(dec2*dtor)-np.sin(dec1*dtor)
    solidangle = delta_ra_rad * delta_dec_rad
    dv = cosmo.comoving_volume(z2).value - cosmo.comoving_volume(z1).value
    return (solidangle/(4*np.pi)) * dv

def solid_angle_str(ra, dec):
    """ for a lat/lon rectangle """
    ra1 = np.min(ra)
    ra2 = np.max(ra)
    dec1 = np.min(dec)
    dec2 = np.max(dec)
    dtor=np.pi/180.
    if ra2>=ra1:
        delta_ra_rad = (ra2-ra1)*dtor
    else:
        delta_ra_rad = ((360.-ra1)+ra2)*dtor
    delta_dec_rad = np.sin(dec2*dtor)-np.sin(dec1*dtor)
    solidangle = delta_ra_rad * delta_dec_rad
    return solidangle

def comoving_volume_shell(z1,z2,H0,Om0,Ode0):
    """
    Compute the comoving volume per steradian of a spherical
    shell with inner redshift `z1` and outer radius `z2`. To
    determine the comoving volume associated with some solid
    angle A on sky, compute A*comoving_volume_per_skyarea(*).
    This function allows for generalization of the `comoving_
    volume` to surveys that do not carve out lat/lon rectang-
    les on sky.

    Parameters
    -----------------
    z1 : scalar
        Inner redshift.
    z2 : scalar
        Outer redshift.
    H0 : scalar
        Hubble constant for LambdaCDM cosmology.
    Om0 : scalar
        Density of non-relativistic matter in units of critical
        density at z=0.
    Ode0 : scalar
        Density of dark matter at z=0 in units of critical density.

    Returns
    ------------------
    volume : scalar
        Volume of shell spanned by the two input redshifts, with
        units of (Mpc)^3/sr.
    """
    cosmo = LambdaCDM(H0,Om0,Ode0)
    dv = cosmo.comoving_volume(z2).value - cosmo.comoving_volume(z1).value
    return (1/(4*np.pi)) * dv


def integrate_volume(redshifts, solid_angle, H0, Om0, Ode0):
    """
    Compute the comoving volume for a survey
    whose field-of-view changes with redshift.
    This function integrates A(z)W(z)dz, where
    A(z) is the solid angle in str, W(z) is the
    differential comoving volume per z per str.

    Parameters
    ---------------------
    redshifts : array_like
        Redshifts spanned by the volume. The spacing
        between elements determines the value of dz.
        Example: np.linspace(0.1,0.2,1000)
        Here, dz=(0.2-0.1)/1000 = 1E-4.
    solid_angle : array_like
        Solid angle at each `z` in redshifts. Length
        should match `redshifts` and its units should
        be expressed in steradians.
    
    Returns
    ---------------------
    volume : scalar
        Volume spanned by the input redshift and
        solid angle, in units Mpc. 
    """
    cosmo = LambdaCDM(H0,Om0,Ode0)
    vol_per_zstr = cosmo.differential_comoving_volume(redshifts)
    integrand = (solid_angle)*vol_per_zstr
    return np.sum(integrand*(redshifts[1]-redshifts[0])).value
   
def sky_area(xx,yy,bins):
    """
    Estimate the on-sky area for an arbitrary geometry. This 
    algorithm partitions the source distribution as a series
    of rectangles, with partition widths set by `bins`, such
    that the total area can be estimated by the sum of the 
    individual partition areas. This algorithm may be sensitive
    to the choice of `bins`, and it is very sensitive to outliers.

    Parameters
    ----------------
    xx : array_like
        x-coordinate in arbitrary units (e.g. RA in decimal degrees).
    yy : array_like
        y-coordinate in arbitary units (e.g. Dec in decimal degrees).
    bins : int or array_like
         If int, `bins` specifies the number of bins, and in turns,
        a constant partition width. If array_like, it specifies the
        edges of the bins (length = # bins + 1).

    Returns
    -----------------
    area : scalar
        Estimate of the total area, in units corresponding to those
        passed for `xx` and `yy` (e.g., degrees squared).
    """
    ymax, bin_edges, _ = binned_statistic(xx,yy,'max',bins=bins)
    ymin, _, _ = binned_statistic(xx,yy,'min',bins=bins)
    dx = bin_edges[1:]-bin_edges[:-1]
    return np.sum((ymax-ymin)*dx)   

if __name__=='__main__':
    x=integrate_volume(np.linspace(2530/3e5,7470/3e5,int(1e4)),np.full(int(1e4),1.465492683585301),100.,0.3,0.7)
    print(x)
