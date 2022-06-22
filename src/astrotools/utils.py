import numpy as np
from astropy.wcs import WCS 
from astropy.io import fits
import astropy.units as u

def project(x, y, pa, inc):
    """General rotation/projection routine.

    Given coordinates (x, y), will rotate and project given position angle (counter-clockwise from N), and
    inclination. Assumes centre is at (0, 0).

    Args:
        x (float or numpy.ndarray): x-coordinate(s)
        y (float or numpy.ndarray): y-coordinates(s)
        pa (float): Position angle (degrees)
        inc (float): Inclination (degrees)

    Returns:
        x_proj, y_proj: The rotated, projected (x, y) coordinates.

    """

    angle = np.radians(pa)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    x_proj = x * cos_a + y * sin_a
    y_proj = - x * sin_a + y * cos_a

    # Account for inclination

    x_proj /= np.cos(np.radians(inc))

    return x_proj, y_proj

def deprojected_galactic_radius(catalogue,sample_table):
    '''
    
    this function requires the absolute path the the MAPS file. 
    just copy it and adjust it in the function
    
    '''

    for gal_name in sample_table['name']:
    
        tmp = catalogue[catalogue['gal_name']==gal_name]
        wcs = WCS(fits.getheader(data_ext/'MUSE'/'DR2.1'/'MUSEDAP'/f'{gal_name}_MAPS.fits',extname='FLUX'))
        x_cen,y_cen=sample_table.loc[gal_name]['SkyCoord'].to_pixel(wcs=wcs)
        
        inc = sample_table.loc[gal_name]['Inclination']
        pa  = sample_table.loc[gal_name]['posang']

        x_proj = (tmp['x_neb']-x_cen) * np.cos(pa*u.deg) + (tmp['y_neb']-y_cen) * np.sin(pa*u.deg)
        y_proj = - (tmp['x_neb']-x_cen) * np.sin(pa*u.deg) + (tmp['y_neb']-y_cen) * np.cos(pa*u.deg)

        # Account for inclination
        x_proj /= np.cos(inc*u.deg)

        gal_dist = np.sqrt(x_proj**2+y_proj**2)/5
        
    