import logging 

logger = logging.getLogger(__name__)
from astropy.coordinates import SkyCoord
import astropy.units as u 

def resolution_from_wcs(wcs):
    '''calculate the resolution in arcsecond for any given wcs
    
    Parameters
    ----------
    wcs
    
    Returns
    -------
    tuple : the extend per pixel in arcseconds
    '''
    
    shape = wcs._naxis
    
    x = [0,0,shape[0]]
    y = [0,shape[1],0]

    coords = SkyCoord.from_pixel(x,y,wcs)
    
    dx = coords[0].separation(coords[1]) / shape[0]
    dy = coords[0].separation(coords[1]) / shape[1]
    
    return (round(dx.to(u.arcsecond).value,3),round(dy.to(u.arcsecond).value,3))