import numpy as np 
import astropy.constants as c
import astropy.units as u 

def logq_D91(ratio):
    '''Calculate the ionisation parameter
    
    based on the prescription by Diaz et al. (1991)

    Dimensionless ionisation parameter:
    
    log𝑢=(−1.684±0.076)⋅log([SIII]/[SII])−(2.986±0.027)
    
    This is related to the ionisation parameter via:
    
    logq = log𝑢 - log c (in cgs)
    
    The line ratio used here is 
    [SIII]/[SII] = [SIII](9069+9532)/[SII](6717+6713)
    
    If [SIII]9532 is not available, the following relation can be used
    [SIII]9532 = 2.47 * [SII]9069
    
    Parameters
    ----------
    
    ratio : float
        the [SIII]/[SII] line ratio
    
    Returns 
    -------
    
    logq : float
        the ionisation parameter
        
    Notes
    -----
    
    all log are log10

    '''
    
    speed_of_light = c.c.to(u.cm/u.s).value
    
    return 1.684*np.log10(ratio)-2.986 + np.log10(speed_of_light)

def logq_D91_reverse(logq):
    '''
    get [SIII]/[SII] from log q
    
    '''
    speed_of_light = c.c.to(u.cm/u.s).value

    return 10**((logq - np.log10(speed_of_light) + 2.986)/1.684)