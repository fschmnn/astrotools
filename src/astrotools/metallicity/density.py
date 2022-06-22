import numpy as np

def ratio_from_electron_density(n_e,element='OII'):
    '''calculate line ratios from the electron density
    
    based on Sanders+2016
    '''
    
    coefficients = {
        'OII' : (0.3771, 2468, 638.4),
        'SII' : (0.4315, 2107, 627.1)
    }
    
    a,b,c = coefficients.get(element,'OII')
    
    return a * (b+n_e) / (c+n_e)

def electron_density_from_ratio(ratio,element='OII'):
    '''calculate the electron density from the line ratio
    
    this function takes the calibration from Sanders+2016
    and inverts it to get the electron temperature (a grid
    of values is computed and the value interpolated)
    '''
    
    # the electron density
    n_e_grid = np.logspace(1,5,1000)[::-1]
    # the corresponding [SII] ratio
    ratio_grid = ratio_from_electron_density(n_e_grid,element) 

    return np.interp(ratio,ratio_grid,n_e_grid)
