import numpy as np
import matplotlib.pyplot as plt
import astropy 
from astropy.visualization import simple_norm
import astropy.units as u
from astropy.coordinates import SkyCoord

tab10 = ['#e15759','#4e79a7','#f28e2b','#76b7b2','#59a14e','#edc949','#b07aa2','#ff9da7','#9c755f','#bab0ac']    
single_column = 3.321 # in inch
two_column    = 6.974 # in inch

def figsize(scale=1):
    '''Create nicely proportioned figure

    This function calculates the optimal figuresize for any given scale
    (the ratio between figuresize and textwidth. A figure with scale 1
    covers the entire writing area). Therefor it is important to know 
    the textwidth of your target document. This can be obtained by using
    the command "\the\textwidth" somewhere inside your document.
    '''

    # for one column: 504.0p
    width_pt  = 240                           # textwidth from latex
    in_per_pt = 1.0/72.27                     # Convert pt to inch
    golden    = 1.61803398875                 # Aesthetic ratio 
    width  = width_pt * in_per_pt * scale     # width in inches
    height = width / golden                   # height in inches
    return [width,height]


def create_RGB(r,g,b,stretch='linear',weights=None,percentile=95):
    '''combie three arrays to one RGB image
    
    Parameters
    ----------
    r : ndarray
        (n,m) array that is used for the red channel
        
    g : ndarray
        (n,m) array that is used for the green channel
        
    b : ndarray
        (n,m) array that is used for the blue channel
    
    percentile : float
        percentile that is used for the normalization
        
    Returns
    -------
    rgb : ndarray
        (n,m,3) array that is normalized to 1
    '''

    if not r.shape == g.shape == b.shape:
        raise ValueError('input arrays must have the dimensions')
    
    # create an empty array with the correct size
    rgb = np.empty((*r.shape,3))
    
    if type(percentile)==float or type(percentile)==int:
        percentile = 3*[percentile]

    # assign the input arrays to the 3 channels and normalize them to 1
    rgb[...,0] = r / np.nanpercentile(r,percentile[0])
    rgb[...,1] = g / np.nanpercentile(g,percentile[1])
    rgb[...,2] = b / np.nanpercentile(b,percentile[2])

    if weights:
        rgb[...,0] *= weights[0]
        rgb[...,1] *= weights[1]
        rgb[...,2] *= weights[2]

    #rgb /= np.nanpercentile(rgb,percentile)
    
    # clip values (we use percentile for the normalization) and fill nan
    rgb = np.clip(np.nan_to_num(rgb,nan=1),a_min=0,a_max=1)
    
    return rgb

def create_RGB_new(r,g,b):
    '''combie three arrays to one RGB image
    
    Parameters
    ----------
    r : ndarray
        (n,m) array that is used for the red channel
        
    g : ndarray
        (n,m) array that is used for the green channel
        
    b : ndarray
        (n,m) array that is used for the blue channel
    
    percentile : float
        percentile that is used for the normalization
        
    Returns
    -------
    rgb : ndarray
        (n,m,3) array that is normalized to 1
    '''

    if not r.shape == g.shape == b.shape:
        raise ValueError('input arrays must have the dimensions')
    
    weights = [1,1,1]

    # create an empty array with the correct size
    rgb = np.empty((*r.shape,3))
    
    r_lim = np.nanpercentile(r,[51,94])
    g_lim = np.nanpercentile(g,[61,94])
    b_lim = np.nanpercentile(b,[1,98])

    # assign the input arrays to the 3 channels and normalize them to 1
    rgb[...,0] = weights[0]*(r-r_lim[0]) / (r_lim[1]-r_lim[0])
    rgb[...,1] = weights[1]*(g-g_lim[0]) / (g_lim[1]-g_lim[0])
    rgb[...,2] = weights[2]*(b-b_lim[0]) / (b_lim[1]-b_lim[0])

    if False:
        fig,ax=plt.subplots()
        ax.hist(rgb[...,0].flatten(),bins=np.linspace(0,1,20),label='red',alpha=0.5)
        ax.hist(rgb[...,1].flatten(),bins=np.linspace(0,1,20),label='green',alpha=0.5)
        ax.hist(rgb[...,2].flatten(),bins=np.linspace(0,1,20),label='blue',alpha=0.5)
        ax.legend()
        plt.show()

    #rgb /= np.nanpercentile(rgb,percentile)
    
    # clip values (we use percentile for the normalization) and fill nan
    rgb = np.clip(np.nan_to_num(rgb,nan=1),a_min=0,a_max=1)
    
    return rgb

def quick_plot(data,wcs=None,figsize=(two_column,two_column),cmap=plt.cm.hot,filename=None,**kwargs):
    '''create a quick plot 

    uses norm     
    '''
    
    fig = plt.figure(figsize=figsize)
    
    if isinstance(data,astropy.nddata.nddata.NDData):
        ax = fig.add_subplot(projection=data.wcs)
        img = data.data
    if isinstance(data,astropy.nddata.utils.Cutout2D):
        ax = fig.add_subplot(projection=data.wcs)
        img = data.data
    elif wcs:
        ax = fig.add_subplot(projection=wcs)
        img = data
    else:
        ax = fig.add_subplot()
        img = data
        
    norm = simple_norm(img,clip=False,percent=99)
    ax.imshow(img,norm=norm,cmap=cmap)
    ax.set(**kwargs)
    
    if filename:
        plt.savefig(filename,dpi=600)
    
    #plt.show()

    return ax

def add_scale(ax,length,label=None,color='black',fontsize=10):
    '''add a scale to a plot
    
    The scale if calculated from the wcs information of the plot
    '''
    
    if not hasattr(ax,'wcs'):
        raise AttributeError('axis is missing wcs information')
        
    wcs = ax.wcs
    
    w,h = wcs._naxis
    x,y = 0.05*w, 0.05*h
    start = SkyCoord.from_pixel(0,y,wcs)
    end   = SkyCoord.from_pixel(w,y,wcs)
    scale = length / start.separation(end)
    
    ax.plot([x,x+w*scale],[y,y],color=color,marker='|')
    
    if label:
        ax.text(x+0.5*w*scale,y*1.2,label,horizontalalignment='center',color=color,fontsize=fontsize)
    else:
        ax.text(x+0.5*w*scale,y*1.2,length,horizontalalignment='center',color=color,fontsize=fontsize)

    return w*scale


from matplotlib.transforms import Affine2D


def rainbow_text(x, y, strings, colors, orientation='horizontal',
                 ax=None, **kwargs):
    """
    Take a list of *strings* and *colors* and place them next to each
    other, with text strings[i] being shown in colors[i].

    Parameters
    ----------
    x, y : float
        Text position in data coordinates.
    strings : list of str
        The strings to draw.
    colors : list of color
        The colors to use.
    orientation : {'horizontal', 'vertical'}
    ax : Axes, optional
        The Axes to draw into. If None, the current axes will be used.
    **kwargs
        All other keyword arguments are passed to plt.text(), so you can
        set the font size, family, etc.
    """
    if ax is None:
        ax = plt.gca()
    t = ax.transData
    canvas = ax.figure.canvas

    assert orientation in ['horizontal', 'vertical']
    if orientation == 'vertical':
        kwargs.update(rotation=90, verticalalignment='bottom')

    for s, c in zip(strings, colors):
        text = ax.text(x, y, s + " ", color=c, transform=t, **kwargs)

        # Need to draw to update the text position.
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        if orientation == 'horizontal':
            t = text.get_transform() + Affine2D().translate(ex.width, 0)
        else:
            t = text.get_transform() + Affine2D().translate(0, ex.height)


def fix_aspect_ratio(ax,aspect_ratio=1):
    '''set the aspect ratio of an axis
    
    takes the limits of the plot to make sure the axis has the desired aspect ratio
    '''

    ax.set_aspect(aspect_ratio*np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))

    return 0

from scipy.stats import binned_statistic

def bin_stat(x,y,bins=10,range=None,statistic='mean'):
    '''calculate the binned statistics'''

    # just ignore nan values
    x, y = x[~np.isnan(y) & np.isfinite(y)], y[~np.isnan(y) & np.isfinite(y)]
    x, y = x[~np.isnan(x) & np.isfinite(x)], y[~np.isnan(x) & np.isfinite(x)]

    mean, edges, _ = binned_statistic(x,y,statistic=statistic,bins=bins,range=range)
    std, _, _ = binned_statistic(x,y,statistic='std',bins=bins,range=range)
    return (edges[1:]+edges[:-1])/2,mean,std


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16)/255 for i in range(0, lv, lv // 3))

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def create_multi_rgb(images,
                   colors=['#ff0000','#00ff00','#0000ff'],
                   percentile=None):
    '''combie multiple arrays to one RGB image
    
    Parameters 
    ----------
    
    images : list of images
    
    colors : the colors to which the images are assigned (default r,g,b)
    
    percentile : percentile that is used 
    '''
    
    if not percentile:
        percentile = [(0,100) for i in range(len(images))]
            
    colors_rgb = np.vstack([hex_to_rgb(x) for x in colors])

    # create an empty array with the correct size
    rgb = np.zeros((*images[0].shape,3))
    
    for i,image in enumerate(images):
        lim = np.nanpercentile(image,percentile[i])
        scaled_img = (image-lim[0]) / (lim[1]-lim[0])
        
        rgb[...,0] += scaled_img * colors_rgb[i,0] / np.sum(colors_rgb[...,0]) * colors_rgb[i,0] / np.sum(colors_rgb[i,...])
        rgb[...,1] += scaled_img * colors_rgb[i,1] / np.sum(colors_rgb[...,1]) * colors_rgb[i,1] / np.sum(colors_rgb[i,...])
        rgb[...,2] += scaled_img * colors_rgb[i,2] / np.sum(colors_rgb[...,2]) * colors_rgb[i,2] / np.sum(colors_rgb[i,...])

    # clip values (we use percentile for the normalization) and fill nan
    rgb = np.clip(np.nan_to_num(rgb,nan=1),a_min=0,a_max=1)
    
    return rgb
