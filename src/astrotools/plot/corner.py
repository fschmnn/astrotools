import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.stats import spearmanr, gaussian_kde, binned_statistic, binned_statistic_2d
from scipy.interpolate import interpn

from .utils import fix_aspect_ratio, bin_stat


def corner_scatter(x,y,ax,**kwargs):
    '''a simple scatter plot'''
    
    ax.scatter(x,y,**kwargs)

    return 0 

def corner_binned_stat(x,y,ax,nbins=10,**kwargs):
    '''a scatter plot with binned trends'''

    xlim = ax.get_xlim()

    #ax.scatter(x,y,**kwargs)

    # calculate spearmann correlation
    not_nan = ~np.isnan(x) & ~np.isnan(y)
    r,p = spearmanr(x[not_nan],y[not_nan])

    x,mean,std = bin_stat(x,y,xlim,nbins=nbins,statistic='median')
    ax.errorbar(x,mean,yerr=std,fmt='-',color='gray',lw=0.5)

    return 0

def corner_density_scatter(x,y,ax,nbins=10,**kwargs):
    '''scatter plot with point density from histogram'''

    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    
    hist, x_e, y_e = np.histogram2d(x,y,bins=nbins,range=[xlim,ylim],density=True)
    z = interpn((0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ),hist,np.vstack([x,y]).T,method="nearest",bounds_error=False)
    sc=ax.scatter(x,y,c=z,**kwargs)

    return 0

def corner_density_histogram(x,y,ax,nbins=10,**kwargs):
    '''2d histogram of point density
    
    https://stackoverflow.com/a/53865762
    '''

    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    
    hist, x_e, y_e = np.histogram2d(x,y,bins=nbins,range=[xlim,ylim],density=True)
    hist[hist==0]  = np.nan
    im = ax.imshow(hist.T,origin='lower',extent=[*xlim,*ylim],**kwargs)

    return 0

def corner_gaussian_kde_scatter(x,y,ax,**kwargs):
    '''use a gaussian kernel density estimate for the point density'''

    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    sc=ax.scatter(x,y,c=z,cmap=plt.cm.Reds,**kwargs)

    return 0

def corner_binned_stat2d(x,y,ax,z,nbins=10,**kwargs):
    '''
    
    '''

    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    
    x,y,z = x[~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)], y[~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)], z[~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)]

    stat ,x_e, y_e, _ = binned_statistic_2d(x,y,z,bins=nbins,range=[xlim,ylim])
    z = interpn((0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ),stat,np.vstack([x,y]).T,method="nearest",bounds_error=False)
    sc=ax.scatter(x,y,c=z,**kwargs)


    return 0

def corner_binned_stat2d_histogram(x,y,ax,z,nbins=10,**kwargs):
    '''
    
    '''

    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    
    x,y,z = x[~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)], y[~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)], z[~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)]

    stat ,x_e, y_e, _ = binned_statistic_2d(x,y,z,bins=nbins,range=[xlim,ylim])
    im = ax.imshow(stat.T,origin='lower',extent=[*xlim,*ylim],**kwargs)

    return 0

def corner_binned_percentile(x,y,ax,nbins=10,n=68,**kwargs):

    x,y = x[~np.isnan(x) & ~np.isnan(y)], y[~np.isnan(x) & ~np.isnan(y)]

    pl, edges, _ = binned_statistic(x,y,statistic=lambda data: np.percentile(data,50+n/2),bins=nbins)
    pu, edges, _ = binned_statistic(x,y,statistic=lambda data: np.percentile(data,50-n/2),bins=nbins)
    xp = (edges[1:]+edges[:-1])/2
    ax.fill_between(xp,y1=pl,y2=pu,**kwargs)

def corner(table,columns,function=None,limits={},labels={},filename=None,figsize=10,aspect_ratio=1,**kwargs):
    '''Create a pairwise plot for all names in columns
    
    Parameters
    ----------
    
    table : table with the data
    
    columns : list of the columns that should be used
    
    limits : dictionary with the limits (tuple) for the columns

    labels : dictionary with the axis labels 

    filename : name of the file to save to
    '''
    
    # create a figure with the correct proportions
    fig, axes = plt.subplots(nrows=len(columns)-1,ncols=len(columns),figsize=(figsize,aspect_ratio*(len(columns)/(len(columns)+1))*figsize))

    for i,row in enumerate(columns[1:]):
        for j,col in enumerate(columns):
            ax=axes[i,j]
            if j>i:
                ax.remove()
            else:
                if j==0:
                    ax.set_ylabel(labels.get(row,row.replace("_","")))
                else:
                    ax.set_yticklabels([])
                if i==len(columns)-2:
                    ax.set_xlabel(labels.get(col,col.replace("_","")))
                else:
                    ax.set_xticklabels([])

                # we set the axis limit before we plot the data (some functions might need the limits)
                ax.set(xlim=limits.get(col,[np.min(table[col]),np.max(table[col])]),ylim=limits.get(row,[np.min(table[row]),np.max(table[row])]))

                # make sure there are no NaN in the data points
                x,y = table[col],table[row]
                
                # either use a special function for the plot or just use a scatter plot
                if function:
                    function(x,y,ax,**kwargs)
                else:
                    ax.scatter(x,y,**kwargs)

            fix_aspect_ratio(ax,aspect_ratio=aspect_ratio)

    plt.subplots_adjust(wspace=0.12, hspace=0.12)
    
    if filename:
        plt.savefig(filename,dpi=600)

    plt.show()


def corner_old(table,columns,limits={},labels={},colors=None,nbins=10,one_to_one=False,filename=None,
          figsize=10,aspect_ratio=1,**kwargs):
    '''Create a pairwise plot for all names in columns
    
    Parameters
    ----------
    
    table : table with the data
    
    columns : name of the columns ought to be used
    
    limits : dictionary with the limits (tuple) for the columns
    filename : name of the file to save to
    '''
    
    # create a figure with the correct proportions
    fig, axes = plt.subplots(nrows=len(columns)-1,ncols=len(columns),figsize=(figsize,aspect_ratio*(len(columns)/(len(columns)+1))*figsize))

    #gs = axes[1,1].get_gridspec()


    for i,row in enumerate(columns[1:]):
        for j,col in enumerate(columns):
            ax=axes[i,j]
            if j>i:
                ax.remove()
            else:
                if j==0:
                    ax.set_ylabel(labels.get(row,row.replace("_","")))
                else:
                    ax.set_yticklabels([])
                if i==len(columns)-2:
                    ax.set_xlabel(labels.get(col,col.replace("_","")))
                else:
                    ax.set_xticklabels([])

                
                x,y = table[col],table[row]
                x,y = x[~np.isnan(x) & ~np.isnan(y)], y[~np.isnan(x) & ~np.isnan(y)]
                
                
                xy = np.vstack([x,y])
                z = gaussian_kde(xy)(xy)
                idx = z.argsort()
                x, y, z = x[idx], y[idx], z[idx]
                sc=ax.scatter(x,y,c=z,cmap=plt.cm.Reds,**kwargs)

                #density_scatter(x,y,ax)

                xlim = limits.get(col,None)
                ylim = limits.get(row,None)
                if xlim: 
                    x,mean,std = bin_stat(table[col],table[row],xlim,nbins=nbins,statistic='median')
                    ax.errorbar(x,mean,yerr=std,fmt='-',color='gray',lw=0.5)
                    ax.set_xlim(xlim)

                if ylim: ax.set_ylim(ylim)

                #
                if one_to_one:
                    if not xlim:
                        xlim = ax.get_xlim()
                    if not ylim:
                        ylim = ax.get_ylim()
                    lim = np.min([xlim[0],ylim[0]]),np.max([xlim[1],ylim[1]])
                    ax.plot(lim,lim,color='black')
                    ax.set(xlim=lim,ylim=lim)

                # calculate spearmann correlation
                not_nan = ~np.isnan(table[col]) & ~np.isnan(table[row])
                r,p = spearmanr(table[col][not_nan],table[row][not_nan])
    
                #t = ax.text(0.07,0.85,r'$\rho'+f'={r:.2f}$',transform=ax.transAxes,fontsize=7)
                #t.set_bbox(dict(facecolor='white', alpha=1, ec='white'))

                #ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
                #ax.yaxis.set_minor_locator(mpl.ticker.MaxNLocator(4))

            # make sure the aspect ratio of each subplot is 1
            #ax.set_aspect(aspect_ratio*np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))
            fix_aspect_ratio(ax,aspect_ratio=aspect_ratio)


    #axbig = fig.add_subplot(gs[1,2:])
    #fig.colorbar(sc,cax=axbig,label='density',ticks=[],orientation='horizontal')

    plt.subplots_adjust(wspace=0.12, hspace=0.12)
    
    if filename:
        plt.savefig(filename,dpi=600)

    plt.show()