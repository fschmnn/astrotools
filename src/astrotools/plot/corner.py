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

def corner_binned_stat(x,y,ax,bins=10,color='black',**kwargs):
    '''a scatter plot with binned trends in y'''

    xlim = ax.get_xlim()
    #x,mean,std = bin_stat(x,y,bins=bins,range=xlim,statistic='median')
    
    # just ignore nan values
    x, y = x[np.isfinite(x) & np.isfinite(y)], y[np.isfinite(x) & np.isfinite(y)]
    mean, edges, _ = binned_statistic(x,y,statistic='median',bins=bins,range=xlim)
    x = (edges[1:]+edges[:-1])/2
    
    ax.errorbar(x,mean,yerr=None,fmt='-',color=color,**kwargs)

    return 0

def corner_density_scatter(x,y,ax,nbins=10,**kwargs):
    '''scatter plot with point density from histogram'''

    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    
    hist, x_e, y_e = np.histogram2d(x,y,bins=nbins,range=[xlim,ylim],density=True)
    z = interpn((0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ),hist,np.vstack([x,y]).T,method="linear",bounds_error=False)
    vmin,vmax=np.nanpercentile(z,[15,95])
    sc=ax.scatter(x,y,c=z,vmin=vmin,vmax=vmax,**kwargs)

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
    sc=ax.scatter(x,y,c=z,**kwargs)

    return 0

def corner_binned_stat2d(x,y,ax,z,nbins=10,**kwargs):
    '''make a scatter plot of x and y with the color based on binned stats of z'''

    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    
    x,y,z = x[~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)], y[~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)], z[~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)]

    stat ,x_e, y_e, _ = binned_statistic_2d(x,y,z,bins=nbins,range=[xlim,ylim])
    z = interpn((0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ),stat,np.vstack([x,y]).T,method="nearest",bounds_error=False)
    sc=ax.scatter(x,y,c=z,**kwargs)


    return 0

def corner_binned_stat2d_histogram(x,y,ax,z,nbins=10,**kwargs):
    '''show z binned based on x and y'''


    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    
    x,y,z = x[~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)], y[~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)], z[~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)]

    stat ,x_e, y_e, _ = binned_statistic_2d(x,y,z,bins=nbins,range=[xlim,ylim])
    im = ax.imshow(stat.T,origin='lower',extent=[*xlim,*ylim],**kwargs)

    return 0

def corner_binned_percentile(x,y,ax,nbins=10,range=None,n=68,**kwargs):
    '''plot the n percentile of y along the xaxis'''

    x,y = x[~np.isnan(x) & ~np.isnan(y)], y[~np.isnan(x) & ~np.isnan(y)]

    pl, edges, _ = binned_statistic(x,y,statistic=lambda data: np.percentile(data,50+n/2),bins=nbins,range=range)
    pu, edges, _ = binned_statistic(x,y,statistic=lambda data: np.percentile(data,50-n/2),bins=nbins,range=range)
    xp = (edges[1:]+edges[:-1])/2
    ax.fill_between(xp,y1=pl,y2=pu,**kwargs)


def corner_violin(x,y,ax,positions,**kwargs):
    '''create violin plots for the data'''

    #x,y = x[~np.isnan(y)],y[~np.isnan(y)]
    
    bins = (positions[1:]+positions[:-1])/2
    binned_data = [y[(bins[i]<x) & (x<bins[i+1])] for i in range(len(bins)-1)]

    ax.violinplot(binned_data,positions=positions[1:-1],**kwargs)
    
    return binned_data

def corner_spearmanr(x,y,ax,position=(0.93,0.93),pvalue=False,**kwargs):
    '''calculate Spearman correlation coefficient'''

    not_nan = ~np.isnan(x) & ~np.isnan(y)
    r,p = spearmanr(x[not_nan],y[not_nan])

    label = r'$\rho'+f'={r:.2f}$'
    if pvalue: label+=f', p={100*p:.3f}\%'
    t = ax.text(*position,label,transform=ax.transAxes,ha='right',va='top',**kwargs)
    t.set_bbox(dict(facecolor='white', alpha=1, ec='white'))


def corner(table,columns,function=None,histogram=False,limits={},labels={},scale={},filename=None,figsize=10,aspect_ratio=1,**kwargs):
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
    nrows, ncols = len(columns)-1, len(columns)-1

    if histogram:
        nrows+=1
        ncols+=1

    fig, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(figsize,figsize*aspect_ratio*ncols/nrows))
    for i,row in enumerate(columns[1:]):
        for j,col in enumerate(columns[:-1]):
            ax=axes[i,j]
            if j>i:
                ax.remove()
            else:
                # we need to set the scale before 
                ax.set(xscale=scale.get(col,'linear'),yscale=scale.get(row,'linear'))
                if j==0:
                    ax.set_ylabel(labels.get(row,row.replace("_","")))
                else:
                    ax.set_yticklabels([])
                if i==len(columns)-2:
                    ax.set_xlabel(labels.get(col,col.replace("_","")))
                else:
                    ax.set_xticklabels([])
     

                # we set the axis limit before we plot the data (some functions might need the limits)
                xlim = limits.get(col,[np.min(table[col]),np.max(table[col])])
                ylim = limits.get(row,[np.min(table[row]),np.max(table[row])])
                ax.set(xlim=xlim,ylim=ylim)

                # make sure there are no NaN in the data points
                x,y = table[col],table[row]
                x,y = x[np.isfinite(x) & np.isfinite(y)], y[np.isfinite(x) & np.isfinite(y)]
                #x,y = x[(xlim[0]<=x) & (x<=xlim[1]) & (ylim[0]<=y) & (y<=ylim[1])],y[(xlim[0]<=x) & (x<=xlim[1]) & (ylim[0]<=y) & (y<=ylim[1])]
                
                # either use a special function for the plot or just use a scatter plot
                if function:
                    function(x,y,ax,xlim=xlim)
                else:
                    ax.scatter(x,y,**kwargs)
            fix_aspect_ratio(ax,aspect_ratio=aspect_ratio)


    plt.subplots_adjust(wspace=0.12, hspace=0.12)
    
    if filename:
        plt.savefig(filename,dpi=600)

    plt.show()



def corner_by_galaxy(table,sample_table,columns,limits={},labels={},nbins=7,filename=None,figsize=10,aspect_ratio=1,**kwargs):
    '''Create a pairwise plot for all names in columns
    
    Parameters
    ----------
    
    table : table with the data
    
    columns : list of the columns that should be used
    
    limits : dictionary with the limits (tuple) for the columns

    labels : dictionary with the axis labels 

    filename : name of the file to save to
    '''
    
    cmap = mpl.cm.get_cmap('plasma')
    norm = mpl.colors.Normalize(vmin=9.4,vmax=11)

    # create a figure with the correct proportions
    nrows, ncols = len(columns)-1, len(columns)-1

    fig, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(figsize,figsize*aspect_ratio*ncols/nrows))
    for i,row in enumerate(columns[1:]):
        for j,col in enumerate(columns[:-1]):
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
                xlim = limits.get(col,[np.min(table[col]),np.max(table[col])])
                ylim = limits.get(row,[np.min(table[row]),np.max(table[row])])
                ax.set(xlim=xlim,ylim=ylim)

                for k,gal_name in enumerate(np.unique(table['gal_name'])):
                    
                    tmp = table[table['gal_name']==gal_name]

                    # make sure there are no NaN in the data points
                    x,y = tmp[col],tmp[row]
                    x,y = x[(xlim[0]<=x) & (x<=xlim[1]) & (ylim[0]<=y) & (y<=ylim[1])],y[(xlim[0]<=x) & (x<=xlim[1]) & (ylim[0]<=y) & (y<=ylim[1])]
                    x, y = x[~np.isnan(y) & np.isfinite(y)], y[~np.isnan(y) & np.isfinite(y)]

                    if len(x)>0:
                        mean, edges, _ = binned_statistic(x,y,statistic='median',bins=nbins,range=xlim)
                        std, _, _ = binned_statistic(x,y,statistic='std',bins=nbins,range=xlim)
                        x,mean,std = (edges[1:]+edges[:-1])/2,mean,std
                        ax.errorbar(x,mean,fmt='o-',ms=1.5,color=cmap(norm(sample_table.loc[gal_name]['mass'])),label=gal_name)

                x,y = table[col],table[row]
                x,y = x[(xlim[0]<=x) & (x<=xlim[1]) & (ylim[0]<=y) & (y<=ylim[1])],y[(xlim[0]<=x) & (x<=xlim[1]) & (ylim[0]<=y) & (y<=ylim[1])]
                x, y = x[~np.isnan(y) & np.isfinite(y)], y[~np.isnan(y) & np.isfinite(y)]

                mean, edges, _ = binned_statistic(x,y,statistic='median',bins=7,range=xlim)
                std, _, _ = binned_statistic(x,y,statistic='std',bins=7,range=xlim)
                x,mean,std = (edges[1:]+edges[:-1])/2,mean,std
                ax.errorbar(x,mean,fmt='o-',ms=1.5,color='black',label=gal_name)

            fix_aspect_ratio(ax,aspect_ratio=aspect_ratio)


    plt.subplots_adjust(wspace=0.12, hspace=0.12)
    
    if filename:
        plt.savefig(filename,dpi=600)

    plt.show()
