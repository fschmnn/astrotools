# Astrotools

A collection of python tools for astronomy. 

## Description

This package collects general purpose functions that are used in several projects of mine (e.g. [Planetary Nebula Luminosity Function](https://github.com/fschmnn/pnlf) or [Stellar associations powering HII regions](https://github.com/fschmnn/cluster)). Most of them are build on existing `astropy` or `scipy` functions and add certain functionality / shortcuts to make my work easier. 

* **regions**: handle regions in images (either masks or outlines) and re-project them to other images.

* **plot**: help function to create corner plots ...

  



## Installation

to install this package

```bash
python setup.py develop
```



### Required packages:

**[NumPy](https://numpy.org/)**

```
conda install numpy
```

**[matplotlib](https://matplotlib.org/)**

```
conda install matplotlib
```

**[astropy](https://www.astropy.org/)**

```
conda install astropy
```

**[scipy](https://scipy.org/)**

```
 conda install scipy 
```

**[Astropy Regions](https://astropy-regions.readthedocs.io/en/stable/index.html)**

```
conda install -c conda-forge regions
```

**[scikit-image](https://scikit-image.org/)**

``` 
conda install scikit-image
```

**[reproject](https://reproject.readthedocs.io/en/stable/#)**

```
conda install -c astropy reproject
```



## Usage



```python
import astrotools as tools

...
```
