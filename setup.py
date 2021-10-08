
long_description = f"""
Python tools for astronomy

... 

"""

from setuptools import setup, find_namespace_packages

setup(name='astrotools',
      version='0.1',
      author='Fabian Scheuermann',
      author_email='f.scheuermann@uni-heidelberg.de',
      license='MIT',
      package_dir={"": "src"},
      packages=find_namespace_packages(where="src"),
      description='useful tools for astronomy',
      long_description = long_description)