# Quick-start Guide to FurgeHullam

## Setup
Once we have a conda environment with `enterprise` and `enterprise_extensions` installed, `FurgeHullam` can be locally pip installed. Below is a potential setup from scratch.

Create new `conda` environment:
```
conda create --name FurgeHullam python=3.9
```
Active our new environment:
```
conda activate FurgeHullam
```
Install `enterprise_extensions` via `conda-forge`, which also installs `enterprise` as a requirement:
```
conda install -c conda-forge enterprise_extensions
```
Pip install `FurgeHullam` via `git`:
```
pip install git+https://github.com/bencebecsy/FurgeHullam.git
```

## Running FurgeHullam
Once installed, `FurgeHullam` can be used in two stages. First, one needs to set up a grid for the inner product interpolations. This is the most expensive part of the analysis. Once this is done, the likelihood, phase-marginalized likelihood, and distance-and-phase-marginalized likelihood can be calculated very quickly and can be used in subsequent analysis as needed. The setup can also be very quickly updated with new data as long as the pulsars, their observing epochs, and noise remains the same (can be really useful for many realizations of simulate data). Details on how to do all this in code can be found in this notebook:
[Setup and Run Tutorial](https://github.com/bencebecsy/FurgeHullam/blob/main/docs/run_FurgeHullam.ipynb)
