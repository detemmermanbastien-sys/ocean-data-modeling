# Oceanographic Data Processing & Predictive Modeling

This repository contains algorithms developed during my research to process oceanographic data, integrate hydrodynamic models, and deploy machine learning predictions.

## Pipeline Architecture
This project was originally deployed as a MATLAB operational dashboard. The underlying core pipeline consists of:

1. Data Ingestion & Alignment: Extraction, spatial masking, and time-alignment of NetCDF data (CMEMS, Ifremer MARC).
2. Physical Oceanography Engine: Calculation of oceanographic metrics such as Mixed Layer Depth (MLD) and Euphotic Zone (ZE1%).
3. Machine Learning Core: XGBoost integration via Python bridge for spatio-temporal predictions, evaluated by a "Leave-One-24h-Out" cross-validation.

##  Core Scripts Included

### 1. `extract_ocean_data.m`
Function to handle NetCDF datasets from **Copernicus** (CMEMS) and **Ifremer MARC** models. 
- Manages epoch time conversions (1900/1950/1970).
- Aligns multi-source spatial grids with in-situ temporal measurements.

### 2. `train_xgboost_kfold.m`
Spatio-temporal machine learning pipeline using XGBoost.
- Implements a MATLAB-to-Python bridge (`py.importlib`) to leverage Python ML libraries within a MATLAB workflow.
- Features a Leave-One-24h-Out cross-validation to prevent temporal data leakage.

- ### 3. `calculate_baroclinic_modes.m`
A fundamental physical oceanography engine solving eigenvalue problems to determine baroclinic modes for horizontal and vertical velocities.
- Discretizes double derivative equations and calculates the Deformation Radius (Rd).
- Includes structural coherency checks against WKB approximations.

- ### 4. `hyperspectral_inversion.py`
A native Python module utilizing `scipy.optimize` to invert the Lee bio-optical model on hyperspectral satellite imagery (e.g., HICO). 
- Employs non-linear least squares (`curve_fit`) to retrieve coastal water quality parameters from surface reflectance.
- Generates spatial maps of Chlorophyll-a, Sediments (NAP), CDOM, and shallow bathymetry.

- ### 5. `pollutant_dispersion_solver.m`
A custom 2D hydrodynamic dispersion module demonstrating the fundamental numerical methods used in commercial solvers (e.g., MOHID, TELEMAC).
- Eulerian Advection: Implements an Upwind scheme on an Arakawa-C grid to model continuous pollutant concentration fields (e.g., oil spills) while ensuring mass conservation.
- Lagrangian Tracking: Utilizes a 4th-order Runge-Kutta (RK4) integration for highly precise discrete particle tracking across the modeled current velocity fields.

- ### 6. `barotropic_streamfunction_solver.m`
A numerical solver for wind-driven barotropic ocean circulation based on the streamfunction elliptic equation.
- Implements a finite difference method with an iterative relaxation scheme.
- Demonstrates rigorous, manual mathematical handling of spatial boundary conditions (coastal and open-ocean edges).

## Tech Stack
- Languages: MATLAB, Python (NumPy, SciPy, XGBoost)
- Data Types: NetCDF, Oceanographic profiles (CTD, ADCP), Hydrodynamic model outputs.
