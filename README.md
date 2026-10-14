# Oceanographic Data Processing & Predictive Modeling

This repository contains algorithms developed during my research to process oceanographic data, integrate hydrodynamic models, and deploy machine learning predictions.

## Pipeline Architecture
This project was originally deployed as a MATLAB operational dashboard. The underlying core pipeline consists of:

1. **Data Ingestion & Alignment:** Extraction, spatial masking, and time-alignment of NetCDF data (CMEMS, Ifremer MARC).
2. **Physical Oceanography Engine:** Calculation of oceanographic metrics such as Mixed Layer Depth (MLD) and Euphotic Zone (ZE1%).
3. **Machine Learning Core:** XGBoost integration via Python bridge for spatio-temporal predictions, evaluated by a "Leave-One-24h-Out" cross-validation.

##  Core Scripts Included

### 1. `extract_ocean_data.m`
Function to handle NetCDF datasets from **Copernicus** (CMEMS) and **Ifremer MARC** models. 
- Manages epoch time conversions (1900/1950/1970).
- Aligns multi-source spatial grids with in-situ temporal measurements.

### 2. `train_xgboost_kfold.m`
Spatio-temporal machine learning pipeline using **XGBoost**.
- Implements a MATLAB-to-Python bridge (`py.importlib`) to leverage Python ML libraries within a MATLAB workflow.
- Features a **Leave-One-24h-Out cross-validation** to prevent temporal data leakage.

## Tech Stack
- **Languages:** MATLAB, Python (NumPy, XGBoost)
- **Data Types:** NetCDF, Oceanographic profiles (CTD, ADCP), Hydrodynamic model outputs.

- ### 3. `calculate_baroclinic_modes.m`
A fundamental physical oceanography engine solving eigenvalue problems to determine baroclinic modes for horizontal and vertical velocities.
- Discretizes double derivative equations and calculates the Deformation Radius (Rd).
- Includes structural coherency checks against WKB approximations.
