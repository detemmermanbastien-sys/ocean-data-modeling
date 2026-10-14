"""
Hyperspectral Image Inversion for Coastal Water Quality Mapping
===============================================================
This module applies the Lee bio-optical model to hyperspectral satellite 
imagery (e.g., HICO) to retrieve key water quality parameters.

It uses non-linear least squares optimization (SciPy) to estimate:
- Chlorophyll-a concentration (C_chl)
- Non-Algal Particles / Sediments (C_nap)
- Colored Dissolved Organic Matter (C_cdom)
- Bathymetry (z)
- Bottom reflectance fraction (alpha)

Author: Bastien Detemmerman
"""

import numpy as np
from scipy.optimize import curve_fit

def lee_bio_optical_model(wl, C_chl, C_nap, C_cdom, z, alpha, aw, aphya, sand, algue):
    """
    Computes the theoretical remote sensing reflectance (RRS) based on the Lee model.
    """
    # Fixed model constants
    anap440 = 0.0048
    snap = 0.0106
    scdom = 0.0157
    lambdabase = 440
    bbnap = 0.0054
    yphy = 0.681
    bbphy = 0.00038
    ynap = 0.681
    lambdabase1 = 542
    tetaw = 30
    tetav = 0

    # Absorption coefficients
    aphy = C_chl * aphya
    anap = C_nap * anap440 * np.exp(-snap * (wl - lambdabase))
    acdom = C_cdom * np.exp(-scdom * (wl - lambdabase))
    absp = aw + anap + aphy + acdom

    # Backscattering coefficients
    bbw = 0.00144 * (wl / 500)**-4.32
    bbp = (C_chl * bbphy * (lambdabase1 / wl)**yphy) + (C_nap * bbnap * (lambdabase1 / wl)**ynap)
    coefretrodif = bbw + bbp

    # Light distribution and subsurface reflectance
    K = absp + coefretrodif
    u = coefretrodif / (absp + coefretrodif)
    up = bbp / (absp + coefretrodif)
    DuC = 1.03 * (1 + 2.4 * u)**0.5
    DuB = 1.04 * (1 + 5.4 * u)**0.5
    gp = 0.184 * (1 - 0.602 * np.exp(-3.805 * up))
    gw = 0.115
    rsdp = (gw * bbw / (absp + coefretrodif)) + (gp * bbp / (absp + coefretrodif))

    # Water column and bottom reflectance components
    rrsc = rsdp * (1 - np.exp(-((1 / np.cos(np.radians(tetaw))) + (DuC / np.cos(np.radians(tetav)))) * K * z))
    rhof = alpha * sand + (1 - alpha) * algue
    rrsb = (rhof / np.pi) * np.exp(-((1 / np.cos(np.radians(tetaw))) + (DuB / np.cos(np.radians(tetav)))) * K * z)

    # Total Remote Sensing Reflectance
    rrs = rrsc + rrsb
    RRS = (0.52 * rrs) / (1 - 1.56 * rrs)

    return RRS


def process_hyperspectral_image(ro_s, masque, wl, aw, aphya, sand, algue):
    """
    Iterates over a masked hyperspectral image to invert the Lee model pixel by pixel.
    Returns spatial maps of Chlorophyll, Sediments, CDOM, Depth, and Bottom Type.
    """
    rows, cols, bands = ro_s.shape
    carto = np.zeros((rows, cols, 6), dtype=float)

    # Initial guess: [C_chl, C_nap, C_cdom, z, alpha]
    X0 = [1.0, 1.0, 0.15, 20.0, 0.5]
    bounds_lower = [0, 0, 0, 0, 0]
    bounds_upper = [10, 10, 1, 50, 1]

    # Wrapper for curve_fit passing external spectral libraries
    def model_wrapper(wavelengths, *params):
        return lee_bio_optical_model(wavelengths, *params, aw=aw, aphya=aphya, sand=sand, algue=algue)

    print(f"Starting inversion for {rows}x{cols} grid...")

    for i in range(rows):
        for j in range(cols):
            if masque[i, j] == 1:  # Process only valid water pixels
                RRS_mesure = ro_s[i, j, :]
                try:
                    X_opt, _ = curve_fit(
                        model_wrapper,
                        wl, RRS_mesure,
                        p0=X0, bounds=(bounds_lower, bounds_upper), method='trf'
                    )
                    carto[i, j, 0:5] = X_opt
                    carto[i, j, 5] = 1 - X_opt[4] # Algae fraction
                except Exception as e:
                    # Handle optimization failure for specific pixels
                    carto[i, j, :] = np.nan

    return carto
