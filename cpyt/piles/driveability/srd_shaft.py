# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:32:51 2019

#TODO: 
    ## Integral to code function
    - Verify De Beer method
    - In De Beer, the depth of penetration is calculated for the CPT. This should be directly in the CPTScrape function, however.
    
    ## Nice addition
    - Incorporate distance function which gets distance between CPTs and pile locations
    - Automated way of obtaining beta
    - Add in pile top into the calculations, don't forget pile_depth is w.r.t NAP, pile length is longer'
    - Could we use a different friction angle for the De Beer function?
    
POTENTIAL ISSUES
    - The UWA-05 is currently using the Lengkeek UW correlation for obtaining sig_eff, not the NEN9997 Rule.

#REFERENCES
NEN6742 5.4.2.2.1

All functions have the same structure:
    (1) Set variables relating to pile geometry
    (2) Check correlations are present in CPT and trim CPT to required segment
    (3) Derive parameters for relevant equation
    (4) Execute calculation
    (5) Prepare results for output
    
@author: kduffy
"""

import pandas as pd, numpy as np
import matplotlib.pyplot as plt

from cpyt.interpretation import correlations,classify
from cpyt.processing.read_cpt import CPT

#%% Alm & Hamre
def alm_and_hamre(cpt, z_top_inc, z_base_inc, z_base, closed_ended=True, 
            material="concrete",
            h=None, b=None,
            D_inner = None, D_outer=None):
    """
    From Lehane et al., 2022
    
    Function which calculates the pile base capacity based on:
        :cpt:           Input dataframe of CPT data
        :z_top:         Elevation of pile head relative to CPT
        :z_base:        Elevation of pile base relative to CPT
        :d_cpt:         Diameter of the CPT cone [m]
        
    NOTE: Does not account for the contribution of the pile plug
    """    
    print("_________________________________________")
    print("Calculating shaft capacity using the Unified design method for clay ...\n")
    
    if "sig_eff" not in cpt.columns:
        raise ValueError("The Alm & Hamre SRD method needs sig_eff as an input.\nSee cpyt.correlations")

    segment_length = z_top_inc - z_base_inc
    cpt = cpt.loc[(cpt.z <= z_top_inc) & (cpt.z >= z_base_inc)]
    
    ## Pile Geometry
    if (h==None) & (b==None):       # i.e. if the pile is circular
        circum = np.pi*D_outer
        surf_area = circum*(segment_length)                    # Surface area of entire pile shaft
    if D_outer == None:                               # i.e. if the pile is rectangular
        circum = 2*h + 2*b
        area = h*b
        surf_area = 2*h*segment_length + 2*b*segment_length
     
    h = cpt.z - z_base
    delta_f_dict = {"concrete": 29,"steel": 29}
    delta_f = delta_f_dict[material]    # degrees
        
    # Main calculations
    pa = 101    # [kN/m2]
    k = ((cpt.qc/cpt.sig_eff)**0.5)/80
    K = 0.0132*cpt.qc*((cpt.sig_eff/pa)**0.13)/cpt.sig_eff
    
    qs_max = K*cpt.sig_eff*np.tan(np.radians(delta_f))
    qs_res = 0.2*qs_max
    
    cpt["qs"] = qs_res + (qs_max - qs_res) * np.exp(-k*h)
    
    # Return result
    cpt["depth_diff"] = abs(cpt.z.diff())                   # Depth difference between CPT soundings
    cpt["Qsmax_contrib"] = cpt.depth_diff*circum*cpt.qs     # Contribution of each ~2cm layer to total shear resistance
    cpt["Qsmax"] = cpt.Qsmax_contrib.cumsum()

    cpt.reset_index(drop=True,inplace=True)
    near_z_ix = cpt.z.sub(z_base).abs().idxmin()        # Index of row with depth closest to pile depth
    Qsmax = cpt.Qsmax.iloc[near_z_ix]                   # Shaft capacity in kN
    qsmax = Qsmax/surf_area                             # Shaft capacity in kPa
    
    print(f"The pile shaft SRD is {Qsmax:.2f} kN")
    print("_________________________________________")
    return Qsmax

def unified_sand(cpt, z_top_inc, z_base_inc, z_base, closed_ended=True, 
            material="concrete",
            h=None, b=None, d_cpt=35.7E-3,
            D_inner = None, D_outer=None, 
            exclude_friction_fatigue=False):
    """
    From Lehane et al., 2022
    
    Function which calculates the pile base capacity based on:
        :cpt:           Input dataframe of CPT data
        :z_top:         Elevation of pile head relative to CPT
        :z_base:        Elevation of pile base relative to CPT
        :d_cpt:         Diameter of the CPT cone [m]
        
    NOTE: Does not account for the contribution of the pile plug
    """    
    print("_________________________________________")
    print("Calculating shaft capacity using the Unified design method for clay ...\n")
    
    if "sig_eff" not in cpt.columns:
        raise ValueError("The Unified SRD method needs sig_eff as an input.\nSee cpyt.correlations")

    segment_length = z_top_inc - z_base_inc
    cpt = cpt.loc[(cpt.z <= z_top_inc) & (cpt.z >= z_base_inc)]
    
    ## Pile Geometry
    if (h==None) & (b==None):       # i.e. if the pile is circular
        D_eq = D_outer
        circum = np.pi*D_outer
        surf_area = circum*(segment_length)                    # Surface area of entire pile shaft
    if D_outer == None:                               # i.e. if the pile is rectangular
        circum = 2*h + 2*b
        area = h*b
        D_eq = ((area/np.pi)**0.5)*2
        surf_area = 2*h*segment_length + 2*b*segment_length
     
    h_D = (cpt.z - z_base)/D_eq
    delta_f_dict = {"concrete": 29,"steel": 29}
    delta_f = delta_f_dict[material]    # degrees
    
    if closed_ended == True:
        A_re = 1
    else:
        PLR = np.tanh(0.3*(D_inner/d_cpt)**0.5)
        A_re = 1 - PLR*(D_inner/D_eq)**2
    
    # Main calculations
    if exclude_friction_fatigue:
        h_D = 1
    else:  
        h_D.loc[h_D < 1] = 1    # i.e. max(1, h/D)
        
    sig_eff_rc = ((cpt.qc*1000)/44)*(A_re**0.3)*(h_D)**-0.4
    sig_eff_rd = ((cpt.qc*1000)/10)*(((cpt.qc*1000)/cpt.sig_eff)**-0.33)*(d_cpt/D_eq)
    cpt["qs"] = (sig_eff_rc + sig_eff_rd)*np.tan(np.radians(delta_f))
    
    # Return result
    cpt["depth_diff"] = abs(cpt.z.diff())                   # Depth difference between CPT soundings
    cpt["Qsmax_contrib"] = cpt.depth_diff*circum*cpt.qs     # Contribution of each ~2cm layer to total shear resistance
    cpt["Qsmax"] = cpt.Qsmax_contrib.cumsum()

    cpt.reset_index(drop=True,inplace=True)
    near_z_ix = cpt.z.sub(z_base).abs().idxmin()        # Index of row with depth closest to pile depth
    Qsmax = cpt.Qsmax.iloc[near_z_ix]                   # Shaft capacity in kN
    qsmax = Qsmax/surf_area                             # Shaft capacity in kPa
    
    print(f"The pile shaft SRD is {Qsmax:.2f} kN")
    print("_________________________________________")
    return Qsmax
    

