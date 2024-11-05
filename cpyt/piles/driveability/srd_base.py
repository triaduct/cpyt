5# -*- coding: utf-8 -*-
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

@author: kduffy
"""

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import os, sys

from cpyt.piles.axialcapacity import averaging_methods
from cpyt.processing.read_cpt import CPT
import cpyt.interpretation.correlations

#%%
def alm_and_hamre(cpt,z_base, closed_ended=True, 
            alpha_p = None, avg_method = None,
            h=None, b=None, 
            D_inner = None, D_outer=None):
    """
    
    """
    if "sig_eff" not in cpt.columns:
        raise ValueError("The Alm & Hamre SRD method needs sig_eff as an input.\nSee cpyt.correlations")
    
    ## Pile Geometry
    if (h==None) & (b==None):       # i.e. if the pile is circular
        D = D_outer
        circum = 2*np.pi*(D_outer/2)
        area = np.pi*(D_outer/2)**2
        t = (D_outer-D_inner)*0.5   # Wall thickness
    if D_outer == None:                               # i.e. if the pile is rectangular
        circum = 2*h + 2*b
        area = h*b
        D = ((area/np.pi)**0.5)*2
        
    near_z_ix = cpt.z.sub(z_base).abs().idxmin()       # Index of row with depth closest to pile depth
    sig_eff = cpt.loc[near_z_ix, "sig_eff"]
    
    if avg_method == None:       # Get just qc at the pile tip
        qc_tip = cpt.loc[near_z_ix, "qc"]
    elif avg_method == "lcpc":
        qc_tip = averaging_methods.lcpc(cpt, D, z_base)      # qp = qc,avg
    elif avg_method == "koppejan":
        qc_tip = averaging_methods.koppejan(cpt, D, z_base, plot_min_path = False)
    elif avg_method == "de_boorder":
        qc_tip = averaging_methods.de_boorder(cpt, D, z_base)
    elif avg_method == "boulanger_dejong":
        raise ValueError("Check that Boulanger De Jong can be included")

    q_ann = 0.15*qc_tip*(qc_tip*1000/sig_eff)**0.2
    Qb = q_ann * np.pi * D * t

    print(f"The pile base SRD at {z_base:.2f} m is {Qb:.2f} kN")
    print("_________________________________________")
    return Qb


def unified_sand(cpt,z_base, closed_ended=True, 
            avg_method = None,
            h=None, b=None, d_cpt=35.7E-3,
            D_inner = None, D_outer=None):
    """
    Specify equivalent diameter area if square pile
    
    From Lehane et al., 2022
    
    Function which calculates the pile base capacity based on:
        :cpt:           Input dataframe of CPT data
        :z_top:         Elevation of pile head relative to CPT
        :z_base:        Elevation of pile base relative to CPT
        :d_cpt:         Diameter of the CPT cone [m]
        
    Due to the inertia of the pile lplug, a phase shift occurs between the pile wall
    and the internal soil plug. Therefore, piles greater than 1.5m in diameter are
    typically full-coring. For piles with 0.75m <= D <= 1.5m, the plug length ratio (PLR)
    is used.     
    """
    print("_________________________________________")
    print("Calculating the base SRD using the Unified method for sand...\n")
    
    ## Pile Geometry
    if (h==None) & (b==None):       # i.e. if the pile is circular
        D = D_outer
        circum = 2*np.pi*(D_outer/2)
        area = np.pi*(D_outer/2)**2
        t = (D_outer-D_inner)*0.5   # Wall thickness
    if D_outer == None:                               # i.e. if the pile is rectangular
        circum = 2*h + 2*b
        area = h*b
        D = ((area/np.pi)**0.5)*2
    
    if closed_ended == True:
        A_re = 1
    else:
        PLR = np.tanh(0.3*(D_inner/d_cpt)**0.5)
        A_re = 1 - PLR*(D_inner/D)**2
    
    t = (D_outer-D_inner)*0.5
    D_eq = D*A_re**0.5
    
    if avg_method == None:       # Get just qc at the pile tip
        near_z_ix = cpt.z.sub(z_base).abs().idxmin()       # Index of row with depth closest to pile depth
        qc_tip = cpt.loc[near_z_ix, "qc"]
    elif avg_method == "lcpc":
        qc_tip = averaging_methods.lcpc(cpt, D_eq, z_base)      # qp = qc,avg
    elif avg_method == "koppejan":
        qc_tip = averaging_methods.koppejan(cpt, D_eq, z_base, plot_min_path = False)
    elif avg_method == "de_boorder":
        qc_tip = averaging_methods.de_boorder(cpt, D_eq, z_base)
    elif avg_method == "boulanger_dejong":
        raise ValueError("Check that Boulanger De Jong can be included")
    
    qb = min(0.4*qc_tip*(np.exp(-2*PLR) + 4*(t/D)), 0.4*qc_tip)
    
    Qb = qb*1000*((np.pi*D**2)/4)      # Max pile tip capacity [kN]

    print(f"The pile base SRD at {z_base:.2f} m is {Qb:.2f} kN")
    print("_________________________________________")
    return Qb


