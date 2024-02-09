# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 08:32:15 2020

@author: kevin

This deals with the dataframe output of the CPT class.
"""
import pandas as pd, numpy as np
from cpyt.interpretation import correlations

#%%
def best_possible_method(cpt):
    """
    Returns the best possible classification method based on the CPT results available
    """
    if "fs" in cpt.columns:
        cpt = Rf_based(cpt)
    else:
        cpt = qc_based(cpt)
    
    return cpt
    
def qc_based(cpt,incl_silt=False):
    """
    :incl_silt:         Classify silty soils as well
    """
    cpt["soil_type"] = np.nan
    
    cpt.loc[cpt.qc < 2,"soil_type"] = "clay"
    cpt.loc[cpt.qc > 2,"soil_type"]= "sand"

    return cpt

def Rf_based(cpt,incl_silt=False):
    cpt["soil_type"] = np.nan
    if "Rf" not in cpt:
        cpt["Rf"] = (cpt.fs/(cpt.qc*1000))*100
    cpt.loc[cpt.Rf < 2,"soil_type"] = "sand"
    cpt.loc[cpt.Rf > 2,"soil_type"] = "clay"
    
    return cpt

