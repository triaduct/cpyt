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

from cpyt.interpretation import correlations,classify_cpt
from cpyt.processing.read_cpt import CPT


def nen9997_2017(cpt, pile_type, z_top, z_base, h=np.nan, b=np.nan, pile_dia=np.nan, 
                 limit_qs=False, excl_depths=[], alpha_s="default", result="force"):
    """
    Function which calculates the pile base capacity based on:
        :cpt:            Input dataframe of CPT data

        :pile_type:     Pile type
                        Can be: "driven_precast"
        :z_top:      Level of pile top relative to NAP
        :z_base:      Depth of pile
        :pile_dia:           pile_diameter of pile shaft
        :excl_depths:   List of a start and end depth, between which no shaft friction
                        is deemed to occur
        
    NOTE: Loam is not included within this function.
    
    """    
    print("_________________________________________")
    print("Calculating shaft capacity using the NEN 9997-1 design method ...\n")
    
    segment_length = z_top-z_base
    
    if "Isbt" not in cpt:
        raise ValueError("The NEN9997-1 design method needs Isbt as an input.\nSee cpyt.correlations")
    
    ## Pile Geom
    if (np.isnan(h) == True) & (np.isnan(b) == True):       # i.e. if the pile is circular
        circum = 2*np.pi*(pile_dia/2)
        d_eq = pile_dia                                          # Equivalent pile_diameter = pile_diameter for circular pile
        surf_area = circum*(segment_length)                    # Surface area of entire pile shaft
    if np.isnan(pile_dia) == True:                               # i.e. if the pile is rectangular
        circum = 2*h + 2*b
        area = h*b
        d_eq = ((area/np.pi)**0.5)*2
        surf_area = 2*h*segment_length + 2*b*segment_length

    cpt = cpt.loc[(cpt.z >= z_base) & (cpt.z <= z_top)]

    cpt["alpha_s"] = np.nan
    alpha_s_T7c = {"driven_precast": 0.01,                      # As per Table 7c NEN9997
                    "screw_injection": 0.009, 
                    "vibro": 0.014}


    # Identify potential clay regions using the Isbt and Table 7d NEN9997-1 and fill in alpha_s factors
    cpt.alpha_s.loc[(cpt.qc > 2.5) & (cpt.Isbt > 2.95) & (cpt.Isbt < 3.6)] = 0.03              # Note that 0.03 is the upper limit
    cpt.alpha_s.loc[(cpt.qc < 2.5) & (cpt.qc > 2.0) & (cpt.Isbt > 2.95) & (cpt.Isbt < 3.6)]= 0.02*(cpt.qc-1)
    cpt.alpha_s.loc[(cpt.qc < 2) & (cpt.Isbt > 2.95) & (cpt.Isbt < 3.6)] = 0.02               # Note that 0.02 is the upper limit
    cpt.alpha_s.loc[(cpt.Isbt > 3.6)] = 0.0
    
    # print("NOTE: constant alpha_s for clay layers has been applied")
    # cpt.alpha_s.loc[cpt.qc < 2] = 0.025

    # Fill the remaining alpha_s with the alpha_s for sand
    if alpha_s == "default":
        cpt.alpha_s.fillna(alpha_s_T7c[pile_type], inplace=True)     # Fill in alpha_s values for sand and gravel
    else:  # (else if alpha_s is provided)
        cpt.alpha_s.fillna(alpha_s, inplace=True)

    if limit_qs:
        print("WARNING: Only a simple constant limiting value has been imposed")
        print("See p166 for what it actually should be")
        cpt.qc.loc[cpt.qc > 15] = 15
        print("Limits according to NEN9997 have been imposed")
        # if (isinstance(incl_lim,float)) | (isinstance(incl_lim,int)):
        #     cpt.qc.loc[cpt.qc > incl_lim] = incl_lim
        #     print("A custom limit has been imposed")
        
    cpt["qs"]=cpt.alpha_s*cpt.qc*1000                                  # Pile shaft resistance [kN/m2]
    
    cpt["depth_diff"] = abs(cpt.z.diff())                     # Depth difference between CPT soundings
    cpt["Qsmax_contrib"] = cpt.depth_diff*circum*cpt.qs     # Contribution of each ~2cm layer to total shear resistance
    if len(excl_depths) == 2:
        cpt.Qsmax_contrib.loc[(cpt.z <= excl_depths[0]) & (cpt.z >= excl_depths[-1])] = 0
    cpt["Qsmax"] = cpt.Qsmax_contrib.cumsum()

    cpt.reset_index(drop=True,inplace=True)
    near_z_ix = cpt.z.sub(z_base).abs().idxmin()           # Index of row with depth closest to pile depth
    Qsmax = cpt.Qsmax.iloc[near_z_ix]                  # Shaft capacity in kN
    qsmax = (Qsmax/1000)/surf_area                       # Shaft capacity in MPa
    
    if result=="stress":
        print(f"The pile shaft capacity is {qsmax:.2f} kPa")
        print("_________________________________________")
        return qsmax
    elif result=="force":
        print(f"The pile shaft capacity is {Qsmax:.2f} kN")
        print("_________________________________________")
        return Qsmax
    
    
def doan_lehane_2021(cpt, z_top, z_base,pile_dia, 
                     result="force",tension_load = False):
    """
    For bored piles in clay (including CFA/ACIP)

    """    
    print("_________________________________________")
    print("Calculating shaft capacity using the Doan & Lehane (2022) design method ...\n")
    
    segment_length = z_top-z_base
    circum = 2*np.pi*(pile_dia/2)
    d_eq = pile_dia                         # Equivalent pile_diameter = pile_diameter for circular pile
    surf_area = circum*(segment_length)     # Surface area of segment

    if "Ic" not in cpt.columns:
        raise ValueError("The Doan & Lehane (2022) method needs the normalised SBT index Ic as an input. See cpyt.correlations")
    if "qt" not in cpt.columns:
        raise ValueError("Doan & Lehane (2022) needs qt as an input. See cpyt.correlations")
    
    cpt = cpt.loc[(cpt.z >= z_base) & (cpt.z <= z_top)]
    
    pa = 101        # [kPa]
    qt = cpt.qt*1000 # [kPa]
    
    if tension_load:
        ft_fc = 0.8
    else:
        ft_fc = 1
        
    cpt["qs"] = ft_fc*0.008*pa*(cpt.Ic**1.6)*((qt/pa)**0.8) # Shaft resistance [kPa]

    cpt["depth_diff"] = abs(cpt.z.diff())                   # Depth difference between CPT soundings
    cpt["Qsmax_contrib"] = cpt.depth_diff*circum*cpt.qs     # Contribution of each ~2cm layer to total shear resistance
    cpt["Qsmax"] = cpt.Qsmax_contrib.cumsum()

    cpt.reset_index(drop=True,inplace=True)
    near_z_ix = cpt.z.sub(z_base).abs().idxmin()        # Index of row with depth closest to pile depth
    Qsmax = cpt.Qsmax.iloc[near_z_ix]                   # Shaft capacity in kN
    qsmax = (Qsmax/1000)/surf_area                      # Shaft capacity in MPa
    
    if result=="stress":
        print(f"The pile shaft capacity is {qsmax:.2f} kPa")
        print("_________________________________________")
        return qsmax
    elif result=="force":
        print(f"The pile shaft capacity is {Qsmax:.2f} kN")
        print("_________________________________________")
        return Qsmax


def uwa05(cpt, pile_type, z_top, z_base,
            h=np.nan, b=np.nan, pile_dia=np.nan, 
            alpha_s="default", result="force"):
    """
    Function which calculates the pile base capacity based on:
        :cpt:            Input dataframe of CPT data
        :pile_type:     Pile type
                        Can be: "driven_precast"
        :z_top:      Level of pile top relative to NAP
        :z_base:      Depth of pile
        :pile_dia:           pile_diameter of pile shaft
    """    
    print("_________________________________________")
    print("Calculating shaft capacity using the UWA design method ...\n")
    
    segment_length = z_top-z_base
    
    ## Pile Geom
    if (np.isnan(h) == True) & (np.isnan(b) == True):       # i.e. if the pile is circular
        circum = 2*np.pi*(pile_dia/2)
        d_eq = pile_dia                                          # Equivalent pile_diameter = pile_diameter for circular pile
        surf_area = circum*(segment_length)                    # Surface area of entire pile shaft
    if np.isnan(pile_dia) == True:                               # i.e. if the pile is rectangular
        circum = 2*h + 2*b
        area = h*b
        d_eq = ((area/np.pi)**0.5)*2
        surf_area = 2*h*segment_length + 2*b*segment_length
      
    #%%  
    """NB: IFR is not in this - assumed closed-ended pile
    THIS IS ALSO INCORRECT - take average h/D across the depth with which it applies
    """
    raise ValueError("ERROR: THE UWA-05 METHOD IS NOT WORKING HERE. CHECK FUNCTION DIRECTLY")
    if "sig_eff" not in cpt.columns:
        raise ValueError("The UWA design method needs sig_eff as an input.\nSee cpyt.correlations")

    cpt = cpt.loc[(cpt.z >= z_base) & (cpt.z <= z_top)]
    
    IFR = 0
    A_rs = 1-IFR*(1)
    delta_r = 2E-5                      # Dilation parameter = 0.02mm (lehane et al., 2007)
    pa = 100                            # Reference stress = 100kPa
    f_div_fc = 1                        # = 1 for compression (Lehane et al., 2007)
    delta_cv = 28.8                     # As per UWA-05 recommendation upper limit (Lehane et al., 2007)
    
    cpt["temp"] = cpt.z/pile_dia          # For use in sig_rc calculation
    cpt.temp[cpt.temp < 2] = 2
    cpt["sig_rc"] = 0.03*cpt.qc*1000*((A_rs)**0.3)*(cpt.temp)**-0.5
    cpt["qc1n"] = (cpt.qc*1000/pa)/(cpt.sig_eff/pa)**0.5
    cpt["G"] = cpt.qc*1000*185*cpt.qc1n**-0.75
    cpt["delta_sig_rd"] = 4*cpt.G*(delta_r/pile_dia)
    cpt["qs"] = f_div_fc*(cpt.sig_rc + cpt.delta_sig_rd)*np.tan(np.deg2rad(delta_cv))   # Pile shaft resistance [kN/m2]   
    
    # Return result
    cpt["depth_diff"] = abs(cpt.z.diff())                     # Depth difference between CPT soundings
    cpt["Qsmax_contrib"] = cpt.depth_diff*circum*cpt.qs     # Contribution of each ~2cm layer to total shear resistance
    cpt["Qsmax"] = cpt.Qsmax_contrib.cumsum()

    cpt.reset_index(drop=True,inplace=True)
    near_z_ix = cpt.z.sub(z_base).abs().idxmin()           # Index of row with depth closest to pile depth
    Qsmax = cpt.Qsmax.iloc[near_z_ix]                  # Shaft capacity in kN
    qsmax = (Qsmax/1000)/surf_area                       # Shaft capacity in MPa
    
    if result=="stress":
        print(f"The pile shaft capacity is {qsmax:.2f} kPa")
        print("_________________________________________")
        return qsmax
    elif result=="force":
        print(f"The pile shaft capacity is {Qsmax:.2f} kN")
        print("_________________________________________")
        return Qsmax
    
#%%
def unified(cpt, z_top, z_base, closed_ended=True, 
            material="concrete",
            h=None, b=None, d_cpt=35.7E-3,
            D_inner = None, D_outer=None, 
            tension_load = False,
            result="force"):
    """
    Function which calculates the pile base capacity based on:
        :cpt:           Input dataframe of CPT data
        :z_top:         Elevation of pile head relative to CPT
        :z_base:        Elevation of pile base relative to CPT
    """    
    print("_________________________________________")
    print("Calculating shaft capacity using the Unified design method ...\n")
    
    if "sig_eff" not in cpt.columns:
        raise ValueError("The Unified design method needs sig_eff as an input.\nSee cpyt.correlations")

    segment_length = z_top-z_base
    
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
     
    # Set up input parameters
    if tension_load:
        ft_fc = 0.75         # An appropriate range is 0.7 to 0.8 (Lehane et al., 2022) 
    else:
        ft_fc = 1
     
    h_D = (cpt.z - z_base)/D_eq
    delta_f_dict = {"concrete": 29,"steel": 29}
    delta_f = delta_f_dict[material]    # degrees
    
    if closed_ended == True:
        A_re = 1
    else:
        PLR = np.tanh(0.3*(D_inner/d_cpt)**0.5)
        A_re = 1 - PLR*(D_inner/D_eq)**2
    
    # Main calculations
    h_D.loc[h_D < 1] = 1    # i.e. max(1, h/D)
    sig_eff_rc = ((cpt.qc*1000)/44)*(A_re**0.3)*(h_D)**-0.4
    sig_eff_rd = ((cpt.qc*1000)/10)*((cpt.qc/(cpt.sig_eff/1000))**-0.33)*(d_cpt/D_eq)
    cpt["qs"] = ft_fc*(sig_eff_rc + sig_eff_rd)*np.tan(np.radians(delta_f))
    
    # Return result
    cpt["depth_diff"] = abs(cpt.z.diff())                   # Depth difference between CPT soundings
    cpt["Qsmax_contrib"] = cpt.depth_diff*circum*cpt.qs     # Contribution of each ~2cm layer to total shear resistance
    cpt["Qsmax"] = cpt.Qsmax_contrib.cumsum()

    cpt.reset_index(drop=True,inplace=True)
    near_z_ix = cpt.z.sub(z_base).abs().idxmin()        # Index of row with depth closest to pile depth
    Qsmax = cpt.Qsmax.iloc[near_z_ix]                   # Shaft capacity in kN
    qsmax = (Qsmax/1000)/surf_area                      # Shaft capacity in MPa
    
    if result=="stress":
        print(f"The pile shaft capacity is {qsmax:.2f} kPa")
        print("_________________________________________")
        return qsmax
    elif result=="force":
        print(f"The pile shaft capacity is {Qsmax:.2f} kN")
        print("_________________________________________")
        return Qsmax


def afnor_2012(cpt, pile_type, z_top, z_base,
                h=np.nan, b=np.nan, pile_dia=np.nan, limit_qs=False,
                result="force"):
    """
    French method. See Frank, 2017

    """    
    print("_________________________________________")
    print("Calculating shaft capacity using the AFNOR design method ...\n")
    segment_length = z_top-z_base
    
    if "soil_type" not in cpt.columns:
        cpt = classify_cpt.best_possible_method(cpt)
    
    ## Pile Geom
    if (np.isnan(h) == True) & (np.isnan(b) == True):   # i.e. if the pile is circular
        circum = 2*np.pi*(pile_dia/2)
        d_eq = pile_dia                                 # Equivalent pile_diameter = pile_diameter for circular pile
        surf_area = circum*(segment_length)             # Surface area of segment
    if np.isnan(pile_dia) == True:                      # i.e. if the pile is rectangular
        circum = 2*h + 2*b
        area = h*b
        d_eq = ((area/np.pi)**0.5)*2
        surf_area = 2*h*segment_length + 2*b*segment_length
      
    cpt = cpt.loc[(cpt.z >= z_base) & (cpt.z <= z_top)]
    
    cpt["alpha"] = np.nan         # Factor relating to installation method [-]
    cpt["f_sol"] = np.nan         # f_sol is a factor relating to the soil [kPa]
    
    # NB: The "units" of the alpha factor in AFNOR are different (kPa/MPa)
    alpha_dict_sand = {"driven_closed_ended": 1,
                       "screw_cast_in_place": 1.45,
                       "screw_pile_with_casing": 0.40
                       }
    alpha_dict_inter = {"driven_closed_ended": 0.65,     # Intermediate
                       "screw_cast_in_place": 1.15,
                       "screw_pile_with_casing": 0.35
                       }
    alpha_dict_clay = {"driven_closed_ended": 0.55,
                       "screw_cast_in_place": 0.95,
                       "screw_pile_with_casing": 0.30
                       }
    
    cpt.alpha.loc[cpt.soil_type == "sand"] = alpha_dict_sand[pile_type]
    cpt.alpha.loc[cpt.soil_type == "silt"] = alpha_dict_inter[pile_type]
    cpt.alpha.loc[cpt.soil_type == "clay"] = alpha_dict_clay[pile_type]
    
    qc = cpt.qc     # For ease of reading
    
    # A quick fit polynomial was used to get these functions. Better hyperbolic functions could be used
    cpt.loc[cpt.soil_type == "sand","f_sol"] = 0.0132*qc**3 - 0.6771*qc**2 + 14.16*qc
    cpt.loc[cpt.soil_type == "silt","f_sol"] = -0.0017*qc**4 + 0.0954*qc**3 - 2.0864*qc**2 + 23.618*qc
    cpt.loc[cpt.soil_type == "clay","f_sol"] = -0.0049*qc**4 + 0.2389*qc**3 - 4.2588*qc**2 + 35.049*qc
    
    # Functions for f_sol only apply for qc<20 MPa so limit at values beyond this
    cpt.loc[(cpt.soil_type == "sand") & (cpt.qc > 20),"f_sol"] = 118
    cpt.loc[(cpt.soil_type == "silt") & (cpt.qc > 20),"f_sol"] = 129
    cpt.loc[(cpt.soil_type == "clay") & (cpt.qc > 20),"f_sol"] = 125
    
    cpt["qs"]=cpt.alpha*cpt.f_sol    # Pile shaft resistance [kPa]
    
    # Limit
    if limit_qs:
        limit_dict_sand = {"driven_closed_ended": 130,
                           "screw_cast_in_place": 130,
                           "screw_pile_with_casing": 90
                           }
        limit_dict_clay = {"driven_closed_ended": 130,
                           "screw_cast_in_place": 130,
                           "screw_pile_with_casing": 50
                           }
        limit_dict_inter = {"driven_closed_ended": 130,     # Intermediate
                           "screw_cast_in_place": 130,
                           "screw_pile_with_casing": 50
                           }
        
        cpt.loc[(cpt.soil_type == "sand") & (cpt.qs > limit_dict_sand[pile_type])].qs = limit_dict_sand[pile_type]
        cpt.loc[(cpt.soil_type == "silt") & (cpt.qs > limit_dict_inter[pile_type])].qs = limit_dict_inter[pile_type]
        cpt.loc[(cpt.soil_type == "clay") & (cpt.qs > limit_dict_clay[pile_type])].qs = limit_dict_clay[pile_type]
        print("Limits according to AFNOR 2012 have been imposed")
    
    cpt["depth_diff"] = abs(cpt.z.diff())                     # Depth difference between CPT soundings
    cpt["Qsmax_contrib"] = cpt.depth_diff*circum*cpt.qs     # Contribution of each ~2cm layer to total shear resistance
    cpt["Qsmax"] = cpt.Qsmax_contrib.cumsum()

    cpt.reset_index(drop=True,inplace=True)
    near_z_ix = cpt.z.sub(z_base).abs().idxmin()           # Index of row with depth closest to pile depth
    Qsmax = cpt.Qsmax.iloc[near_z_ix]                  # Shaft capacity in kN
    qsmax = (Qsmax/1000)/surf_area                       # Shaft capacity in MPa
    
    if result=="stress":
        print(f"The pile shaft capacity is {qsmax:.2f} kPa")
        print("_________________________________________")
        return qsmax
    elif result=="force":
        print(f"The pile shaft capacity is {Qsmax:.2f} kN")
        print("_________________________________________")
        return Qsmax
    
    
def nesmith_2002(cpt, z_top, z_base, pile_dia, ws=0,
                 limit_qc=True,result="force"):
    """
    Method by NeSmith for screw displacement piles in sandy soils.
    
    Note that failure in this method is a settlement based criterion where the
    pile head displacement is equal to 25.4mm (=1 inch)
    
    :ws: is a constant which depends on soil gradation and angularity
    
    For soils containing uniform, rounded particles with up to 40 % fines, 
    ws = 0 and the limiting value of qsi is 0.16 MPa. For soils with well-graded, 
    angular particles having less than 10 % fines, ws = 0.05 MPa and the 
    limiting value of qsi is 0.21 MPa.
    
    Since :ws: is given on a per layer basis, it is recommended to sum up the 
    number of soil layers and input ws manually. So if we have three well-graded
    layers, set :ws: to 3*0.05 MPa.
    
    """  
    print("_________________________________________")
    print("Calculating shaft capacity using the NeSmith (2002) design method ...\n")

    segment_length = z_top-z_base
    
    ## Pile Geom
    circum = 2*np.pi*(pile_dia/2)
    d_eq = pile_dia                         # Equivalent pile_diameter = pile_diameter for circular pile
    surf_area = circum*(segment_length)     # Surface area of segment

    if limit_qc:
        cpt.qc.loc[cpt.qc > 19] = 19
        print("Limits to qc have been imposed")

    cpt = cpt.loc[(cpt.z >= z_base) & (cpt.z <= z_top)]
    
    cpt["alpha_s"] = 0.01       # Just one alpha_s is applied to all soil layers
    cpt["qs"]=cpt.alpha_s*cpt.qc*1000  + ws                                # Pile shaft resistance [kPa]
    
    cpt["depth_diff"] = abs(cpt.z.diff())                     # Depth difference between CPT soundings
    cpt["Qsmax_contrib"] = cpt.depth_diff*circum*cpt.qs     # Contribution of each ~2cm layer to total shear resistance
    cpt["Qsmax"] = cpt.Qsmax_contrib.cumsum()

    cpt.reset_index(drop=True,inplace=True)
    near_z_ix = cpt.z.sub(z_base).abs().idxmin()           # Index of row with depth closest to pile depth
    Qsmax = cpt.Qsmax.iloc[near_z_ix]                  # Shaft capacity in kN
    qsmax = (Qsmax/1000)/surf_area                       # Shaft capacity in MPa
    
    
    if result=="stress":
        print(f"The pile shaft capacity is {qsmax:.2f} kPa")
        print("_________________________________________")
        return qsmax
    elif result=="force":
        print(f"The pile shaft capacity is {Qsmax:.2f} kN")
        print("_________________________________________")
        return Qsmax
    