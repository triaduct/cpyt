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
    qsmax = Qsmax/surf_area                         # Shaft capacity in kPa
    
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
    qsmax = Qsmax/surf_area                             # Shaft capacity in kPa
    
    if result=="stress":
        print(f"The pile shaft capacity is {qsmax:.2f} kPa")
        print("_________________________________________")
        return qsmax
    elif result=="force":
        print(f"The pile shaft capacity is {Qsmax:.2f} kN")
        print("_________________________________________")
        return Qsmax


def uwa05(cpt, pile_type, z_top_inc, z_base_inc, z_base,
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
    
    segment_length = z_top_inc-z_base_inc
    
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

    cpt = cpt.loc[(cpt.z >= z_base_inc) & (cpt.z <= z_top_inc)]
    
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
    near_z_ix = cpt.z.sub(z_base).abs().idxmin()        # Index of row with depth closest to pile depth
    Qsmax = cpt.Qsmax.iloc[near_z_ix]                   # Shaft capacity in kN
    qsmax = Qsmax/surf_area                             # Shaft capacity in kPa
    
    if result=="stress":
        print(f"The pile shaft capacity is {qsmax:.2f} kPa")
        print("_________________________________________")
        return qsmax
    elif result=="force":
        print(f"The pile shaft capacity is {Qsmax:.2f} kN")
        print("_________________________________________")
        return Qsmax
    
#%%
def unified_clay(cpt, z_top_inc, z_base_inc, z_base, closed_ended=True, 
            Fst = 1,
            D_inner = None, D_outer=None, 
            tension_load = False,
            result="force"):
    """
    (Lehane et al., 2022)
    :z_base:            Needed to calculate h/D
    :param Fst:             Describes the sensitivity of the soil. Fst =1 for Zone 2,3,4
                            clays and Fst = 0.5+-0.2 for Zone 1 Clays
                            TODO: Implement automatic check for Zone 1 clay
    :param tension_load:    Unified method for clay does not differentiate between tension
                            load and comparession (ft/fc =1). Parameter included for clarity
    
    Function which calculates the pile base capacity based on:
        :cpt:           Input dataframe of CPT data
        :z_top:         Elevation of pile head relative to CPT
        :z_base:        Elevation of pile base relative to CPT
        :d_cpt:         Diameter of the CPT cone [m]
    """    
    print("_________________________________________")
    print("Calculating shaft capacity using the Unified design method for sand...\n")
    
    if "qt" not in cpt.columns:
        raise ValueError("The Unified design method needs qt as an input.\nSee cpyt.correlations")

    segment_length = z_top_inc - z_base_inc
    cpt = cpt.loc[(cpt.z <= z_top_inc) & (cpt.z >= z_base)]
    
    circum = np.pi*D_outer
    surf_area = circum*segment_length
    
    if closed_ended:
        Dstar = D_outer
    else:
        Dstar = (D_outer**2 - D_inner**2)**0.5  
        
    cpt["h_D"] = (cpt.z - z_base)/Dstar
    cpt.h_D.loc[cpt.h_D < 1] = 1    # i.e. max(1, h/D)
    
    cpt["qs"] = 0.07 * Fst * (cpt.qt*1000) * cpt.h_D **-0.25    # kPa
    
    # Return result
    cpt["depth_diff"] = abs(cpt.z.diff())                   # Depth difference between CPT soundings
    cpt["Qsmax_contrib"] = cpt.depth_diff*circum*cpt.qs     # Contribution of each ~2cm layer to total shear resistance
    cpt["Qsmax"] = cpt.Qsmax_contrib.cumsum()

    cpt.reset_index(drop=True,inplace=True)
    near_z_ix = cpt.z.sub(z_base).abs().idxmin()        # Index of row with depth closest to pile depth
    Qsmax = cpt.Qsmax.iloc[near_z_ix]                   # Shaft capacity in kN
    qsmax = Qsmax/surf_area                             # Shaft capacity in kPa
    
    if result=="stress":
        print(f"The pile shaft capacity is {qsmax:.2f} kPa")
        print("_________________________________________")
        return qsmax
    elif result=="force":
        print(f"The pile shaft capacity is {Qsmax:.2f} kN")
        print("_________________________________________")
        return Qsmax


def unified_sand(cpt, z_top_inc, z_base_inc, z_base, closed_ended=True, 
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
        :d_cpt:         Diameter of the CPT cone [m]
    """    
    print("_________________________________________")
    print("Calculating shaft capacity using the Unified design method for clay ...\n")
    
    if "sig_eff" not in cpt.columns:
        raise ValueError("The Unified design method needs sig_eff as an input.\nSee cpyt.correlations")

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
    qsmax = Qsmax/surf_area                             # Shaft capacity in kPa
    
    if result=="stress":
        print(f"The pile shaft capacity is {qsmax:.2f} kPa")
        print("_________________________________________")
        return qsmax
    elif result=="force":
        print(f"The pile shaft capacity is {Qsmax:.2f} kN")
        print("_________________________________________")
        return Qsmax
    
#%%
def afnor_2012(cpt, pile_type, z_top, z_base,
                h=np.nan, b=np.nan, pile_dia=np.nan, limit_qs=True,
                result="force"):
    """
    French method. See Frank, 2017

    """    
    print("_________________________________________")
    print("Calculating shaft capacity using the AFNOR design method ...\n")
    segment_length = z_top-z_base
    
    if "soil_type" not in cpt.columns:
        cpt = classify.best_possible_method(cpt)
    
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
    qc[qc > 20] = 20    # f_sol function only applicable at qc<20 MPa
    
    a = 0.0012; b = 0.1; c = 0.15
    cpt.loc[cpt.soil_type == "sand","f_sol"] = (0.0012*qc + b)*(1-np.exp(-c*qc))
    a = 0.0018; b = 0.1; c = 0.4
    cpt.loc[cpt.soil_type == "clay","f_sol"] = (0.0012*qc + b)*(1-np.exp(-c*qc))
    a = 0.0015; b = 0.1; c = 0.25
    cpt.loc[cpt.soil_type == "intermediate","f_sol"] = (0.0012*qc + b)*(1-np.exp(-c*qc))        # For intermediate soils, chalk, weather rock, marl
    
    cpt["qs"] = cpt.alpha * (cpt.f_sol*1000)    # Pile shaft resistance [kPa]
    
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
    near_z_ix = cpt.z.sub(z_base).abs().idxmin()        # Index of row with depth closest to pile depth
    Qsmax = cpt.Qsmax.iloc[near_z_ix]                   # Shaft capacity in kN
    qsmax = Qsmax/surf_area                             # Shaft capacity in kPa
    
    if result=="stress":
        print(f"The pile shaft capacity is {qsmax:.2f} kPa")
        print("_________________________________________")
        return qsmax
    elif result=="force":
        print(f"The pile shaft capacity is {Qsmax:.2f} kN")
        print("_________________________________________")
        return Qsmax

    #%%
def bustamente_gianeselli_1998(cpt, pile_type, z_top, z_base,soil_type,
                               pile_dia,result="force"):
    """
    Bustamente & Gianeselli (1998) method developed primarily for screw displacement
    piles. Formed the basis for the current AFNOR method
    
    Need to specify soil_type and pile_type
    """    
    print("_________________________________________")
    print("Calculating shaft capacity using the Bustamente & Gianeselli (1998) design method...\n")
    segment_length = z_top-z_base
    

    circum = 2*np.pi*(pile_dia/2)
    d_eq = pile_dia                                 # Equivalent pile_diameter = pile_diameter for circular pile
    surf_area = circum*(segment_length)             # Surface area of segment

      
    cpt = cpt.loc[(cpt.z >= z_base) & (cpt.z <= z_top)]
    qc = cpt.qc.mean()
    
    # Curves are given as points and are dependent on pile and soil type
    curves = {"Q1": np.array([[0,0.4,0.7,1.2],[0,0.02,0.03,0.04]]),
              "Q2": np.array([[0.4,1,1.8,3],[0.03,0.06,0.08,0.09]]),
              "Q3": np.array([[0.2,1,2,3],[0.03,0.08,0.115,0.125]]),
              "Q4": np.array([[0.5,1,2,3],[0.07,0.1,0.14,0.16]]),
              "Q5": np.array([[1,1.5,3],[0.12,0.15,0.2]])}

    if ("clay" in soil_type) | ("silt" in soil_type):    # No prescriptions for silt in original document, but stateed in Bu&Gi, 1997
        if qc < 1.5:        # Was originally 1 but doesn't describe anything between 1.0 and 1.5
            Q = "Q1"
        elif (qc > 1.5) & (qc < 3):
            if pile_type == "screw_cast_in_place":
                Q = "Q3"
            if pile_type == "screw_pile_with_casing":
                Q = "Q2"
        elif (qc > 3):
            if pile_type == "screw_cast_in_place":
                Q = "Q4"
            if pile_type == "screw_pile_with_casing":
                Q = "Q2"
    elif "sand" in soil_type:
        if qc < 3.5:
            Q = "Q1"
        elif (qc > 3.5) & (qc < 8):
            if pile_type == "screw_cast_in_place":
                Q = "Q4"
            if pile_type == "screw_pile_with_casing":
                Q = "Q2"
        elif (qc > 8):
            if pile_type == "screw_cast_in_place":
                Q = "Q5"
            if pile_type == "screw_pile_with_casing":
                Q = "Q2"
    
    print(soil_type)
    print(Q)
    Q = curves[Q]
    
    # Curves originally given according to the pressuremeter. Scale the curves up 
    # to get CPT values and adjust per soil type.
    soil_factor_dict = {"clay": 3, "clayey-silt": 3, "sandy-clay": 3, 
                        "silt": 3,      # No guidance for pure silt soils
                        "sand": 8, "gravel": 8, "marl": 3.5}               
    soil_factor = soil_factor_dict[soil_type]
    Q[0] = soil_factor*Q[0]        # Multiply the x-axis values by this amount
    
    n_points = Q.shape[1]
    for i in range(n_points):           # For each point on the curve
        if i == n_points- 1:            # If last point, limiting value reached
            qsmax = Q[1][i]*1000        # Shaft resistance in [kPa]
            break
        
        if (qc > Q[0][i]) & (qc < Q[0][i+1]):
            x1, x2, y1, y2 = Q[0][i], Q[0][i+1], Q[1][i], Q[1][i+1],
            qsmax = np.interp(qc, [x1,x2], [y1,y2])*1000     # Shaft resistance in [kPa]
            break
        # else:
            # raise ValueError(f"A qc value of {qc:.1f} is not available on the curve")
    
    Qsmax = qsmax*surf_area                             # Shaft capacity in kN
    
    # Plot results to double-check
    # fig = plt.figure()
    # ax = fig.gca()
    # for curve in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
    #     ax.plot(curves[curve][0],curves[curve][1],alpha=0.5,c="black")
    # ax.plot(Q[0]/soil_factor,Q[1],c="k")
    # ax.scatter(qc/soil_factor,qsmax/1000)
    # ax.set_xlim(0,3)
    # plt.show()
    
    if result=="stress":
        print(f"The pile shaft capacity is {qsmax:.2f} kPa")
        print("_________________________________________")
        return qsmax
    elif result=="force":
        print(f"The pile shaft capacity is {Qsmax:.2f} kN")
        print("_________________________________________")
        return Qsmax

#%%
def nbn_2022(cpt, pile_type, z_top, z_base,soil_type,
                h=np.nan, b=np.nan, pile_dia=np.nan,limit_qs = True,
                result="force"):
    """
    Belgian method. See Huybrechts et al. 2016. alpha factors have since been updated

    Performed layer by layer
    """    
    print("_________________________________________")
    print("Calculating shaft capacity using the Belgian NBN design method ...\n")
    segment_length = z_top-z_base
    
    if "soil_type" not in cpt.columns:
        cpt = classify.best_possible_method(cpt)
    
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
    qc = cpt.qc.mean()
    
    # Apply alpha factors (relates to installation method)
    alpha_dict_other = {"driven_closed_ended": 1.0,             # "other" means soil types other than tclay
                       "screw_with_temporary_tube": 0.6,         # With temporary tube and shaft in plastic concrete
                       "screw_with_lost_tube": 0.6,
                       "screw_injection": 0.6
                       }
    alpha_dict_clay = {"driven_closed_ended": 0.9,             # 
                           "screw_with_temporary_tube": 0.6,         # With temporary tube and shaft in plastic concrete
                           "screw_with_lost_tube": 0.6,
                           "screw_injection": 0.6
                           }
    
    if soil_type in ["sand","silt"]:
        alpha_s = alpha_dict_other[pile_type]
    elif soil_type == "clay":
        alpha_s = alpha_dict_clay[pile_type]      # Have stated that tertiary clay == clay

    # Apply eta factors    
    if soil_type == "clay":
        if qc < 1: 
            eta_p = 1/30            # NOTE: no value specified in Huybrechts et al. 2016. Just set as 1/30
        elif (qc >= 1) & (qc < 4.5):
            eta_p = 1/30
        elif (qc > 4.5):            # NOTE: no value specified in Huybrechts et al. 2016. Just set as 1/30      # 11/03: work around for calcs.py
            eta_p = 1/30
    elif soil_type == "silt":
        if qc < 1: 
            eta_p = 1/60            # NOTE: no value specified in Huybrechts et al. 2016. Just set as 1/30
        if (qc >= 1) & (qc < 6):
            eta_p = 1/60      
    elif soil_type == "sandy_clay":
        if qc < 1: 
            eta_p = 1/80            # NOTE: no value specified in Huybrechts et al. 2016. Just set as 1/30
        if (qc >= 1) & (qc < 10):
            eta_p = 1/80      
    elif soil_type == "sand":
        if (qc <= 1):
            eta_p = 1/90         # NOTE: no value specified in Huybrechts et al. 2016. Just set as 1/90
        if (qc >= 1) & (qc < 10):
            eta_p = 1/90  
    else:
        eta_p = 1/100
        print("NO ETA_P HAS BEEN PRESCRIBED")
    
    # Calculate unit shaft friction
    if (soil_type == "clay") & (qc > 4.5):
        qs = 150
    elif (soil_type == "silt") & (qc > 6.0):
        qs = 100
    elif (soil_type == "clayey_sand") & (qc > 10):
        qs = 125
    elif (soil_type == "sand") & (qc >= 10.0) & (qc < 20.0):
        qs = 110 + 4*(qc - 10)      # NB: qc here is in MPa -- empirical correlation
    elif (soil_type == "sand") & (qc > 20):
        qs = 150
    else: 
        qs = 1000*eta_p*qc
    
    qsmax = qs*alpha_s # [kPa]
    Qsmax = segment_length*circum*qs # [kN]

    
    if result=="stress":
        print(f"The pile shaft capacity is {qsmax:.2f} kPa")
        print("_________________________________________")
        return qsmax
    elif result=="force":
        print(f"The pile shaft capacity is {Qsmax:.2f} kN")
        print("_________________________________________")
        return Qsmax    
   

#%%
def nesmith_2002(cpt, z_top, z_base, pile_dia, ws=0,
                 limit_qc=True,limit_qs=True,
                 result="force"):
    """
    Method by NeSmith for screw displacement piles in SANDY SOILS ONLY.
    
    NOTE: Method must be applied layer-by-layer. Otherwise the contribution of 
    ws will not be accounted for properly.
    
    
    Note that failure in this method is a settlement based criterion where the
    pile head displacement is equal to 25.4mm (=1 inch)
    
    :ws: is a constant which depends on soil gradation and angularity [kPa]
    
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

    cpt = cpt.loc[(cpt.z >= z_base) & (cpt.z <= z_top)]

    if limit_qc:
        cpt.qc.loc[cpt.qc > 19] = 19
        print("Limits to qc have been imposed")

    cpt["alpha_s"] = 0.01       # Just one alpha_s is applied to all soil layers
    cpt["qs"]=cpt.alpha_s*cpt.qc*1000  + ws                                # Pile shaft resistance [kPa]
        
    cpt["depth_diff"] = abs(cpt.z.diff())                     # Depth difference between CPT soundings
    cpt["Qsmax_contrib"] = cpt.depth_diff*circum*cpt.qs     # Contribution of each ~2cm layer to total shear resistance
    cpt["Qsmax"] = cpt.Qsmax_contrib.cumsum()

    cpt.reset_index(drop=True,inplace=True)
    near_z_ix = cpt.z.sub(z_base).abs().idxmin()           # Index of row with depth closest to pile depth
    Qsmax = cpt.Qsmax.iloc[near_z_ix]             # Shaft capacity in kN
    qsmax = Qsmax/surf_area                       # Shaft capacity in kPa
    
    if limit_qs == True:
        # qb_lim needs to be interpolated between 160 and 210 kPa
        ws_perc = ws/50
        qslim = 160 + ws_perc*(210 - 160) 
        qsmax = min(qslim,qsmax)
        Qsmax = qsmax * surf_area  
    
    
    if result=="stress":
        print(f"The pile shaft capacity is {qsmax:.2f} kPa")
        print("_________________________________________")
        return qsmax
    elif result=="force":
        print(f"The pile shaft capacity is {Qsmax:.2f} kN")
        print("_________________________________________")
        return Qsmax
    