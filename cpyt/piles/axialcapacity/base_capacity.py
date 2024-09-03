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
def alpha_method(cpt,z_base,alpha_p, avg_method,
                 pile_h=np.nan, pile_b=np.nan, pile_dia=np.nan, limit_qb=False,
                 result="stress"):
    """
    Specify equivalent diameter area if square pile
    
    Generic method
    """
    print("_________________________________________")
    print("Calculating base capacity using a custom method ...\n")
    
    if (np.isnan(pile_h) == True) & (np.isnan(pile_b) == True):               # i.e. if the pile is circular
        area = np.pi*(pile_dia/2)**2
        pile_dia = pile_dia        
        
    if np.isnan(pile_dia) == True:                                       # i.e. if the pile is rectangular
        area = pile_h*pile_b
        pile_dia = ((area/np.pi)**0.5)*2        # Equivalent diameter
        
    if avg_method == "koppejan":
        qc_avg = averaging_methods.koppejan(cpt, z_base, pile_dia, plot_min_path = False)
    if avg_method == "de_boorder":
        qc_avg = averaging_methods.de_boorder(cpt, pile_dia, z_base)
    if avg_method == "de_beer":
        qc_avg = averaging_methods.de_beer(cpt, pile_dia, z_base)
        
    qbmax = qc_avg*alpha_p    # Maximum pile tip resistance [MPa]
    
    # Impose limits
    if limit_qb != False:
        if limit_qb == True:
            qb_lim = 15         # Original NEN9997 qblim
            print("Limits according to NEN9997 have been imposed")
        elif (isinstance(limit_qb,int)) | (isinstance(limit_qb,float)):
            print("A custom base limiting resistance has been applied")
            qb_lim = limit_qb
        
        qbmax = min(qb_lim,qbmax)
            
    Qbmax = (qbmax*1000*area)   # Max pile tip capacity [kN]

    if result=="stress":
        print(f"The pile base capacity at {z_base:.2f} m is {qbmax:.2f} MPa")
        print("_________________________________________")
        return qbmax
    elif result=="force":
        print(f"The pile base capacity at {z_base:.2f} m is {Qbmax:.2f} kN")
        print("_________________________________________")
        return Qbmax


def nen9997_2017(cpt,z_base,pile_type,pile_h=np.nan, pile_b=np.nan, pile_dia=np.nan, 
                 limit_qb=True,alpha_p=None,
                 beta=1,s=1,result="stress"):
    """
    Specify equivalent diameter area if square pile
    """
    print("_________________________________________")
    print("Calculating base capacity using the NEN 9997-1 design method ...\n")
    if (np.isnan(pile_h) == True) & (np.isnan(pile_b) == True):               # i.e. if the pile is circular
        area = np.pi*(pile_dia/2)**2
        pile_dia = pile_dia        
        
    if np.isnan(pile_dia) == True:                                       # i.e. if the pile is rectangular
        area = pile_h*pile_b
        pile_dia = ((area/np.pi)**0.5)*2        # Equivalent diameter
    
    if alpha_p == None:
        # Use default alpha_p from Table 7c
        Table7c = {"driven_precast": 0.7, "screw_injection": 0.63, "vibro": 0.7}
        alpha_p = Table7c[pile_type]
        
    qc_avg = averaging_methods.koppejan(cpt, pile_dia, z_base, plot_min_path = False)
    qbmax = qc_avg*alpha_p*beta*s    # Maximum pile tip resistance [MPa]
    
    # Impose limits
    if limit_qb != False:
        if limit_qb == True:
            qb_lim = 15         # Original NEN9997 qblim
            print("Limits according to NEN9997 have been imposed")
        elif (isinstance(limit_qb,int)) | (isinstance(limit_qb,float)):
            print("A custom base limiting resistance has been applied")
            qb_lim = limit_qb
        
        qbmax = min(qb_lim,qbmax)
            
    Qbmax = qbmax*1000*area   # Max pile tip capacity [kN]


    if result=="stress":
        print(f"The pile base capacity at {z_base:.2f} m is {qbmax:.2f} MPa")
        print("_________________________________________")
        return qbmax
    elif result=="force":
        print(f"The pile base capacity at {z_base:.2f} m is {Qbmax:.2f} kN")
        print("_________________________________________")
        return Qbmax


def nbn_2022(cpt,z_base,pile_type,pile_h=np.nan, pile_b=np.nan, pile_dia=np.nan,
             alpha_b=None, beta=None,lmbda=1,result="stress"):
    """
    Specify equivalent diameter area if square pile
    
    FOR SAND ONLY
     
    NOTE: Not include in this is :lmbda: which is a reduction factor that accounts
    for an enlarged pile base which may generate soil relaxation during installation.
    It is determined as follows:
        - for piles with an enlarged base that has been formed at depth, not causing soil relaxation
        around the pile shaft during installation: λ = 1.00
        - for piles with a prefabricated enlarged base, with Db,eq < Ds + 0.05 m: λ = 1.00
        - for all other piles with a prefabricated base:
            - Calculate:    x = Db**2/Ds**2
            - Screw pile:
                if x < 1.5: lmbda = 1
                if x > 1.7: lmbda = 0.7
            - Driven pile
                if x < 1.25: lmbda = 1
                if x > 1.7: lmbda = 0.7
    """
    
    print("_________________________________________")
    print("Calculating base capacity using the Belgian NBN 2014 design method ...\n")
    if (np.isnan(pile_h) == True) & (np.isnan(pile_b) == True):               # i.e. if the pile is circular
        area = np.pi*(pile_dia/2)**2
        pile_dia = pile_dia        
        pile_shape = "circle"
        
    if np.isnan(pile_dia) == True:                                       # i.e. if the pile is rectangular
        area = pile_h*pile_b
        pile_dia = ((area/np.pi)**0.5)*2        # Equivalent diameter
        if pile_h == pile_b:
            pile_shape = "square"
        else:
            pile_shape = "rectangle"
    
    # alpha_factor (sand only)
    if alpha_b == None:
        alpha_b_dict = {"driven_closed_ended": 1.0,             # "other" means soil types other than tertiary clay"
                        "screw_with_temporary_tube": 0.5,         # With temporary tube and shaft in plastic concrete
                        "screw_with_lost_tube": 0.5,
                        "screw_injection": 0.5
                        }
        
        alpha_b = alpha_b_dict[pile_type]
        
    # beta factor (shape factor)
    if beta == None:
        if pile_shape == "circle":
            beta = 1
        elif pile_shape == "square":
            beta = 1
        elif pile_shape == "rectangle":
            beta = (1 + 0.3*(pile_h/pile_b))/1.3
        
    qc_avg = averaging_methods.de_beer(cpt, pile_dia, target_depth = z_base)
    qbmax = qc_avg*alpha_b*beta*lmbda    # Maximum pile tip resistance [MPa]
    Qbmax = qbmax*1000*area   # Max pile tip capacity [kN]


    if result=="stress":
        print(f"The pile base capacity at {z_base:.2f} m is {qbmax:.2f} MPa")
        print("_________________________________________")
        return qbmax
    elif result=="force":
        print(f"The pile base capacity at {z_base:.2f} m is {Qbmax:.2f} kN")
        print("_________________________________________")
        return Qbmax
    
    
def afnor_2012(cpt,z_base,pile_type,soil_type,top_bearing_layer,
               pile_h=np.nan, pile_b=np.nan, pile_dia=np.nan, 
               bearing_factor=True,
               alpha_p=None,result="stress"):
    """
    Specify equivalent diameter area if square pile
    
    :top_bearing_layer: is needed to use the lpc_2012 averaging method. More details
    in (Verheyde & Baguelin, 2019)
    :bearing_factor:    Takes into account short embedment depth
    """
    print("_________________________________________")
    print("Calculating base capacity using the AFNOR design method ...\n")
    if (np.isnan(pile_h) == True) & (np.isnan(pile_b) == True):               # i.e. if the pile is circular
        area = np.pi*(pile_dia/2)**2
        pile_dia = pile_dia        
        
    if np.isnan(pile_dia) == True:                                       # i.e. if the pile is rectangular
        area = pile_h*pile_b
        pile_dia = ((area/np.pi)**0.5)*2        # Equivalent diameter
        print("TOFIX: AFNOR does not use equivalent diameter when considering the embedmeent length (see :bearing_factor:")
    
    
    kc_dict_sand = {"driven_closed_ended": 0.40,
                       "screw_cast_in_place": 0.50,
                       "screw_pile_with_casing": 0.50
                       }
    kc_dict_inter = {"driven_closed_ended": 0.40,
                       "screw_cast_in_place": 0.50,
                       "screw_pile_with_casing": 0.50
                       }
    kc_dict_clay = {"driven_closed_ended": 0.45,
                       "screw_cast_in_place": 0.50,
                       "screw_pile_with_casing": 0.50
                       }
    
    if soil_type == "sand":
        kcmax = kc_dict_sand[pile_type]
    if soil_type == "silt":
        kcmax = kc_dict_inter[pile_type]
    if soil_type == "clay":
        kcmax = kc_dict_clay[pile_type]
    
    qc_avg = averaging_methods.lpc_2012(cpt, pile_dia,z_base,top_bearing_layer)         # Referred to as q_ce in AFNOR
    
    if bearing_factor:
        qc_avg_upper = cpt[(cpt.z <= z_base + 10*pile_dia) & (cpt.z >= z_base)].qc.mean()
        if qc_avg_upper >= 0.5*qc_avg:
            kc = kcmax
        else:
            # Calculate reduced bearing factor
            z_base_less_10D = z_base - 10*pile_dia
            qc_10D_below = cpt[(cpt.z <= z_base) & (cpt.z >= z_base_less_10D)].qc.mean()
            D_ef = (1/qc_avg)*qc_10D_below
            if D_ef/pile_dia > 5:
                kc = kcmax
            else:
                if soil_type == "sand":
                    kc = 0.3 + (kcmax - 0.3)*(D_ef/pile_dia)/5
                elif soil_type == "silt":
                    kc = 0.3 + (kcmax - 0.2)*(D_ef/pile_dia)/5
                elif soil_type == "clay":
                    kc = 0.3 + (kcmax - 0.1)*(D_ef/pile_dia)/5
    else:   
        kc = kcmax
    qbmax = qc_avg*kc    # Maximum pile tip resistance [MPa]
            
    Qbmax = (qbmax*1000*area)   # Max pile tip capacity [kN]


    if result=="stress":
        print(f"The pile base capacity at {z_base:.2f} m is {qbmax:.2f} MPa")
        print("_________________________________________")
        return qbmax
    elif result=="force":
        print(f"The pile base capacity at {z_base:.2f} m is {Qbmax:.2f} kN")
        print("_________________________________________")
        return Qbmax
    
    
def nesmith_2002(cpt, z_base, pile_dia, wb=0, limit_qc_avg=True,
                 limit_qb = True, result="stress"):
    """
    Method be NeSmith for screw displacement piles in sandy soils.
    
    Note that failure in this method is a settlement based criterion where the
    pile head displacement is equal to 25.4mm (=1 inch)
    
    :wb: is a constant which depends on soil gradation and angularity
    
    For soils containing uniform, rounded particles with up to 40% fines, 
    wb = 0 and the upper limit of qb is 7.2 MPa. For soils with well-graded, 
    angular particles having less than 10% fines, wb = 1.34 MPa  and the upper 
    limit for qb is 8.62 MPa. Interpolation (based on percentage of fines) is 
    suggested to determine the values of wb for other types of soils (NeSmith 2002).
    """
    print("___________________________________")
    print("Calculating base capacity using the NeSmith method ...\n")
    area = np.pi*(pile_dia/2)**2
    qc_avg = averaging_methods.fleming_and_thorburn(cpt, pile_dia, z_base)
    
    if (limit_qc_avg==True) & (qc_avg > 19):
        print("qc_avg has been limited to 19 MPa")
        qc_avg = 19
    
    alpha_p = 0.4
    qbmax = qc_avg*alpha_p + wb    # Maximum pile tip resistance [MPa]
    
    if limit_qb == True:
        # qb_lim needs to be interpolated between 7.2 and 8.62
        wb_perc = wb/1.34   # As a percentage of the maximum i.e. 1.34 MPa
        qblim = 7.2 + wb_perc*(8.62 - 7.2) 
        qbmax = min(qblim,qbmax)

    Qbmax = (qbmax*1000*area)   # Max pile tip capacity [kN]
    
    if result=="stress":
        print(f"The pile base capacity at {z_base:.2f} m is {qbmax:.2f} MPa")
        print("_________________________________________")
        return qbmax
    elif result=="force":
        print(f"The pile base capacity at {z_base:.2f} m is {Qbmax:.2f} kN")
        print("_________________________________________")
        return Qbmax

def unified(cpt,z_base, closed_ended=True, 
            alpha_p = None, avg_method = "lcpc",
            h=None, b=None, d_cpt=35.7E-3,
            D_inner = None, D_outer=None, 
            result="force"):
    """
    Specify equivalent diameter area if square pile
    
    Generic method
    """
    print("_________________________________________")
    print("Calculating base capacity using the Unified method for sand...\n")
    
    ## Pile Geometry
    if (h==None) & (b==None):       # i.e. if the pile is circular
        D = D_outer
        circum = 2*np.pi*(D_outer/2)
        area = np.pi*(D_outer/2)**2
    if D_outer == None:                               # i.e. if the pile is rectangular
        circum = 2*h + 2*b
        area = h*b
        D = ((area/np.pi)**0.5)*2
    
    if closed_ended == True:
        A_re = 1
    else:
        PLR = np.tanh(0.3*(D_inner/d_cpt)**0.5)
        A_re = 1 - PLR*(D_inner/D)**2
    
    D_eq = D*A_re**0.5
    
    if avg_method == "lcpc":
        qp = averaging_methods.lcpc(cpt, D_eq, z_base)      # qp = qc,avg
    elif avg_method == "koppejan":
        qp = averaging_methods.koppejan(cpt, D_eq, z_base, plot_min_path = False)
    elif avg_method == "de_boorder":
        qp = averaging_methods.de_boorder(cpt, D_eq, z_base)
    elif avg_method == "boulanger_dejong":
        raise ValueError("Check that Boulanger De Jong can be included")
    

    qb01 = (0.12 + 0.38*A_re)*qp    # Base stress [kPa]
    Qbase = (qb01*1000*area)       # Max pile tip capacity [kN]
    # TODO: Should Qbase be over the entirety of the pile tip?

    if result=="stress":
        print(f"The pile base capacity at {z_base:.2f} m is {qb01:.2f} MPa")
        print("_________________________________________")
        return qb01
    elif result=="force":
        print(f"The pile base capacity at {z_base:.2f} m is {Qbase:.2f} kN")
        print("_________________________________________")
        return Qbase


