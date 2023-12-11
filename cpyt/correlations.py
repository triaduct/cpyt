# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 08:32:15 2020

@author: kevin

This deals with the dataframe output of the CPT class.
"""
import pandas as pd, numpy as np
# from read_cpt import CPT

#%% 
def qt(df, u2=False, a=None):
    """
    a is the net area ratio and ranges from 0.70 to 0.85. Only used if pore 
    pressure was measured behind the cone tip (u2). Some typical values:
            Fugro 43.85mm 15cm2 cone: a = 0.58
    
    Returns qt in MPa
    """
    if "u2" in df.columns:
        if a == None:
            print("The net area ratio has not been specified in deriving q_t")
            print("---> The default value of 0.8 has been assumed")
            a = 0.8
        df["qt"] = df.qc + (df.u2/1000)*(1-a)
    else:
        print("No u2 present in dataframe. No correction applied in deriving q_t")
        df["qt"] = df["qc"]
    
    return df

#%%
def uw(df,which="lengkeek et al_2018"):
    if "qt" not in df.columns:
        raise ValueError("Insert qt into dataframe")
        
    df["uw"] = np.nan
    df.loc[df.qc == 0].uw = 0
    df.loc[df.Rf == 0].uw = 0
    if which == "lengkeek et al_2018":
        def uw_lengkeek(row):
            row.uw = 19-4.12*(np.log10(5/row.qt)/np.log10(30/row.Rf))
            return row
        df = df.apply(uw_lengkeek, axis=1)  
    df = df.replace([np.inf, -np.inf], 0)
        
    return df

#%%
def sig_eff(df, water_table=1,sea_level=False):
    """
    Function which appends correlations to an existing dataframe of cpt data. 
    
    :df:            The CPT information. Columns need to be labelled 
                    ["qc","fs",Rf","z"]
    :water_table:   Depth of the water table relative to surface Assumed to be 1m 
                    below surface unless otherwise specified
        :
    """
    if "uw" not in df.columns:
        raise ValueError("Insert uw into dataframe")
    if "qt" not in df.columns:
        raise ValueError("Insert qt into dataframe")
    
    if water_table == 1:
        if sea_level == False:
            print("Default water table (1m below surface) has been used to calculate effective stress")
    if sea_level:       # Include the sea_level into the effective stress calculation
        water_table = 0     # Ground water should always be at surface if offshore
        seawater_density = 10.0910 # kN/m3
        surcharge = sea_level*seawater_density
    else:
        surcharge = 0

    df["depth_diff"] = abs(df.pen.diff())
     
    def sig_eff_contrib(row):
        """ 
        Function that calculates each row's contribution to the effective stress
        """
            
        if row.pen < water_table:
            row["sig_eff_contrib"] = row.uw*row.depth_diff
        if row.pen >= water_table: 
            row["sig_eff_contrib"] = row.uw*row.depth_diff - 10*row.depth_diff
        row["sig_contrib"] = row.uw*row.depth_diff
        return row
    
    df = df.apply(sig_eff_contrib, axis=1)          # Contribution of each row to eff. stress
    df["sig_eff"] = df.sig_eff_contrib.cumsum() + surcharge     # Result
    df["sig"] = df.sig_contrib.cumsum() + surcharge
    df.drop(columns=["sig_contrib","sig_eff_contrib","depth_diff"],inplace=True)

    return df

#%%
def qnet(df):
    if "sig_eff" not in df.columns:
        raise ValueError("Insert sig into dataframe")
    if "qt" not in df.columns:
        raise ValueError("Insert qt into dataframe")
    
    df["qnet"] = (df.qt*1000 - df.sig)/1000
    
    return df
    
#%%
def normalised_features(df):
    if "sig_eff" not in df.columns:
        df = sig_eff(df)
    if "qt" not in df.columns:
        df = qt(df, u2=False, a=0.8)
    
    # Robertson 1990
    df["Qtl"] = (df.qt*1000 - df.sig)/df.sig_eff
    df["Fr"] = ((df.fs*1000)/(df.qt*1000-df.sig))*100        # In [%]
    
    return df

#%%
def Ic(df):
    """

    """
    df = normalised_features(df)
    df["Ic"] = ((3.47 - np.log10(df.Qtl))**2 + (np.log10(df.Fr)+1.22)**2)**0.5
    return df

#%%
def Isbt(df):
    """
    Function which appends the Non-normalised soil behaviour index. 
    :df:            The CPT information. Columns need to be labelled 
                    ["qc","fs",Rf","z"]
    """
    if "Rf" not in df:
        df["Rf"] = (df.fs/df.qc)*100
    pa = 0.1                        # Atmospheric pressure in MPa
    np.seterr(divide = 'ignore')    # Hide warning made by fs = 0 to make output cleaner
    df["Isbt"] = ((3.47 - np.log10(df.qc/pa))**2 + (np.log10(df.Rf)+1.22)**2)**0.5
    
    return 

#%%
def Qtn(df, n = "robertson_2009"):
    pa = 100        # [kPa]
    if "sig_eff" not in df.columns:
        df = sig_eff(df)
    
    # Stress exponent [-]
    """ 
    The stress exponent is typically 1.0 for fine-grained soils, and ranges from
    0.5 (dense sand) to 0.9 (loose sand) for most coarse-grained soils. If 
    sig_eff > 1MPa, then the stress exponent will be essentially 1.0 for most soils
    """
    if n == "robertson_2009":      # Calculate n based on Robertson 2009
        df = Ic(df)
        n = 0.381*df.Ic + 0.05*(df.sig_eff/pa) - 0.15
        n.loc[n>1] = 1

    df["Qtn"] = ((df.qt-df.sig/1000)/(pa/1000))*((pa/df.sig_eff)**n)      # Normalised CPT resistance, corrected for overburden pressure
    
    return df
#%%
def normalised_qc_Rf(df, u2=False, a=0.8, water_table=1):
    """
    Gets the normalised cone resistance
    """
    if "sig_eff" not in df.columns:
        df = sig_eff(df)
    
    df["Qt"] = (df.qt*1000 - df.sig)/df.sig_eff
    df["nFr"] = (df.fs*1000/(df.qt*1000-df.sig))*100
    
    return df

#%%
def friction_angle(df):
    print("WARNING: The function for returning the friction angle needs to verified")
    df["phi"] = np.nan                            # Friction angle
    df.phi[df.Isbt < 2.95] = np.degrees(np.arctan(np.deg2rad((1/2.68)*(np.log10((df.qc*1000)/df.sig_eff)+0.29)))) # Friction angle as per Robertson & Campanella (1983)    
    
    return df

#%% 
def OCR(df, which="Mayne_2005",output_sig_p = False):
    if "qnet" not in df.columns:
        raise ValueError("Please create qnet")
    
    if which == "Mayne_2005":
        df["sig_p_eff"] = 0.32*(df.qnet*1000)**0.72
    
    df["OCR"] = df.sig_p_eff/df.sig_eff
    

    return df

#%%
def relative_density(df,which="baldi et al_1986"):
    """
    Definitons of relative density:
        Very loose      0 - 15%
        Loose           15 - 35%
        Medium dense    35 - 65%
        Dense           65 - 85%
        Very dense      85 - 100%
        
    Equation returns relative density as a percentage
    """

    
    print("_______________________________")
    print("Calculating relative density...")
    
    if "sig_eff" not in df.columns:
        raise ValueError("Insert sig_eff into dataframe")
    if "Isbt" not in df.columns:
        raise ValueError("Insert Isbt into dataframe")


    
    if which == "baldi et al_1986":
        print("Assumed that we have moderately compressible, NC, unaged, uncemented,quartz sand")
        C0 = 15.7
        C2 = 2.41
        pa = 100     # Reference pressure [kPa]
        df = sig_eff(df,water_table=1)
        df = qt(df)
        Qtn = (df.qt/(pa/1000))/(df.sig_eff/pa)**0.5        # Normalised CPT resistance, corrected for overburden pressure
        df["Dr"] = (1/C2)*np.log(Qtn/C0)
        df.Dr = df.Dr*100           # Return as a %
    
    elif which == "kulhawy & mayne_1990":
        Qc = 0.91       # Ranges from 0.91 (low compressibility) to 1.09 (high compressibility)
        t = 13000     # Aging factor (Pleistocene sands = ~13,000 years)
        pa = 100     # Reference pressure [kPa]
        df = sig_eff(df,water_table=1)
        df = qt(df)
        Qtn = (df.qt/(pa/1000))/(df.sig_eff/pa)**0.5        # Normalised CPT resistance, corrected for overburden pressure
        Qa = 1.2 + 0.05*np.log10(t/100)     # Aging factor
        df["OCR"] = 1
        Q_OCR = df.OCR**0.18                # OCR factor
        
        df["Dr"] = np.sqrt(Qtn/(305*Qc*Q_OCR*Qa))
        df.Dr = df.Dr*100           # Return as a %
        
    elif which == "kulhawy & mayne_1990_simplified":
        # df = qt(df)
        # df = sig_eff(df,water_table=4)
        pa = 100     # Reference pressure [kPa]
        Qtn = (df.qt/(pa/1000))/(df.sig_eff/pa)**0.5        # Normalised CPT resistance, corrected for overburden pressure
        
        df["Dr"] = np.sqrt(Qtn/350)
        df.Dr = df.Dr*100           # Return as a %
    
    elif which == "jamiolkowski_2003_dry":
        K0 = 0.5
        df = sig_eff(df,water_table=1)
        
        # Divided into different components to make equation clearer
        comp1 = df.qc/2.494
        comp2 = df.sig_eff*((1+2*K0)/3)
        df["Dr"] = (1/0.0296)*np.log(comp1*(comp2/100)**0.46)       # Outputs as a %

    elif which == "jamiolkowski_2003_saturated":
        K0 = 0.5
        df = sig_eff(df,water_table=1)
        
        # Divided into different components to make equation clearer
        comp1 = df.qc/2.494
        comp2 = df.sig_eff*((1+2*K0)/3)
        Dr_dry = (1/0.0296)*np.log(comp1*(comp2/100)**0.46)       # Outputs as a %
        
        comp3 = (1000*df.qc)/((100*df.sig_eff)**0.5)
        df["Dr"] = (((-1.87 + 2.32*np.log(comp3))/100)+1)*Dr_dry
        
    # Exclude clay
    df = Isbt(df)
    # df.loc[df.Isbt > 2.95] = np.nan
    # print("Clay has been excluded")

    print("_______________________________")
    return df

#%%
def G0(df,method="constant alpha", alpha=5.77,v=0.2,K_g = 300):
    """
    Initial shear modulus. alpha ranges from 1 to 20.
    Maasvlakte Kreftenheye: alpha = 5.77

    alpha = G0/qc = rigidity index
    """
    if method == "constant alpha":
        if alpha==5.77:
            print("NOTE: :alpha: needs to be checked. Can have huge variation here") 
        df["G0"] = df.qc*alpha
        
 
    elif method == "robertson_2009":
        if "u2" not in df.columns:
            raise ValueError("u2 is need to apply the Robertson (2009) correlation")
        if v == 0.2:
            print("A Poisson's ratio of 0.2 for sand has been assumed")
            
        E = 0.015*(10**(0.55*df.Ic+1.68))*(df.qt-df.sig)
        df["G0"] = E/(2*(1+v))
    
    elif method == "schnaid et al_2004":
        # The Schnaid constant varies from 110 to for uncemented sands, to 800
        # for cemented sands
        if "sig_eff" not in df.columns:
            raise ValueError("sig_eff is need to apply the Schnaid et al. (2004) correlation")
            
        pa = 101.3  # kPa
        df["G0"] = (K_g*(df.qc*df.sig_eff*pa))**(1/3)
        # raise ValueError("Korrelation needs to be double-checked")
        
    elif method == "schneider & moss_2011":
        """
        This is more of an updated version of schnaid et al_2004
        """
        # More like an updated version of Schnaid et al_2004
        if "Qtn" not in df.columns:
            raise ValueError("Qtn is need to apply the Schneider and Moss (2011) correlation")
        df["G0"] = K_g*df.qt/(df.Qtn**0.75)
        
    elif method == "stuyts et al_2022":
        if "sig_eff" not in df.columns:
            raise ValueError("Insert sig_eff into dataframe")
        df["G0"] = df.qc*2241*(df.qc*1000/np.sqrt(df.sig_eff))**-0.747
        
    else:
        raise ValueError("Method has not been specified")
    return df
