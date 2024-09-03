# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 08:32:15 2020

@author: kevin

This deals with the dataframe output of the CPT class.
"""
import pandas as pd, numpy as np
# from read_cpt import CPT
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import os

imgDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images\\')

#%% 
def qt(df, u2=False, a=None):
    """
    a is the net area ratio and ranges from 0.70 to 0.85. Only used if pore 
    pressure was measured behind the cone tip (u2). Some typical values:
            Fugro 43.85mm 15cm2 cone: a = 0.58
    
    Returns qt in MPa
    """
    if ("u2" not in df.columns):
        print("No u2 present in dataframe. No correction applied in deriving q_t")
        df["qt"] = df["qc"]
    elif pd.isnull(df.u2).all():
        print("No u2 present in dataframe. No correction applied in deriving q_t")
        df["qt"] = df["qc"]
    else:
        if a == None:
            print("The net area ratio has not been specified in deriving q_t")
            print("---> The default value of 0.8 has been assumed")
            a = 0.8
        df["qt"] = df.qc + (df.u2/1000)*(1-a)
    
    return df

#%%
def Rf(df):
    if "qt" in df.columns:
        df["Rf"] = (df.fs/df.qt)*100
    else:
        df["Rf"] = (df.fs/df.qc)*100
    
    return df

#%%
def gammaSat(df,which="lengkeek et al_2018"):
    if "qt" not in df.columns:
        raise ValueError("Insert qt into dataframe")
        
    df["gammaSat"] = np.nan
    df.loc[df.qc == 0].gammaSat = 0
    df.loc[df.Rf == 0].gammaSat = 0
    
    if which == "simple":       
        """
        Generally applied when no fs information is available. Note that
        this is a gross simplification and assumes just sand and/or clay.
        
        Taken from  NEN9997-1 Table 2b
        
        #TODO: Interpolate between the values
        """
        # Sand
        df.loc[(df.qc >= 5) & (df.qc < 15)].gammaSat = 19   # Loose sand
        df.loc[(df.qc >= 15) & (df.qc < 25)].gammaSat = 20
        df.loc[(df.qc >= 25)].gammaSat = 21                 # Dense sand
        
        # Clay
        df.loc[(df.qc < 0.5)].gammaSat = 14                 # Soft clay
        df.loc[(df.qc >= 0.5) & (df.qc < 1)].gammaSat = 17  
        df.loc[(df.qc >= 1) & (df.qc < 2)].gammaSat = 19    # Stiff clay
        df.loc[(df.qc >= 2) & (df.qc < 5)].gammaSat = 20    # Clayey sand
        
    elif which == "lengkeek et al_2018":
        """
        Was an enhancement of the Robertson & Cabal (2010) correlation to better
        account for soft soils and peats in the Netherlands
        """
        def gammaSat_lengkeek(row):
            row.gammaSat = 19-4.12*(np.log10(5/row.qt)/np.log10(30/row.Rf))
            return row
        df = df.apply(gammaSat_lengkeek, axis=1)  
    
    elif which == "robertson and cabal_2010":
        """
        Was an enhancement of the Robertson & Cabal (2010) correlation to better
        account for soft soils and peats in the Netherlands
        """
        pa = 0.101      # [MPa]
        gamma_w = 10    # Unit weight of water[kN/m3] 
        def gammaSat_robCab(row):
            row.gammaSat = (0.27*np.log10(row.Rf) + 0.36*np.log10(row.qt/pa) + 1.236)/gamma_w
            return row
        df = df.apply(gammaSat_robCab, axis=1)  
        
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
    :sea_level:     Height of sea_level above seabed. If float is specified, water_table=0.
    NOTE: sig_eff assumes the CPT penetrates from the surface and no pre-drilling
    has occurred
    """
    if "gammaSat" not in df.columns:
        raise ValueError("Insert gammaSat into dataframe")
    if "qt" not in df.columns:
        raise ValueError("Insert qt into dataframe")
    
    if water_table == 1:
        if sea_level == False:
            print("Default water table (1m below surface) has been used to calculate effective stress")
    if abs(water_table) != water_table:
        print("Note: :water_table: input takes a positive value (i.e. +1m below surface)")
    if sea_level:       # Include the sea_level into the effective stress calculation
        water_table = 0     # Ground water should always be at surface if offshore
        seawater_density = 10.0910 # kN/m3
        surcharge = sea_level*seawater_density
    else:
        surcharge = 0

    df["pen"] = (df.z - df.z.iloc[0])*-1
    df["depth_diff"] = abs(df.pen.diff())
    df["sig_contrib"] = np.nan
    df["sig_eff_contrib"] = np.nan
    def sig_eff_contrib(row):
        """ 
        Function that calculates each row's contribution to the effective stress
        """
            
        if row.pen < water_table:       # If above water table
            row["sig_eff_contrib"] = row.gammaSat*row.depth_diff
        if row.pen >= water_table:      # If below water table
            row["sig_eff_contrib"] = row.gammaSat*row.depth_diff - 10*row.depth_diff
        row["sig_contrib"] = row.gammaSat*row.depth_diff
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
def Ic(df, plot=False,chart="robertson_1990_normalised"):
    """
    Includes calculation of the stress exponent :n: as listed in Eq. 7 of Robertson, 2009.
    The value of :n: is typically 1.0 for fine-grained soils, and ranges from
    0.5 (dense sand) to 0.9 (loose sand) for most coarse-grained soils. If 
    sig_eff > 1MPa, then the stress exponent will be essentially 1.0 for most soils
    
    Robertson, 2009
    Also includes the normalised features Qtn and Fr
    From Bruno Stuyts
    """
    def Ic_func(qt, fs, sig, sig_eff,z):
        try:
            pa=101.0        # Atmospheric pressure [kPa]

            def Qtn(qt, sig, sig_eff, n, pa=0.001 * pa):
                return ((qt - 0.001 * sig) / pa) * ((pa / (0.001 * sig_eff)) ** n)
        
            def Fr(fs, qt, sig):
                return 100 * (fs / (qt - 0.001 * sig))
        
            def stress_exponent(ic, sig_eff, pa):
                return min(1, 0.381 * ic + 0.05 * (sig_eff / pa) - 0.15)
        
            def soilbehaviourtypeindex(qt, fr):
                return np.sqrt((3.47 - np.log10(qt)) ** 2 + (np.log10(fr) + 1.22) ** 2)
        
            def rootfunction(ic, qt, fs, sig, sig_eff):
                _fr = Fr(fs, qt, sig)
                _n = stress_exponent(ic, sig_eff,pa)
                _qtn = Qtn(qt, sig, sig_eff, _n)
                return ic - soilbehaviourtypeindex(_qtn, _fr)
        
            Ic_min=1.0
            Ic_max=4.0      # Search for the solution between ic_min and ic_max
            _Ic = brentq(rootfunction, Ic_min, Ic_max, args=(qt, fs, sig, sig_eff))
            _exponent_zhang = stress_exponent(_Ic, sig_eff, pa)
            _Qtn = Qtn(qt, sig, sig_eff, _exponent_zhang)
            _Fr = Fr(fs, qt, sig)

        except:
            _Ic = np.nan
            _Qtn = np.nan
            _Fr = np.nan
    
        return _Ic, _Qtn, _Fr
    
    df["Ic"] = np.nan
    df["Qtn"] = np.nan
    df["Fr"] = np.nan
    for index,row in df.iterrows():
        Ic, Qtn, Fr = Ic_func(row.qt,row.fs,row.sig,row.sig_eff,row.z)
        df.Ic.iloc[index] = Ic
        df.Qtn.iloc[index] = Qtn
        df.Fr.iloc[index] = Fr
     
    if plot:
        fig = plt.figure(figsize=(5,5))
        ax = fig.gca()
        
        # Include Robertson chart as background; Need to make adjustments as [ax] is on a log scale
        ax_tw_x = ax.twinx()
        ax_tw_x.axis('off')
        ax2 = ax_tw_x.twiny()
        
        
        img = plt.imread(imgDir + chart + ".PNG")
        ax2.imshow(img,extent=[1,10,1,1000], aspect="auto")
        ax2.axis('off')
            
        ax.scatter(df["Fr"],df["Qtn"],ec="k",alpha=0.6,s=10)

        ax.set_xlabel(r"Normalised friction ratio, $F_r$ [%]")
        ax.set_ylabel(r"Normalised cone resistance, $Q_{tn}$ [-]")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.1,10)
        ax.set_ylim(1,1000)
        # ax.ticklabel_format(useOffset=False, style='plain')    # Don't use scientific notation
        ax.set_xticks([0.1,1,10])
        ax.set_yticks([1,10,100,1000])
        ax.patch.set_facecolor('None')      # No background on axis
        ax.set_zorder(ax2.get_zorder()+1)   # Show scatter points above plot
        plt.show()
    
    return df


#%%
def Isbt(df,plot=False,chart="robertson_1990_nonnormalised"):
    """
    Function which appends the Non-normalised soil behaviour index (see Robertson, 2010)
    :df:            The CPT information. Columns need to be labelled 
                    ["qc","fs",Rf","z"]
                     
    For in-situ effctive stresses between 50-150kPa there is little difference between the
    non-normalised I_sbt and the normalised I_c (Robertson, 2010)
    """
    if "Rf" not in df.columns:
        raise ValueError("Please add Rf first before calculating Isbt")
        
    df.fs.loc[df.fs == 0] = np.nan  # Used to hide warnings caused by np.log10
    df.Rf.loc[df.Rf == 0] = np.nan  # Used to hide warnings caused by np.log10

    pa = 0.1                        # Atmospheric pressure in MPa
    np.seterr(divide = 'ignore')    # Hide warning made by fs = 0 to make output cleaner
    df["Isbt"] = ((3.47 - np.log10(df.qc/pa))**2 + (np.log10(df.Rf)+1.22)**2)**0.5
    
    if plot:
        fig = plt.figure(figsize=(5,5))
        ax = fig.gca()
        
        # Include Robertson chart as background; Need to make adjustments as [ax] is on a log scale
        ax_tw_x = ax.twinx()
        ax_tw_x.axis('off')
        ax2 = ax_tw_x.twiny()
        img = plt.imread(imgDir + chart + ".PNG")
        ax2.imshow(img,extent=[1,10,1,1000], aspect="auto")
        ax2.axis('off')
        
        if chart == "robertson_1990_nonnormalised":
            ax.scatter(df.Rf,df.qc/pa,ec="k",alpha=0.6,s=10)
        elif chart == "robertson_1986_nonnormalised":
            ax.scatter(df.Rf,df.qc,ec="k",alpha=0.6,s=10)
        ax.set_xlabel(r"Friction ratio, $R_f$ [%]")
        ax.set_ylabel(r"Cone resistance, $q_c \slash p_a$ [-]")

        if chart == "robertson_1990_nonnormalised":
            ax.set_xlim(0.1,10)
            ax.set_xticks([0.1,1,10])
            ax.set_ylim(1,1000)
            ax.set_yticks([1,10,100,1000])
            ax.set_xscale("log")
            ax.set_yscale("log")
        elif chart == "robertson_1986_nonnormalised":
            ax.set_xlim(0,8)
            # ax.set_xticks([0.1,1,10])
            ax.set_ylim(0.1,100)
            ax.set_yticks([1,10,100]) 
            ax.set_yscale("log")
        # ax.ticklabel_format(useOffset=False, style='plain')    # Don't use scientific notation
        
        ax.patch.set_facecolor('None')      # No background on axis
        ax.set_zorder(ax2.get_zorder()+1)   # Show scatter points above plot
        plt.show()
    
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
        if "Ic" not in df.columns:
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
