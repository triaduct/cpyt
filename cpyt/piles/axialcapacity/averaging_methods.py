# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 19:08:02 2020

@author: kevin
"""
import pandas as pd, numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from copy import deepcopy

import cpyt.interpretation.correlations as correlations


"""
NOTE: 
    
    -ve implies downwards direction
"""
#%%
def koppejan(cpt, pile_dia, target_depth, plot_min_path = False):
    """
    4D/8D method according to Koppejan
    """
    ## Check if the required CPT data exists
    if len(cpt[cpt.z < target_depth - 4*pile_dia]) == 0:
        raise ValueError("ERROR: CPT doesn't cover depth required for averaging technique. Choose a more suitable CPT in the .txt file\n")
    
    # Find the minimum qcavg (dependent on endpoint of qc1)
    qc_avg = 9999            # Initialise qcavg
    
    for endpoint in cpt[(cpt.z <= target_depth-0.7*pile_dia) & (cpt.z >= target_depth - 4*pile_dia)].z:    # Identify endpoint where qcavg is minimal
        ## qc1
        """Mean of values until endpoint"""
        cpt1 = cpt[(cpt.z <= target_depth) & (cpt.z >= endpoint)]
        qc1 = cpt1.qc.mean()
    
        ## qc2
        """Min. path rule from 4D to 0.7D"""
        cpt2 = cpt[(cpt.z <= target_depth) & (cpt.z >= endpoint)]
        cpt2 = cpt2.iloc[::-1]                                        # Flip cpt for use in expanding minimum
        cpt2["min_path"] = cpt2.qc.expanding().min()                  # Expanding minimum
        qc2 = cpt2.min_path.mean()
        
        ## qc3 
        """Min. path rule from pile tip to 8D"""
        start = cpt2.min_path.iloc[-1]                               # Start min path from the end of min path for qc2
        cpt3 = cpt[(cpt.z >= target_depth) & (cpt.z <= target_depth + 8*pile_dia)]   # Search for min within this cpt
        cpt3 = cpt3.iloc[::-1]                                        # Flip cpt for use in expanding minimum
        cpt3["min_path"] = np.nan
        cpt3.qc.iloc[0] = start                                      # Continue on from path from qc2
        cpt3.min_path = cpt3.qc.expanding().min()                    # Expanding minimum
        qc3 = cpt3.min_path.mean()
        
        ## qc_avg
        qc_avg_temp = 0.5*(0.5*(qc1+qc2) + qc3)
        if qc_avg_temp < qc_avg:
            qc_avg = qc_avg_temp
            qc1_min = qc1
            qc2_min = qc2
            qc3_min = qc3
            qc2_min_path = np.array([cpt2.z,cpt2.min_path])
            qc3_min_path = np.array([cpt3.z,cpt3.min_path])
            endpoint_final = endpoint
            
    print(f"qc1 = {qc1_min:.2f}")
    print(f"qc2 = {qc2_min:.2f}")
    print(f"qc3 = {qc3_min:.2f}")
    print(f"Endpoint = {endpoint_final:.2f}")
    
    ## Plot the minimum path
    if plot_min_path == True:
        
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(cpt.qc, cpt.z)
        ax.plot(qc2_min_path[1], qc2_min_path[0], c='r',label="Minimum path",alpha=0.7)
        ax.plot(qc3_min_path[1], qc3_min_path[0], c='r',alpha=0.7)
        ax.axhline(target_depth,linewidth=0.5, c='black')
        ax.hlines([target_depth - 4*pile_dia,target_depth - 0.7*pile_dia,target_depth + 8*pile_dia],-1,70, 
                  linewidth=0.5,linestyle="dashed",color="darkgray")

        # ax.text(0.5, 0.5, 'text', fontsize=30, va='center', ha='center', backgroundcolor='w')
        # ax.text(0.5, 0.5, 'text', fontsize=30, va='center', ha='center', backgroundcolor='w')
        

        xlim = cpt[(cpt.z <= target_depth+8*pile_dia) & (cpt.z >= target_depth - 4*pile_dia)].qc.max() # Max qc in region of pile head (for plotting)
        xlim = np.round(xlim/10)*10+10
        ax.set_xlim(0, xlim)
        ax.set_ylim(np.round(target_depth - 6*pile_dia,0), np.round(target_depth + 10*pile_dia,0))
        ax.set_xlabel(r"Cone resistance $q_c$ [MPa]")
        ax.set_ylabel("Depth [mNAP]")
        
        ax.text(xlim+0.5, target_depth, 'Pile tip', fontsize=7, va='center', ha='left', color="black")
        ax.text(xlim+0.5, target_depth - 4*pile_dia, '-4D', fontsize=7, va='center', ha='left', color="darkgray")
        ax.text(xlim+0.5, target_depth - 0.7*pile_dia, '-0.7D', fontsize=7, va='center', ha='left', color="darkgray")
        ax.text(xlim+0.5, target_depth + 8*pile_dia, '+8D', fontsize=7, va='center', ha='left', color="darkgray")
        
        ax.legend()
        plt.show()

    print(f"Using the Koppejan 4D/8D method, qc_avg at {target_depth:.2f}m is {qc_avg:.2f} MPa") 
    return qc_avg


def lcpc(cpt, pile_dia, target_depth):
    """
    
    """

    D15 = cpt[(cpt.z >= target_depth - 1.5*pile_dia) & (cpt.z <= target_depth + 1.5*pile_dia)]   # Region across which the averaging will be applied
    qc_avg = D15.qc.mean()
    D15.loc[D15.qc >= 1.3*qc_avg,"qc"] = 1.3*qc_avg        # Truncate qc values
    D15.loc[D15.qc <= 0.7*qc_avg,"qc"] = 0.7*qc_avg        # Truncate qc values
    qc_avg = D15.qc.mean()
    
    print(f"Using the LCPC method, qc_avg at {target_depth:.2f}m is {qc_avg:.2f} MPa") 
    return qc_avg


def lpc_2012(cpt, pile_dia, target_depth, top_bearing_layer):
    """
    Similar to the original 1.5D LCPC method. Used in the French design method AFNOR
    
    Only qc values within the :bearing_layer: are considered
    
    See Verheyde, 2019
    """
    z_above = max(0.5*pile_dia,0.5)  # z_above has to be at least 0.5m. 
    z_below = max(1.5*pile_dia,1.5)  # z_below has to be at least 1.5m. 
    
    D15_05 = cpt[(cpt.z >= target_depth - z_below) & (cpt.z <= target_depth + z_above)]   # 1.5 diameters below pile tip, 0.5 dia above
    D15_05.loc[cpt.z > top_bearing_layer,"qc"] = np.nan   # Exclude any values not in the bearing layer
    qcm = D15_05.qc.mean()
    D15_05.loc[D15_05.qc > 1.3*qcm,"qc"] = 1.3*qcm      # Limit qc values to 1.3*qcm
    qce = D15_05.qc.mean()      # Equivalent tip resistance
    
    return qce    


def de_boorder(cpt, pile_dia, target_depth):
    HD_a = 8.3              # Distance over which the cosine function is applied above the pile
    HD_b = 15.5             # "..." below the pile
    f = 13.5                # Damping factor
    s = 0.9                 # Reshapes the weight related to the stiffness ratio
    
    cpt = cpt.loc[(cpt.z <= target_depth + HD_a*pile_dia) & (cpt.z >= target_depth - HD_b*pile_dia)]
    cpt["HD"] = np.nan
    cpt.loc[cpt.z >= target_depth,"HD"] = HD_a
    cpt.loc[cpt.z <= target_depth, "HD"] = HD_b
    cpt["x"] = abs((target_depth-cpt.z)/(pile_dia*cpt.HD))
    cpt["w1"] = np.exp(-f*cpt.x)*np.cos(0.5*np.pi*cpt.x)   # First weight relating to the cosine dampening function and distance to pile tip
    near_z_ix = cpt.z.sub(target_depth).abs().idxmin()       # Index of row with depth closest to pile depth
    qc_tip = cpt.loc[near_z_ix, "qc"]
    cpt["w2"] = (qc_tip/cpt.qc)**s                        # Wegiht of one point related to stiffness ratio
    cpt["w3"] = cpt.w1*cpt.w2                              # Total weight of qc at one point
    
    qc_w = cpt.qc*cpt.w3/(cpt.w3.sum())
    qc_avg = qc_w.sum()
    
    print(f"Using the De Boorder method, qc_avg at {target_depth:.2f}m is {qc_avg:.2f} MPa") 
    return qc_avg


def boulanger_dejong(cpt, dia, target_depth):
    """
    Method prescribed in Boulanger & DeJong (2018). Recommend averaging method
    in the Unified pile design method (Lehane et al., 2022)
    
    :dia:           Diameter of penetrometer.
                    NOTE: The Boulanger & DeJong method was not explicitly formulated for 
                    piles (namely scale effects). This was the further research of de 
                    Lange (2017), de Boorder (2021) and so on.
    :target_depth:  depth for calcualtion, with respect to cpt.z (-ve => downwards)       

    """
    z50_ref=4.0
    mz=3.0
    m50=0.5
    mq=2
            
    qt_tip = cpt.qc.iloc[(cpt.z-target_depth).abs().argsort()[:2]]      # Cone resistance at :target_depth:
    cpt[["C1","C2","z50","w1","w2","w1w2","wc"]] = np.nan
    cpt["z_norm"] = -1*(cpt.z-target_depth)/dia    # Normalised depth. -ve values are above the pile tip, +ve are below
    
    # Equation 6
    cpt.C1.loc[cpt.z_norm >= 0] = 1
    cpt.C1.loc[(cpt.z_norm >= -4) & (cpt.z_norm < 0)] = 1 + cpt.z_norm/8
    cpt.C1.loc[cpt.z_norm < -4] = 0.5
    
    cpt.C2.loc[cpt.z_norm >= 0] = 1         # Equal to unity below the pile tip (+ve numbers imply downwards)
    cpt.C2.loc[cpt.z_norm <= 0] = 0.8       # Equal to 0.8 above the pile tip

    cpt.z50 = 1+2*(cpt.C2*z50_ref - 1)*(1-(1/(1+(qt_tip/cpt.qc)**m50)))     # Equation 7
    
    cpt.w1 = cpt.C1/(1+(cpt.z_norm/z50_ref)**mz)        # Equation 5
    cpt.w2 = np.sqrt(2/(1+(cpt.qc/qt_tip)**mq))         # Equation 8
    cpt.w1w2 = cpt.w1 * cpt.w2                          
    cpt.wc = (cpt.w1*cpt.w2)/cpt.w1w2.sum()             # Equation 3
    qc_w = (cpt.qc*cpt.wc)/(cpt.wc.sum()) 
    qc_avg = qc_w.sum()
    
    print(f"Using the Boulanger method, qc_avg at {target_depth:.2f}m is {qc_avg:.2f} MPa") 
    return qc_avg
    

def fleming_and_thorburn(cpt,pile_dia, target_depth):
    """
    Fleming, W. G. K. and Thorburn, S., 1983. Recent piling advances, state of 
    the art report. Proceedings of the International Conference on Advances in 
    Piling and Ground Treatment for Foundations, ICE London, pp 1-16.
    
    Can't find this paper. Use Bazu et al. 2010 for reference. Used in designing
    for screw displacement piles in the USA.
    
    4D below the pile tip was used specifically for the purpose of determing the 
    capacity of screw displacement piles, as per NeSmith, 2002
    """
    D4_below = cpt[(cpt.z > target_depth - 4*pile_dia) & (cpt.z <= target_depth)]   # 4D below pile tip
    D4_above = cpt[(cpt.z >= target_depth) & (cpt.z < target_depth + 4*pile_dia)]   # 4D above pile tip
    qc0 = D4_below.qc.mean()
    qc1 = D4_below.qc.min()
    
    D4_above.qc.loc[D4_above.qc > qc1] = qc1        # Note: it's not clear from the paper if these values should be eliminated altogether or thresholded. Thresholded has been chosen
    qc2 = D4_above.qc.mean()
    
    qc_avg = 0.25*qc0 + 0.25*qc1 + 0.5*qc2
    
    print(f"Using the Fleming and Thorburn method, qc_avg at {target_depth:.2f}m is {qc_avg:.2f} MPa") 
    return qc_avg
  
    
def de_beer(cpt, pile_dia, target_depth): 
    # I haven't customised this
    # to give values for a specific depth, instead it just looks at the whole CPT
    def de_beer_calc(cpt, pile_dia,dia_cone=None):
        # NOTE: This is from Bruno Stuyts 
        # https://github.com/snakesonabrain/methodedebeer/blob/master/debeer/calculation.py
        """
        Requies input of
        :gammaSat:    Only really makes a big difference in the upper layers
        :pen:   Penetration w.r.t to surface. Positive indicates deeper
        """
        # cpt["pen"] = cpt.pen*-1     # Penetration is negative in this code
        if dia_cone == None:
            dia_cone = 0.0357        # 35.7mm for 10cm2 cone
            print("A diameter of 35.7 mm has been assumed for the De Beer method.")
        
        def resample(cpt, spacing=0.2):
            pen_new = np.arange(cpt.pen.min(), cpt.pen.max(), spacing)
            z_new = np.interp(pen_new,cpt.pen.values,cpt.z.values)
            qc_new = np.interp(pen_new,cpt.pen.values,cpt.qc.values)
            qt_new = np.interp(pen_new,cpt.pen.values,cpt.qt.values)
            gammaSat_new = np.interp(pen_new,cpt.pen.values,cpt.gammaSat.values)
            sig_eff_new = np.interp(pen_new,cpt.pen.values,cpt.sig_eff.values)
            cpt_new = pd.DataFrame({"z":z_new,"pen":pen_new,"qc":qc_new,"qt":qt_new,"gammaSat":gammaSat_new,"sig_eff":sig_eff_new})
            return cpt_new
        
        def optimisation_func(beta, hd, frictionangle):             # Eq 60
            return ((np.tan(0.25 * np.pi + 0.5 * frictionangle) *
                    np.exp(0.5 * np.pi * np.tan(frictionangle)) *
                    np.sin(beta) *
                    np.exp(beta * np.tan(frictionangle))) / \
                   (1 + np.sin(2 * frictionangle))) - hd
        
        def stress_correction(qc, po, diameter_pile, diameter_cone, gamma, hcrit=0.2):
            h_prime_crit = hcrit * (diameter_pile / diameter_cone)
            return qc * ((1.0 + ((gamma * h_prime_crit) / (2.0 * po))) / (1.0 + ((gamma * hcrit) / (2.0 * po))))
        
        def calculate_base_resistance_standard_diameter(cpt, pile_dia,dia_cone, vanimpecorrection=False, hcrit=0.2):
            """
            Calculates the base resistance according to De Beer's method for a pile diameter which is a multiple of 0.2m. The calculation happens in five steps:
                - Step 1: Correct the cone resistance for the different failure surface for a pile and a CPT using Equation 62 from De Beer's paper. This correction is especially necessary for the shallow layer where the angle beta is lower than 90°
                - Step 2: Apply a correction for the different stress level for a pile compared to a CPT
                - Step 3: Account for the transition from weaker the stronger layers by working downward along the CPT trace. The increase of resistance will be slower for a pile compared to a CPT
                - Step 4: Account for the transition from stronger to weaker layers by working through the CPT trace from the bottom up. A weaker layer will be felt sooner by the model pile than by the CPT
                - Step 5: Take the average unit base resistance for one diameter below the given level. The average value should note be greater than :math:`q_{p,q+1}` at the given level.
          
            For numerical stability, rows with zero cone resistance at the top of the cone resistance trace are discarded.
            :param pile_diameter: Diameter of the pile as a multiple of 0.2m
            :param vanimpecorrection: Boolean determining whether the upward correction according to De Beer's original paper (default) or Van Impe (multiplier of 2) needs to be taken into account.
            :param hcrit: :math:`h_{crit}` adopted for De Beer's calculation (based on the mechanical cone). Default=0.2m
            :return: Returns a dataframe `calc` with the different correction stages
            """
    
            
            calc = deepcopy(cpt)
            calc = calc[calc.qc > 0].reset_index(drop=True)
            # ----------------------------------------------------
            # Step 1: Shallow depth failure surface correction
            # ----------------------------------------------------
        
            # Build equation for phi
            phi = np.linspace(np.deg2rad(0.01), np.deg2rad(50), 250)
        
            def v_bd_func(frictionangle):
                return 1.3 * np.exp(2 * np.pi * np.tan(frictionangle)) * ((np.tan(0.25 * np.pi + 0.5 * frictionangle)) ** 2)  # Eq. 21
            v_bd = v_bd_func(phi)
    
            
            # Calculate phi according to Equation 23
            # calc['phi [deg]'] = np.rad2deg(interp1d(v_bd, phi,fill_value="extrapolate")(1000 * calc.qc / calc.sig_eff))     # Eq. 22
            calc['phi [deg]'] = np.rad2deg(interp1d(v_bd, phi,fill_value="extrapolate")(calc.sig_eff))     # Eq. 23
    
            # Determine the values of the normalised depths h/d and h/D
            calc['h/d [-]'] = calc.pen / dia_cone
            calc['h/D [-]'] = calc.pen / pile_dia
            
            # Find values of beta for cone penetration test and pile according to Equation 60
            for i, row in calc.iterrows():
                try:
                    root = brentq(
                        f=optimisation_func,
                        a=0,
                        b=0.5 * np.pi,
                        args=(row['h/d [-]'], np.deg2rad(row['phi [deg]'])))
                except:
                    root = 0.5 * np.pi
                calc.loc[i, 'beta_c [rad]'] = root
                try:
                    root = brentq(
                        f=optimisation_func,
                        a=0,
                        b=0.5 * np.pi,
                        args=(row['h/D [-]'], np.deg2rad(row['phi [deg]'])))
                except:
                    root = 0.5 * np.pi
                calc.loc[i, 'beta_p [rad]'] = root
                
            # Apply Equation 62 to obtain qp
            calc['qp [MPa]'] = calc.qc / \
                (np.exp(
                  2 *
                  (calc['beta_c [rad]'] - calc['beta_p [rad]']) *
                  np.tan(np.deg2rad(calc['phi [deg]']))))
        
            # ----------------------------------------------------
            # Step 2: Stress level correction
            # ----------------------------------------------------
        
            calc['A qp [MPa]'] = list(map(lambda _qp, _po, _gamma, _qc: min(_qc, stress_correction(
                qc=_qp, po=_po, diameter_pile=pile_dia, diameter_cone=dia_cone, gamma=_gamma, hcrit=hcrit)),
                               calc['qp [MPa]'], calc.sig_eff, calc.gammaSat, calc.qc))
        
            # --------------------------------------------------------------------
            # Step 3: Corrections for transition from weaker to stronger layers
            # --------------------------------------------------------------------
            for i, row in calc.iterrows():
                if i > 0:
                    calc.loc[i, 'qp,j+1 [MPa]'] = min(row['A qp [MPa]'] ,
                        calc.loc[i-1, 'qp,j+1 [MPa]'] + \
                        (row['A qp [MPa]'] - calc.loc[i-1, 'qp,j+1 [MPa]']) * \
                        (dia_cone / pile_dia))
                else:
                    calc.loc[i, 'qp,j+1 [MPa]'] = 0
        
            # --------------------------------------------------------------------
            # Step 4: Corrections for transition from stronger to weaker layers
            # --------------------------------------------------------------------
            if vanimpecorrection:
                coefficient = 2.0
            else:
                coefficient = 1.0
        
            qu = np.zeros(len(calc.pen))
            # Assign the last value of qd as the starting value of qu
            qu[-1] = calc['qp,j+1 [MPa]'].iloc[-1]
            for i, _qd in enumerate(calc['qp,j+1 [MPa]']):
                if i != 0:
                    qu[-1 - i] = min(
                        qu[-i] +
                        coefficient * (calc['qp,j+1 [MPa]'].iloc[-1 - i] - qu[-i]) *
                        (dia_cone / pile_dia),
                        calc['qp,j+1 [MPa]'].iloc[-1 - i]
                    )
            calc['qp,q+1 [MPa]'] = qu
        
            # --------------------------------------------------------------------
            # Step 5: Averaging to 1D below the reference level
            # --------------------------------------------------------------------
            for i, row in calc.iterrows():
                try:
                    _window_data = calc[
                        (calc['z [m]'] >= row['z [m]']) &
                        (calc['z [m]'] <= (row['z [m]'] + pile_dia))]
                    calc.loc[i, "qb [MPa]"] = min(
                        row['qp,q+1 [MPa]'],
                        _window_data['qp,q+1 [MPa]'].mean())
                except:
                    calc.loc[i, "qb [MPa]"] = row['qp,q+1 [MPa]']
        
            return calc
        
        #%% Action
        if cpt.pen.min() < 0:
            raise ValueError("cpt.pen must be positive numbers")
            
        if "gammaSat" not in cpt.columns:
            print("Recommend calculating gammaSat and sig_eff for the De Beer method")
            print("For now, the default values have been chosen")
            water_table = 1
            cpt = correlations.qt(cpt, u2=False, a=0.8)
            cpt.loc[cpt.pen<water_table].gammaSat = 16
            cpt.loc[cpt.pen>water_table].gammaSat = 20
            cpt = correlations.sig_eff(cpt, water_table=water_table,sea_level=False)
        cpt = resample(cpt)
        
        # Check if "standard diameter" i.e. a multiple of 0.2. Otherwise need to linear interpolate between these
        if pile_dia % 0.2 != 0:
            diameter1 = np.round(pile_dia - 0.2, 1)
            diameter2 = np.round(pile_dia + 0.2, 1)
            
            calc1 = calculate_base_resistance_standard_diameter(cpt, diameter1, dia_cone=dia_cone, vanimpecorrection=False, hcrit=0.2)
            calc2 = calculate_base_resistance_standard_diameter(cpt, diameter2, dia_cone=dia_cone, vanimpecorrection=False, hcrit=0.2)
            
            calc=calc1
            qc_avg = calc1["qb [MPa]"] + (pile_dia-diameter1)*((calc2["qb [MPa]"]-calc1["qb [MPa]"])/(diameter2-diameter1))        # interpolate
        else:
            calc = calculate_base_resistance_standard_diameter(cpt, pile_dia, dia_cone=dia_cone, vanimpecorrection=False, hcrit=0.2)
            qc_avg = calc["qb [MPa]"] 
        
        calc["qc_avg"] = qc_avg
        return calc    


    calc=de_beer_calc(cpt, pile_dia,dia_cone=None)
    find_closest = calc.z.sub(target_depth).abs()
    near_z_ix1 = find_closest.idxmin()         # Index of row with depth closest to pile depth
    near_z1 = calc.z.iloc[near_z_ix1]
    
    if near_z_ix1 == target_depth:
        qc_avg = calc.loc[near_z_ix1, "qc_avg"]  
    else:       # Interpolate
        find_closest = find_closest.drop(index=near_z_ix1)
        near_z_ix2 = find_closest.idxmin()    # Next nearest row
        near_z2 = calc.z.iloc[near_z_ix2]
        
        qc_avg1 = calc.loc[near_z_ix1, "qc_avg"]            # Get qc at point closest to required depth
        qc_avg2 = calc.loc[near_z_ix2, "qc_avg"]            # Get qc at point closest to required depth
        qc_avg = qc_avg1+(target_depth-near_z1)*((qc_avg2-qc_avg1)/(near_z2-near_z1))   # Interpolation

    print(f"Using the De Beer method, qc_avg at {target_depth:.2f}m is {qc_avg:.2f} MPa") 
    return qc_avg
      


