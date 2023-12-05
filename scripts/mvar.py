# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 14:05:09 2021

@author: kduffy
"""
import numpy as np

def mvar(mvar_dict, printer=True):
    mvar_expl = {1: "Nom. surface area of cone tip",
                 2: "Nom. surface area of friction casing",
                 3: "Net surface area quotient of cone tip",
                 4: "Net surface area quotient of friction casing",
                 5: "Cone distance to centre of friction casing",
                 6: "Friction present",
                 7: "PPT u1 present",
                 8: "PPT u2 present",
                 9: "PPT u3 present",
                 10: "Inclination measurement present",
                 11: "Use of back-flow compensator",
                 12: "Type of penetration test",
                 13: "Pre-excavated depth",
                 14: "Groundwater level",
                 15: "Water depth",
                 16: "End depth of penetration test",
                 17: "Stop criteria",
                 18: "future_use1",
                 19: "future_use2",
                 20: "Zero measurement of cone before penetration",
                 21: "Zero measurement of cone after penetration",
                 22: "Zero measurement friction before penetration",
                 23: "Zero measurement friction after penetration",
                 24: "Zero measurement PPT u1 before penetration",
                 25: "Zero measurement PPT u1 after penetration",
                 26: "Zero measurement PPT u2 before penetration",
                 27: "Zero measurement PPT u2 after penetration",
                 28: "Zero measurement PPT u3 before penetration",
                 29: "Zero measurement PPT u3 after penetration",
                 30: "Zero measurement inclination before penetration",
                 31: "Zero measurement inclination after penetration",
                 32: "Zero measurement inclination NS before penetration",
                 33: "Zero measurement inclination NS after penetration",
                 34: "Zero measurement inclination EW before penetration",
                 35: "Zero measurement inclination EW after penetration",
                 36: "future_use3",
                 37: "future_use4",
                 38: "future_use5",
                 39: "future_use6",
                 40: "future_use7",
                 41: "mileage",
                 99: "KDU custom: Cone diameter"
                 }
    
    # Check stopping criteria
    if 17 in mvar_dict:
        if isinstance(mvar_dict[17],float):
            mvar_dict[17] = int(mvar_dict[17])
            stop_criteria = {0:"end depth reached",
                             1:"max.penetration force",
                             2:"cone value",
                             3:"max. friction value",
                             4:"max. PPT value",
                             5:"max inclination value",
                             6:"obstacle",
                             7:"danger of buckling"}
            try:
                mvar_dict[17] = stop_criteria[mvar_dict[17]]
            except:
                mvar_dict[17] = "ID not recognised: " + str(mvar_dict[17])
    
    # Check type of CPT
    if 12 in mvar_dict:
        if isinstance(mvar_dict[12],float):
            mvar_dict[12] = int(mvar_dict[12])
            pen_type = {0:"electronic penetration test",
                        1:"mechanical discontinuous",
                        2:"mechanical continuous",
                        "-": "electronic penetration test"}
            try:
                mvar_dict[12] = pen_type[mvar_dict[12]]
            except:
                mvar_dict[12] = "ID not recognised: " + str(mvar_dict[12])

    
    # Convert certain datatypes to Boolean
    bool_dtypes = [6,7,8,9,10,11]
    check = [key in mvar_dict for key in bool_dtypes]       # Check if bool_dtypes are present
    present_dtypes = np.array(bool_dtypes)[np.array(check)]
    for dtype in present_dtypes:
        mvar_dict[dtype] = int(mvar_dict[dtype])
        if mvar_dict[dtype] == 1:
            mvar_dict[dtype] = True
        elif mvar_dict[dtype] == 0:
            mvar_dict[dtype] = False
        else:
            raise ValueError("Non-boolean value encountered")
       
    # Get cone diameter
    if 1 in mvar_dict:
        if (mvar_dict[1] > 1020) | (mvar_dict[1] < 980):
            mvar_dict[99] = 0.0357   # [m]
        if (mvar_dict[1] > 1510) | (mvar_dict[1] < 1490):
            mvar_dict[99] = 0.0437   # [m]
            
    if printer:
        # Print warning message if one of the crucial ones get filled
        if 1 in mvar_dict:
            if (mvar_dict[1] > 1020) | (mvar_dict[1] < 980):
                if (mvar_dict[1] > 1510) | (mvar_dict[1] < 1490):
                    print("A non-standard CPT cone has been used (" + str(mvar_dict[1]) + " mm2)")
        if 12 in mvar_dict:
            if (mvar_dict[12] == 1) | (mvar_dict[12] == 2):
                print("NOTE: A mechanical CPT was possibly used")
        if 13 in mvar_dict:
            if mvar_dict[13] != 0:
                print("Pre-excavation carried out to " + str(mvar_dict[13]) + "m")
        if 15 in mvar_dict:
            if mvar_dict[13] != 0:
                print("Depth of water above ground surface = " + str(mvar_dict[15]) + "m")

    
    new_dict = {}
    for key, name in mvar_dict.items():
        try:
            new_dict[mvar_expl[int(key)]] = name
        except:
            new_dict[int(key)] = name
    
    return new_dict

