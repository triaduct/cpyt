# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 14:23:21 2021

@author: kduffy
"""

def unit_check(unit_input):
    unit_dict = {1: "m",        # Penetration length
                 2: "MPa",      # Cone resistance
                 3: "MPa",      # Friction resistance
                 4: "%",        # Friction ratio
                 5: "MPa",      # Pore pressure u1
                 6: "MPa",      # Pore pressure u2
                 7: "MPa",      # Pore pressure u3
                 8: "Â°",      # Inclination (resultant)
                 9: "Â°",      # Inclination N-S
                 10: "Â°",     # Inclination E-W
                 11: "m",       # Corrected depth, relative to surface
                 12: "s",       # Time        
        }
    
    for key, value in unit_input.items():
        if (key == 8) | (key == 9) | (key == 10):
            continue                        # Skip inclination measurements as it's reported in many different ways
        try:
            if unit_dict[key] != value:
                print()
                print(f"NOTE: The units in the GEF file ({value}) is not equal to the standard unit ({unit_dict[key]})")
                print()
        except:
            pass
        
def check_magnitude(df):
    """
    Desired units:  qc [MPa], fs [MPa], u2 [MPa]
    """
    print("Checking units...")
    
    # Cone resistance
    if (df.qc.mean() > 0.2) & (df.qc.mean() < 1000):
        print("qc values are ok")
    else:
        df.qc *= 1000       
        print("qc values have been converted from kPa to MPa")
        print(f"Mean qc of {df.qc.mean():.2f}")
    
    # Friction sleeve (should be in MPa)
    if (df.fs.mean() > 0.05) & (df.fs.mean() < 1):
        print("fs values are ok")
    else:
        print(f"Mean fs of {df.fs.mean():.2f} kPa")
        df.fs /= 1000       
        print("fs values have been converted from kPa to MPa")
    
    # Pore pressure
    if (df.u2.mean() > 0.05) & (df.u2.mean() < 1):
        print("u2 values are ok")
    else:
        df.u2 / 1000       
        print("u2 values have been converted from kPa to MPa")
        print(f"Mean u2 of {df.u2.mean():.2f}")

    # # Friction Ratio
    # if (df.Rf.mean() > 0.01) & (df.Rf.mean() < 10):
    #     print("Rf values are ok")
    # else:
    #     df.Rf *= 100     
    #     print("Rf values have been converted from dimensionless to %")
    #     print(f"Mean Rf of {df.Rf.mean():.2f}")

    return(df)
    print("-----------")
        
