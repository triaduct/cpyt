# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 14:05:09 2021

@author: kduffy
"""
import numpy as np

def mtext(mtext_dict, printer=True):
    mtext_expl = {1: "client",
                 2: "name of the project",
                 3: "name of the location",
                 4: "cone type and serial number",
                 5: "Mass and geometry of probe apparatus, including anchoring",
                 6: "applied standard, including class",
                 7: "own coordinate system",
                 8: "own reference level",
                 9: "fixed horizontal level",
                 10: "orientation direction biaxial inclination measurement (N-direction)",
                 11: "unusual circumstances",
                 12: "future_use1",
                 13: "future_use2",
                 14: "future_use3",
                 15: "future_use4",
                 16: "future_use5",
                 17: "future_use6",
                 18: "future_use7",
                 19: "future_use8",
                 20: "correction method for zero drift",
                 21: "method for processing interruptions",
                 22: "remarks1",
                 23: "remarks2",
                 24: "future_use9",
                 25: "future_use10",
                 26: "future_use11",
                 27: "future_use12",
                 28: "future_use13",
                 29: "future_use14",
                 30: "formula or reference 1",
                 31: "formula or reference 2",
                 32: "formula or reference 3",
                 33: "formula or reference 4",
                 34: "formula or reference 5",
                 35: "formula or reference 6",
                 36: "future_use15",
                 37: "future_use16",
                 38: "future_use17",
                 39: "future_use18",
                 40: "future_use19",
                 41: "highway, railway or dike code",
                 42: "future_use20",
                 43: "future_use21",
                 }

    if printer:
        # Print warning message if one of the crucial ones get filled
        if 22 in mtext_dict:
            if mtext_dict[22] != 0:
                print("Remark:")
                print(mtext_dict[22])
        if 23 in mtext_dict:
            if mtext_dict[23] != 0:
                print("Remark:")
                print(mtext_dict[23])
    
    new_dict = {}
    for key, name in mtext_dict.items():
        try:
            new_dict[mtext_expl[int(key)]] = name
        except:
            new_dict[int(key)] = name
    
    return new_dict

