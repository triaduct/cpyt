# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 13:00:32 2021

@author: kduffy
"""

import pandas as pd,numpy as np
from readGEF import GEF
import os,sys

phdDir = "\\".join(os.getcwd().split("\\")[0:7]) + "\\"   # Root directory i.e. the PhD folder

#%%
def resampleCpt(df,spacing=0.05):
    """
    Function used to (artificially) increase the resolution of the fibre 
    readings (or thermistor readings) so other functions can be performed on them.
    
    Resolution increased to 1cm through linear interpolation
    """
    df.z = np.round(df.z,2)
    z_min = df.z.min()
    z_max = df.z.max()
    df = df.set_index("z")
    df = df.reindex(np.round(np.arange(z_min, z_max,spacing),2))
    df = df.interpolate(method="index")
    df = df.sort_index(ascending=False)
    df = df.reset_index()
    return df

#%%
if __name__ == "__main__":
    dataDir = phdDir + "BB_Data\\60_Rdam CPTs_with u2\\"    
    file = "CPT000000151213_IMBRO_A.gef"

    g=GEF()
    g.readFile(dataDir + file)
    df = g.asDataFrame()
    
    df=resampleCpt(df)