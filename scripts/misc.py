# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 22:37:16 2023

@author: kduffy
"""

#%%
def gefToExcel(file, targetDir):
    """
    "Converts" a .gef file to one with the data in an Excel spreadsheet
    """
    g = GEF()
    g.readFile(file)
    df = g.asDataFrame()
    df.to_excel(targetDir + g.cpt + ".xlsx")
    
    print(f"Data from {g.cpt} has been placed in an Excel file")
    
#%%
def getCoords(dataFolder):
    """
    Gets the x,y,z coordinates of all GEF files in a specified dataFolder and
    returns a .csv file of these coordinates.
    """
    cpt = []
    x = []
    y = []
    z = []
    
    for file in os.listdir(dataFolder):
        if file.split(".")[-1].lower() == "gef":
            g=GEF()
            g.readFile(dataFolder + file)
            cpt.append(g.cpt)
            x.append(g.x)
            y.append(g.y)
            z.append(g.z)
    
    df = pd.DataFrame({"cpt": cpt, "x": x, "y": y, "z": z})
    df.to_csv("cpt locations.csv")
    
    return df