# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 22:43:39 2018
By: Kevin Duffy | kevinjamesduffy@gmail.com

Part of the code has been adapted from Rob van Putten, see 
https://www.linkedin.com/pulse/power-python-rob-van-putten/
"""

#%% IMPORT DEPENDENCIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

#%%
class CPT:
    def __init__(self):
        self.filename = ''
        self.name = ''              # Name of the CPT
        self.date = ''              # Date of CPT execution
        self.x0 = 0.
        self.y0 = 0.
        self.z0 = 0.                # z-coordinate/elevation w.r.t an ordnance datum
        
        # Cone geometry
        self.cone_type = ""         # Cone type
        self.tip_area = 0.          # Cone tip area [mm2]  
        self.dia = 0.               # Cone diameter [mm]
        self.a = 0.                 # Cone area ratio
        
        # Measurement Variables
        self.z = []                 # Depth w.r.t ordnance datum
        self.pen = []               # Depth w.r.t surface
        self.corr_z = False         # Check if depth is corrected
        self.qc = []                # Cone resistance           [MPa]
        self.fs = []                # Sleeve resistance         [MPa]
        self.Rf = []                # Friction ratio            [%]
        self.u1 = []                # Pore water pressure u2    [MPa]
        self.u2 = []                # Pore water pressure u2    [MPa]
        self.u3 = []                # Pore water pressure u2    [MPa]
        self.time = []              # Time [s]
        self.incl = []              # Absolute inclination of cone [degrees]
        self.incl_x = []            # Inclination of cone in x direction [degrees]
        self.incl_y = []            # Inclination of cone in y direction [degrees]
        self.temperature = []       # Temperature in the ocne

        # Derived variables
        self.rate = []              # Rate of penetration [cm/s]
        
        
        self.units = {}             # Data type units
        self.mtext = {}
        
    #%%
    def readGEF(self, filename):
        self.filename = filename
        print("==  " + filename.split("\\")[-1] + "  =============\n")
        
        GEF_COL_DTYPE = {1: "pen", 2: "qc", 3: "fs", 4: "Rf", 5: "u1", 6: "u2",
                         7: "u3", 8: "incl", 9: "incl_x", 10: "incl_y", 11: "corr_z",
                         12: "time", 21: "incl_x", 22: "incl_y", 129: "temperature"}        # See deltares_gef explanation.pdf for dtypes
        
        col_sep = ";"          # Column separator/delimiter
        rec_sep = "\n"         # Record separator. Separates each line from one another
        # col_void = dict(zip(list(range(1,12)),[9999.0]*11))         # 9999.0 typically used to represent NaN in GEF files
        col_void = {}
        _columns = {}           # Structure: {:dtype:, :column #:}
        mvar = {}               # For storing #MEASUREMENTVAR
        mtext = {}               # For storing #MEASUREMENTTEXT
        
        def _check_header():                
            """
            Function which checks that we have the data needed
            """
            if not 1 in _columns:
                print(f'Error: This GEF is missing a depth column ({self.filename})\n')
                return False
                    
            if not 2 in _columns:
                print(f'Error: This GEF is missing a qc column ({self.filename})\n')
                return False
                    
            if not 3 in _columns:
                print(f'Error: This GEF is missing an fs column ({self.filename})\n')
                return False 
            
            return True
                        
        lines = open(filename, 'r').readlines()
        EOH = False       # Switches to False when we reach the CPT data (End of Header)
        
        for line in lines:            
            if EOH == False:
                # Parse a header line
                keyword, argline, *rest = line.split('=')   # *rest is used if there is an equals sign in the argline
                keyword = keyword.strip()
                argline = argline.strip()
                args = argline.split(',')
                    
                if keyword == '#TESTID':
                    self.cpt = args[0].strip()
                elif keyword == '#XYID':
                    self.x0 = float(args[1].strip())
                    self.y0 = float(args[2].strip())
                elif keyword == '#ZID':
                    self.z0 = float(args[1].strip())
                elif keyword == '#COLUMNINFO':
                    column = int(args[0])                   
                    dtype = int(args[-1].strip())           
                    if dtype == 11:    # override depth with corrected depth
                        dtype = 1
                        self.corr_z = True
                        print("Elevations corrected by default elevation in GEF file")
                    _columns[dtype] = column - 1       # Actual column ID where the info can be found
                    self.units[dtype] = args[1].split("(")[0].strip(" ")
                elif keyword == "#COLUMNSEPARATOR":
                    col_sep = args[-1]
                elif keyword == "#RECORDSEPARATOR":
                    rec_sep = args[-1]
                # elif keyword == "#DATAFORMAT":
                #     if args[-1] == "ASCII":
                #         rec_sep = args[-1]
                elif keyword == "#COLUMNVOID":
                    col_void[int(args[0])-1] = float(args[-1])    # args[0] = column number; args[-1] = outlier number. -1 as indexing in .gef starts at 1
                elif keyword == "#MEASUREMENTVAR":
                    try:
                        mvar[int(args[0])] = float(args[1].strip(" "))
                    except:
                        mvar[int(args[0])] = str(args[1])
                elif keyword == "#MEASUREMENTTEXT":
                    mtext[int(args[0])] = str(args[1])
                elif keyword == "#STARTDATE":
                    self.date = datetime(int(args[0]),int(args[1]),int(args[2]))
            
            else:
                # Parse a data line
                line=line.strip("\n")   # Some GEF files don't specify RECORDSEPARATOR but put \n in its place
                line=line.strip(rec_sep)
                args = line.split(col_sep)
                if len(args) <= 1:          # 
                    args = line.split(" ") 
            
                args = list(filter(None, args))
                for i,value in enumerate(args):
                    value = float(value)
                    if len(col_void) > 0:       # if #COLUMNVOID is specified in GEF file
                        if float(value) == col_void[i]:
                            value = np.nan
                    args[i] = value
                        
                for dtype, col_id in _columns.items():
                    self.__dict__[GEF_COL_DTYPE[dtype]].append(float(args[col_id]))     # Appends to class attribute based on dtype
 

            if line.find('#EOH') > -1:
                if _check_header():
                    EOH = True
                else:
                    return 
            
        # Get cone diameter
        if 1 in mvar:
            self.tip_area = mvar[1]
            if (self.tip_area > 1020) | (self.tip_area < 980):
                self.dia = 35.7   # [mm]
            if (self.tip_area > 1510) | (self.tip_area < 1490):
                self.dia = 43.7   # [mm]            
        
        # Data processing
        ## Calculate friction ratio
        if self.Rf == []:
            self.Rf = list(np.array(self.fs)/np.array(self.qc)*100)         # [%]
        self.z = [self.z0 - pen for pen in self.pen]        # Calculate penetration relative to NAP
        
        ## Fill class attributes with NaN if blank
        for prop in GEF_COL_DTYPE.values():
            if self.__dict__[prop] == []:
                self.__dict__[prop] = np.empty(len(self.z))            # self.z will always be filled
                
        ## Correct for inclination
        if self.corr_z == False:        # If this hasn't been calculated by GEF software already
            # If total elevation present
            """Need to double-check this, causing issue with "CPT000000155824_IMBRO_A.gef"""
            # if 8 in _columns:
            #     uncorr_z = self.dz
            #     uncorr_z_diff = np.diff(uncorr_z)
            #     uncorr_z_diff = np.insert(uncorr_z_diff,0,0)
            #     dz_corr = np.cumsum(np.cos(np.deg2rad(self.incl))*uncorr_z_diff)
            #     self.dz= self.z + dz_corr
                
            #     self.corr_z = True
            #     # print("Elevation corrected for using the resultant inclination")
            #     # delta_y = row.delta_z*np.tan(np.deg2rad(row.incl_y))
                
            # If incl_x, incl_y present
            if (21 in _columns) & (22 in _columns):
                resultant = np.sqrt(np.array(self.incl_x)**2 + np.array(self.incl_y)**2)
                self.incl = resultant
                
                uncorr_z = self.z
                uncorr_z_diff = np.diff(uncorr_z)
                uncorr_z_diff = np.insert(uncorr_z_diff,0,0)
                dz_corr = np.cumsum(np.cos(np.deg2rad(resultant))*uncorr_z_diff)
                self.z= self.z0 + dz_corr
                
                # raise ValueError(" Inclination needs to be double-checked")
                print("Elevation corrected for using the inclination in the X and Y directions")
                self.corr_z = True
            else:
                print("Inclination not corrected for")
            pass
        
        # DATA CHECKS
        if 12 in mvar:
            if isinstance(mvar[12],float):
                mvar[12] = int(mvar[12])
                pen_type = {0:"electronic penetration test",
                            1:"mechanical discontinuous",
                            2:"mechanical continuous",
                            "-": "electronic penetration test"}
                try:
                    mvar[12] = pen_type[mvar[12]]
                except:
                    mvar[12] = "ID not recognised: " + str(mvar[12])
                    
        if 1 in mvar:
            if (mvar[1] > 1020) | (mvar[1] < 980):
                if (mvar[1] > 1510) | (mvar[1] < 1490):
                    print("A non-standard CPT cone has been used (" + str(mvar[1]) + " mm2)")
        if 12 in mvar:
            if (mvar[12] == 1) | (mvar[12] == 2):
                print("NOTE: A mechanical CPT was possibly used")
        if 13 in mvar:
            if mvar[13] != 0:
                print("Pre-excavation carried out to " + str(mvar[13]) + "m")
        if 15 in mvar:
            if mvar[13] != 0:
                print("Depth of water above ground surface = " + str(mvar[15]) + "m")

        # Check time
        try:
            if self.date.year < 1980:
                print(f"CPT executed in {self.date}")
        except:
            pass
        
        if max(self.incl) > 5:
            print(f"NOTE: inclinations of up to {max(self.incl):.0f} degrees in CPT\n")
        
        print("==================================")
        
    #%% 
    def readAGS(self, filename):
        """
        07/12/23: To be completed. Kevin has a script for this
        """
        pass
        
    
    #%%       
    def asNumpy(self):
        return np.transpose(np.array([self.z, self.qc, self.fs, self.Rf, self.u2,
                                      self.incl_x,self.incl_y,self.incl,self.pen,self.time]))    
    
    #%%
    def asDataFrame(self):
        a = self.asNumpy()
        df = pd.DataFrame(data=a, columns=["z","qc","fs","Rf","u2","incl_x","incl_y","incl","pen","time"])
        df.dropna(inplace=True,how="all")
        return df
    
    #%%
    def asCSV(self,targetDir):
        df = self.asDataFrame()
        df.to_csv(targetDir + self.name + ".csv")
    
    def asExcel(self,targetDir):
        df = self.asDataFrame()
        df.to_excel(targetDir + self.name + ".xlsx")
    
    #%%
    def plot(self,save=False):    
        df = self.asDataFrame()
        fig, ax = plt.subplots(
                nrows=1, ncols=3, figsize=(7.5,9), 
                gridspec_kw = {'width_ratios':[2.5, 1, 1]}, sharey=True)
        ax[0].plot(df.qc,df.pen)
        ax[1].plot(df.fs,df.pen)
        ax[2].plot(df.Rf,df.pen)
        
        for i in range(3): 
            ax[i].grid()
            ax[i].invert_yaxis()
        ax[0].set_xlim(0,df.qc.max()+2)
        ax[0].set_ylim(df.pen.max(),df.pen.min())
        ax[0].set_xlabel("Cone resistance [MPa]")
        ax[0].set_ylabel("Depth [m]")
        ax[1].set_xlabel("Friction sleeve\nresistance [MPa]")
        ax[1].set_xlim(0,df.fs.max())
        ax[2].set_xlabel("Friction ratio [%]")
        ax[2].set_xlim(0,df.Rf.max())
        
        if save:
            plt.savefig(self.name + ".png")
        
        return fig, ax

#%%
if __name__ == "__main__":
    import os
    dataFolder = "C:\\Users\\kduffy\\Downloads\\folder\\"
    gef_file = "sample-cpt.gef"
    
    for cpt in os.listdir(dataFolder):
        if cpt.split(".")[-1] == "GEF":
            g = CPT()
            g.readGEF(dataFolder + cpt)
            df = g.asDataFrame()
            df.to_csv(dataFolder + cpt.strip(".GEF") + ".csv")
            g.plot("delete")


