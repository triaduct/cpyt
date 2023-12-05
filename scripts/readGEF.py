# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 22:43:39 2018

Adapted from the code by Rob van Putten (breinbaasnl@gmail.com)
    - See https://www.linkedin.com/pulse/power-python-rob-van-putten/
Adapted by: Kevin Duffy | kevinjamesduffy@gmail.com

The original code from Rob van Putten as been adapted to fit different formats
as well as enhanced to incorporate extra features. The original code is fantastic
and written in such a proper fashion that I could definitely not replicate. 
Worth checking out the code along with other articles he has published on his 
LinkedIn.

This script uses the corrected depth i.e. the depth after taking into consideration
the inclination of the cone. The returned depths are then given relative to the
ordnance datum.

All CPT units are in mNAP or MPa, where applicable
"""
#%% IMPORT DEPENDENCIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Custom libraries
from mvar import mvar
from mtext import mtext
from unit_check import unit_check

phdDir = "\\".join(os.getcwd().split("\\")[0:7]) + "\\"   # Root directory i.e. the PhD folder

#%%
class GEF:
    def __init__(self):
        self._columns = {}
        
        self.filename = ''
        self.cpt = ''
        self.x = 0.
        self.y = 0.
        self.z = 0.                 # z-coordinate/elevation w.r.t an ordnance datum
        self.dz = []                # Depth w.r.t ordnance datum
        self.pen = []               # Depth w.r.t surface
        self.corr_z = False         # Check if depth is corrected
        self.qc = []                # Cone resistance           [MPa]
        self.fs = []                # Sleeve resistance         [MPa]
        self.Rf = []                # Friction ratio            [%]
        self.u2 = []                # Pore water pressure u2    [MPa]
        self.a = 0.8                # Cone area ratio
        self.time = []              # Time [s]
        self.rate = []              # Rate of penetration [cm/s]
        self.incl = []              # Absolute inclination of cone [degrees]
        self.incl_x = []            # Inclination of cone in x direction [degrees]
        self.incl_y = []            # Inclination of cone in y direction [degrees]
        self.col_sep = ";"          # Column separator/delimiter
        self.rec_sep = "\n"          # Record separator
        self.units = {}             # Data type units
        self.mvar = {}
        self.mtext = {}
        self.date = 0.
        self.col_void = {0: 9999.0, 1: 9999.0, 2: 9999.0, 3: 9999.0,
                         4: 9999.0, 5: 9999.0, 6: 9999.0, 7: 9999.0,
                         8: 9999.0, 9: 9999.0, 10: 9999.0, 11: 9999.0}     # For outlier numbers. With defaults
        
    #%%
    def readFile(self, filename):
        print("==================================")
        print("Reading " + filename.split("\\")[-1] + "...")
        print("---\n")
        
        self.filename = filename
        lines = open(filename, 'r').readlines()
        reading_header = True
        
        for line in lines:            
            if reading_header == True:
                self._parseHeaderLine(line)
            else:
                self._parseDataLine(line)
                    
            if line.find('#EOH') > -1:
                if self._check_header():
                    reading_header = False
                else:
                    return 
        self.mvar = mvar(self.mvar)                         # Check measurement variables & label appropriately
        self.mtext = mtext(self.mtext)                      # Check measurement texts & label appropriately
        unit_check(self.units)                              # Check units
        
        self.dz = [self.z - pen for pen in self.pen]        # Calculate penetration relative to NAP
        
        # Check time
        try:
            if self.date.year < 1980:
                print(f"CPT executed in {self.date}")
        except:
            pass
        
        # Correct for inclination
        if self.corr_z == False:        # If this hasn't been calculated by GEF software already
            # If total elevation present
            """Need to double-check this, causing issue with "CPT000000155824_IMBRO_A.gef"""
            # if 8 in self._columns:
            #     uncorr_z = self.dz
            #     uncorr_z_diff = np.diff(uncorr_z)
            #     uncorr_z_diff = np.insert(uncorr_z_diff,0,0)
            #     dz_corr = np.cumsum(np.cos(np.deg2rad(self.incl))*uncorr_z_diff)
            #     self.dz= self.z + dz_corr
                
            #     self.corr_z = True
            #     # print("Elevation corrected for using the resultant inclination")
            #     # delta_y = row.delta_z*np.tan(np.deg2rad(row.incl_y))
                
            # If incl_x, incl_y present
            if (21 in self._columns) & (22 in self._columns):
                resultant = np.sqrt(np.array(self.incl_x)**2 + np.array(self.incl_y)**2)
                self.incl = resultant
                
                uncorr_z = self.dz
                uncorr_z_diff = np.diff(uncorr_z)
                uncorr_z_diff = np.insert(uncorr_z_diff,0,0)
                dz_corr = np.cumsum(np.cos(np.deg2rad(resultant))*uncorr_z_diff)
                self.dz= self.z + dz_corr
                
                # raise ValueError(" Inclination needs to be double-checked")
                print("Elevation corrected for using the inclination in the X and Y directions")
                self.corr_z = True
            else:
                print("Inclination not corrected for")
            pass
        
        # Print warning messages
        if max(self.incl) > 5:
            print(f"NOTE: inclinations of up to {max(self.incl):.0f} degrees in CPT\n")
        print("\n==================================")
        
    #%%   
    def _check_header(self):                
        """
        Function which checks that we have the data needed
        """
        if not 1 in self._columns:
            print(f'Error: This GEF is missing a depth column ({self.filename})\n')
            return False
                
        if not 2 in self._columns:
            print(f'Error: This GEF is missing a qc column ({self.filename})\n')
            return False
                
        if not 3 in self._columns:
            print(f'Error: This GEF is missing an fs column ({self.filename})\n')
            return False 
        
        return True
    
    #%%
    def _parseHeaderLine(self, line):
        keyword, argline, *rest = line.split('=')   # *rest is used if there is an equals sign in the argline
        keyword = keyword.strip()
        argline = argline.strip()
        args = argline.split(',')
            
        if keyword == '#TESTID':
            self.cpt = args[0].strip()
        elif keyword == '#XYID':
            self.x = float(args[1].strip())
            self.y = float(args[2].strip())
        elif keyword == '#ZID':
            self.z = float(args[1].strip())
        elif keyword == '#COLUMNINFO':
            column = int(args[0])                   
            dtype = int(args[-1].strip())           # See deltares_gef explanation.pdf for dtypes
            if dtype == 11:                         # override depth with corrected depth
                dtype = 1
                self.corr_z = True
                print("Elevations corrected by default elevation in GEF file")
            self._columns[dtype] = column - 1       # Actual column ID where the info can be found
            self.units[dtype] = args[1].split("(")[0].strip(" ")
        elif keyword == "#COLUMNSEPARATOR":
            self.col_sep = args[-1]
        elif keyword == "#RECORDSEPARATOR":
            self.rec_sep = args[-1]
        elif keyword == "#COLUMNVOID":
            self.col_void[int(args[0])-1] = float(args[-1])    # args[0] = column number; args[-1] = outlier number. -1 as indexing in .gef starts at 1
        elif keyword == "#MEASUREMENTVAR":
            try:
                self.mvar[int(args[0])] = float(args[1].strip(" "))
            except:
                self.mvar[int(args[0])] = str(args[1])
        elif keyword == "#MEASUREMENTTEXT":
            self.mtext[int(args[0])] = str(args[1])
        elif keyword == "#STARTDATE":
            self.date = datetime(int(args[0]),int(args[1]),int(args[2]))
            
    #%%      
    def _parseDataLine(self, line):
        line=line.strip("\n")   # Some GEF files don't specify RECORDSEPARATOR but put \n in its place
        line=line.strip(self.rec_sep)
        
        # Check what separator is required:
        args = line.split(self.col_sep)
    
        if len(args) == 1:          # i.e. if it has failed (sometimes #COLUMNSEPARATOR not specified)
            args = line.split(" ")
            if len(args) == 0:
                raise ValueError("The code cannot account for whatever separator is being used in the GEF file")
        
        args = list(filter(None, args))
        args = [np.nan if float(value) == self.col_void[i] else value for i,value in enumerate(args)]    # replace zero values
        
        
        pen = float(args[self._columns[1]]) 
        qc = float(args[self._columns[2]])
        fs = float(args[self._columns[3]])
        
        # Append information
        self.pen.append(pen)
        self.qc.append(qc)
        self.fs.append(fs)
        
        # Calculate friction ratio
        if qc <= 0.:
            self.Rf.append(np.nan)
        else:
            Rf = (fs / qc) * 100
            if Rf > 10: 
                Rf = np.nan    # For ridiculous Rf vlaues
            self.Rf.append(Rf)
        
        ## Add in "extra" information
        if 6 in self._columns:
            u2 = float(args[self._columns[6]])
            self.u2.append(u2)
        else:
            self.u2.append(np.nan)
            
        if 8 in self._columns:
            incl = float(args[self._columns[8]])
            self.incl.append(incl)
        else:
            self.incl.append(np.nan)
            
        if 12 in self._columns:
            time = float(args[self._columns[12]])
            self.time.append(time)
            
            self.rate = (np.diff(self.pen)*100)/np.diff(self.time)
            self.rate = np.insert(self.rate,0,0)
        else:
            self.time.append(np.nan)
            self.rate.append(np.nan)
            
        if 21 in self._columns:
            incl_x = float(args[self._columns[21]])
            self.incl_x.append(incl_x)
        else:
            self.incl_x.append(np.nan)
            
        if 22 in self._columns:
            incl_y = float(args[self._columns[22]])
            self.incl_y.append(incl_y)
        else:
            self.incl_y.append(np.nan)
            
    #%%       
    def asNumpy(self):
        return np.transpose(np.array([self.dz, self.qc, self.fs, self.Rf, self.u2,
                                      self.incl_x,self.incl_y,self.incl,self.pen,self.time,self.rate]))    
    
    #%%
    def asDataFrame(self):
        a = self.asNumpy()
        df = pd.DataFrame(data=a, columns=["z","qc","fs","Rf","u2","incl_x","incl_y","incl","pen","time","rate"])
        df.dropna(inplace=True,how="all")
        return df
    
    #%%
    def plot(self, filename, sample_depth = 0):    
        df = self.asDataFrame()
        fig, ax = plt.subplots(
                nrows=1, ncols=3, figsize=(10,12), 
                gridspec_kw = {'width_ratios':[3, 1, 1]}, sharey=True)
        ax[0].plot(df.qc,df.z, label=r'$q_c$ [MPa]')
        ax[0].legend()
        ax[1].plot(df.fs,df.z, label=r'$f_s$ [MPa]',c='y')
        ax[1].legend()
        ax[2].plot(df.Rf,df.z, label=r'$R_f$ [%]',c='g')
        ax[2].legend()
        
        for i in range(3): 
            ax[i].grid()
            ax[i].invert_yaxis()
            if sample_depth != 0:
                ax[i].axhline(y=sample_depth,c='r', linewidth = 1)
        plt.savefig(filename.strip(".gef"))
        
        return fig, ax

#%%
if __name__ == "__main__":
    pass


