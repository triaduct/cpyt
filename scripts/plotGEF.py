# -*- coding: utf-8 -*-
"""


Requires readGEF.py

Created on Mon Dec 24 21:17:04 2018
@author: Kevin Duffy (kevinjamesduffy@gmail.com)
"""

#%% Install Dependencies
import pandas as pd, numpy as np
from matplotlib import pyplot as plt
import time, os, sys
from readGEF import GEF
import add_features
from tqdm import tqdm
import matplotlib.colors as col
from mpl_toolkits.axes_grid1 import make_axes_locatable

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import plot, iplot

phdDir = "\\".join(os.getcwd().split("\\")[0:7]) + "\\"   # Root directory i.e. the PhD folder

pd.options.mode.chained_assignment = None  # default='warn'
sys.path.insert(1, phdDir + "CC_Work\\01_generic\\")
from graphing import saveGraph,getLine_color

start_time = time.time() 

dataFolder = phdDir + "BB_Data\\40_Terneuzen Pile Test\\1_GEF files\\" # Location of .gef files
imgFolder = phdDir + "BB_Data\\90_CPT vis\\" # Where the images will be saved to 

#%%
def plotGEF(dataFolder, cpt_list1, style=None,  
            cpt_list2=[], list1_name = "List1", list2_name = "List2", with_Isbt=False,
            average=False, depth_indicator=None,):
    """
    Main plotter for GEF files using matplotlib 
    """    
    def _refineLabels(ax):
        """Used in plotting to make it easier to see labels and prevent overlap"""
        for label in ax.xaxis.get_ticklabels()[0::2]:
            label.set_visible(False)
        return ax
    
    if style != None:
        plt.style.use(phdDir + "CC_Work\\01_generic\\stylelib\\" + style + ".mplstyle")
        
    from cycler import cycler
    import matplotlib as mpl
    
    mpl.rcParams['axes.prop_cycle'] = cycler(color=["blue","green","black","red","grey","brown",])
    fig, ax = plt.subplots(nrows=1, ncols=4, sharey = True,
                       gridspec_kw = {"width_ratios": [2, 1, 1, 1]})
    exist_u2 = False    # Variable which tracks to see if there is pore pressure
    
    for cpt in cpt_list1 + cpt_list2: 
        for fileName in os.listdir(dataFolder):
            if (fileName.split(".gef")[0] == cpt) | (fileName.split(".GEF")[0] == cpt):
                g = GEF()
                g.readFile(dataFolder + fileName)
                df = g.asDataFrame()
                
                if df.u2.isnull().all() == False:
                    exist_u2 = True
                    
                # # Process colours
                # if cpt in cpt_list1:
                #     col = "darkgray"
                # elif cpt in cpt_list2:
                #     # col_dict = {"CPTU2_4": "red","CPTU2_5": "green","CPTU2_6":"orange","CPTU2_7":"black","CPTU2_8": "blue"}
                #     col = "red"

                ax[0].plot(df.qc, df.z, label=g.cpt)
                ax[0].set_xlabel(r"Cone resistance $q_c$ [MPa]")
                ax[0].set_ylabel('Depth [mNAP]')
                ax[1].plot(df.fs*1000,df.z,label=g.cpt) 
                ax[1].set_xlabel("Friction sleeve\n" + r"resistance $f_s$ [kPa]")
                # ax[1] = _refineLabels(ax[1])
                ax[2].plot(df.Rf,df.z, label=g.cpt)
                ax[2].set_xlabel("Friction\n" + "ratio $R_f$ [%]")
                ax[3].plot(df.u2*1000,df.z, label=g.cpt)
                ax[3].set_xlabel("Pore water\n" "pressure $u_2$ [kPa]")
                
    
    for i in range(4):
        if depth_indicator != None:
            ax[i].axhline(y=depth_indicator, c="r", linewidth=1, linestyle = "--")
            ax[0].text(0.5, depth_indicator+0.2, "Pile tip", color="r")
        ax[i].set_xlim(0)
        ax[i].set_ylim(-35, 0)
        
    # # Axis limits
    ax[0].set_xlim(0,30)
    ax[1].set_xlim(0,100)
    ax[2].set_xlim(0,10)
    # ax[3].set_xlim(0,300)
    
    ax[0].legend(loc="upper right")
    if exist_u2 == False:
        fig.delaxes(ax[3])
        
    return fig

#%%
def plotGEF_plus(dataFolder, cpt_list1, style=None,  
            cpt_list2=[], list1_name = "List1", list2_name = "List2", with_Isbt=False,
            average=False, depth_indicator=None,):
    """
    Main plotter for GEF files using matplotlib 
    """    
    def _refineLabels(ax):
        """Used in plotting to make it easier to see labels and prevent overlap"""
        for label in ax.xaxis.get_ticklabels()[0::2]:
            label.set_visible(False)
        return ax
    
    if style != None:
        plt.style.use(phdDir + "CC_Work\\01_generic\\stylelib\\" + style + ".mplstyle")
        
    from cycler import cycler
    import matplotlib as mpl
    
    mpl.rcParams['axes.prop_cycle'] = cycler(color=["blue","green","black","red","grey","brown",])
    fig, ax = plt.subplots(nrows=1, ncols=6, sharey = True,
                       gridspec_kw = {"width_ratios": [2, 1, 1, 1, 1,1]})
    exist_u2 = False    # Variable which tracks to see if there is pore pressure
    exist_incl = False    # Variable which tracks to see if there is inclination
    
    for cpt in cpt_list1 + cpt_list2: 
        for fileName in os.listdir(dataFolder):
            if (fileName.split(".gef")[0] == cpt) | (fileName.split(".GEF")[0] == cpt):
                g = GEF()
                g.readFile(dataFolder + fileName)
                df = g.asDataFrame()
                
                if df.u2.isnull().all() == False:
                    exist_u2 = True
                if df.incl.isnull().all() == False:
                    exist_incl = True
                    
                # # Process colours
                # if cpt in cpt_list1:
                #     col = "darkgray"
                # elif cpt in cpt_list2:
                #     # col_dict = {"CPTU2_4": "red","CPTU2_5": "green","CPTU2_6":"orange","CPTU2_7":"black","CPTU2_8": "blue"}
                #     col = "red"

                ax[0].plot(df.qc, df.z, label=g.cpt)
                ax[0].set_xlabel(r"$q_c$ [MPa]")
                ax[0].set_ylabel('Depth [mNAP]')
                ax[1].plot(df.fs*1000,df.z,label=g.cpt) 
                ax[1].set_xlabel("$f_s$ [kPa]")
                # ax[1] = _refineLabels(ax[1])
                ax[2].plot(df.Rf,df.z, label=g.cpt)
                ax[2].set_xlabel("$R_f$ [%]")
                ax[3].plot(df.u2*1000,df.z, label=g.cpt)
                ax[3].set_xlabel("$u_2$ [kPa]")
                ax[4].plot(df.incl,df.z, label=g.cpt)
                ax[4].set_xlabel("Incl. [deg]")
                ax[5].plot(df.rate,df.z, label=g.cpt)
                ax[5].set_xlabel("Rate [cm/s]")
                
    
    for i in range(4):
        if depth_indicator != None:
            ax[i].axhline(y=depth_indicator, c="r", linewidth=1, linestyle = "--")
            ax[0].text(1, depth_indicator+0.1, "Pile tip", color="r")
        ax[i].set_xlim(0)
        ax[i].set_ylim(-35, 0)
        
    # Axis limits
    ax[0].set_xlim(0,30)
    ax[1].set_xlim(0,100)
    ax[2].set_xlim(0,10)
    ax[3].set_xlim(0,300)
    ax[5].set_xlim(0,3)
    
    ax[0].legend(loc="upper right")
    if exist_u2 == False:
        fig.delaxes(ax[3])
    if exist_incl == False:
        fig.delaxes(ax[4])
        
    return fig


#%%
def plotlyGEF(dataFolder, cpts,fig_title=None):
    """
    Main plotter for GEF files using matplotlib 
    """    
    fig = make_subplots(rows=1,cols=4, shared_yaxes=True)
    if fig_title == None:
        fig_title = " ".join(cpts)
        
    exist_u2 = False    # Variable which tracks to see if there is pore pressure
    
    for cpt in cpts: 
        for fileName in os.listdir(dataFolder):
            if (fileName.split(".gef")[0] == cpt) | (fileName.split(".GEF")[0] == cpt):
                g = GEF()
                g.readFile(dataFolder + fileName)
                df = g.asDataFrame()
                
                if df.u2.isnull().all() == False:
                    exist_u2 = True
                    
                fig.add_trace(go.Scatter(x=df.qc,y=df.z, mode='lines', name=g.cpt,
                                         hovertemplate =
                                         r"<i>q<sub>c</i> = %{x:.2f} MPa<br>"+
                                         "<i>z</i> = %{y:.2f} mNAP<br>"),
                                          row=1,col=1)
                fig.add_trace(go.Scatter(x=df.fs,y=df.z, mode='lines', name=g.cpt,
                                         hovertemplate =
                                         r"<i>f<sub>s</i> = %{x:.2f} kPa<br>"+
                                         "<i>z</i> = %{y:.2f} mNAP<br>"),
                                          row=1,col=2)
                fig.add_trace(go.Scatter(x=df.Rf,y=df.z, mode='lines', name=g.cpt,
                                         hovertemplate =
                                         r"<i>R<sub>f</i> = %{x:.1f}<br>"+
                                         "<i>z</i> = %{y:.2f} mNAP<br>"),
                                          row=1,col=3)
                if exist_u2:
                    fig.add_trace(go.Scatter(x=df.u2,y=df.z, mode='lines', name=g.cpt,
                                             hovertemplate =
                                             r"<i>u<sub>2</i> = %{x:.2f} kPa<br>"+
                                             "<i>z</i> = %{y:.2f} mNAP<br>"),
                                              row=1,col=4)
    
    fig.update_xaxes(title_text="Cone resistance <i>q<sub>c</i> [MPa]", range=(0, df.qc.max()), row=1, col=1)
    fig.update_xaxes(title_text="Friction sleeve resistance <i>f<sub>s</i> [kPa]", range=(0, df.fs.max()), row=1, col=2)
    fig.update_xaxes(title_text="Friction ratio <i>R<sub>f</i> [-]", row=1, range=(0, df.Rf.max()), col=3)
    fig.update_xaxes(title_text="Pore pressure <i>u<sub>2</i> [MPa]", row=1, range=(0, df.u2.max()), col=4)
    fig.update_yaxes(title_text="Depth [mNAP]", row=1, col=1)
    
    fig.update_layout(legend_title="<b>CPT</b><br>") 
    
    fig.update_layout(title=fig_title)
    plot(fig, filename= phdDir + "BB_Data\\26_ts3\\" + fig_title + ".html", 
         auto_open=True)

    return fig

#%%
def GEFtoJPG(dataFolder, imgFolder):
    for fileName in tqdm(os.listdir(dataFolder)):
        if fileName.split('.')[-1].lower() != "gef":
            continue
            
        g = GEF()
        g.readFile(dataFolder + fileName)
        df = g.asDataFrame()
        
        # Calculate I_sbt
        df=add_features.Isbt(df)
    
        # Attach dummy variable to I_sbt value
        sbt_bins = [0, 1.31, 2.05, 2.60, 2.95, 3.60, np.inf]
        sbt_names = [7, 6, 5, 4, 3, 2]
        df['sbt'] = pd.cut(df.Isbt, sbt_bins, labels=sbt_names)
        
        # Plot
        fig, ax = plt.subplots(
                nrows=1, ncols=7, figsize=(10,12),
                gridspec_kw = {'width_ratios': [3, 1, 1, 1, 1, 1, 1]}, sharey=True)
        fig.suptitle(g.cpt,y=0.90,fontweight="bold")
        ax[0].plot(df.qc,df.z)
        ax[0].set_xlabel("$q_c$ [MPa]")
        ax[0].set_ylabel('Depth [mNAP]')
        ax[0].invert_yaxis()
        ax[1].plot(df.fs,df.z, color='red') 
        ax[1].set_xlabel(r'$f_s$ [kPa]')
        ax[2].plot(df.Rf,df.z, color='green')
        ax[2].set_xlabel(r'$R_f$ [%]')   
        # ax[3].plot(df.inc_ew,df.z, color='orange')
        # ax[3].set_xlabel('Inclination \n E-W (degrees)')
        # ax[4].plot(df.inc_ns,df.z, color='orange')
        # ax[4].set_xlabel('Inclination \n N-S (degrees)')  
        ax[5].plot(df.Isbt,df.z, color='pink')
        ax[5].set_xlabel(r'$I_{SBT}$')   
        plt.minorticks_on()
        
        for graph in range(6):
            ax[graph].grid(which="minor")
            
        # Plot colours
        soil_colors = ['#ce7e0e', '#047c2c','#9ae050','#e5ffbf', '#ffdfa8', '#f4b342']
        cmap_soil = col.ListedColormap(
                soil_colors[0:len(soil_colors)], 'indexed')
        cluster = np.expand_dims(df.sbt.values,1)
        im = ax[6].imshow(cluster, interpolation='none', aspect='auto', 
                extent=[-5,5, df.z.min(),df.z.max()], cmap=cmap_soil,vmin=0,vmax=5)      
        ax[6].set_xlabel(r'$I_{SBT}$')   
        ax[6].set_xticklabels([])
        
        divider = make_axes_locatable(ax[6])
        cax = divider.append_axes("right", size="20%", pad=0.05)
        cbar=plt.colorbar(im, cax=cax)
        cbar.set_label('Peat & organic          Clay & silty clay           Clayey silt & silty clay     \
        Silty sand & sandy silt           Sand & silty sand                  Sand')
        cbar.set_ticks(range(0,1))
        cbar.set_ticklabels('')
        ax[6].grid(False)
        
        saveGraph(fig, imgFolder, g.cpt)
    return fig, ax

#%% Plot CPT for each pile type
def plotCptsWithinRange(pile, limit, dataFolder, locFolder):
    """
    :limit:     Max. distance between pile and CPT
    """
    pile_db = pd.read_excel(locFolder + "pile_details_post install.xlsx",delimiter=";")
    x_pile = float(pile_db[pile_db.pile == pile].x)
    y_pile = float(pile_db[pile_db.pile == pile].y)
    pile_qcs = pd.DataFrame()       # Fill this dataframe with qc values
#    pile_qcs.z = np.arange(-50,5,0.02)
    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(10,12),sharey=True)

    for fileName in tqdm(os.listdir(dataFolder)):
        if (fileName.split('.')[-1] == "gef") or (fileName.split('.')[-1] == "GEF"):
            g = GEF()
            g.readFile(dataFolder + fileName)
            x_cpt = g.x
            y_cpt = g.y
            dist = np.sqrt((x_cpt-x_pile)**2 + (y_cpt-y_pile)**2)             # Distance between pile and CPT
            
            if dist < limit:
                g = GEF()
                g.readFile(dataFolder + fileName)
                df = g.asDataFrame()
                ax[0].plot(df.qc,df.z,linewidth=1,alpha=0.7, label=g.cpt)

                
                def custom_round(x, base=0.02):                         # round to nearest 0.02 cm to avoid wobbly CPT plots
                    return base * round(float(x)/base)
                df.z = df.z.apply(lambda x: custom_round(x, base=0.02))
                df = df.groupby('z').mean().reset_index()                # Take average of any duplicate depths
                df.set_index("z",inplace=True)
                df = df.sort_values(by="z", ascending = False)
                df = df[["qc"]]
                df.rename(columns={"qc": str(g.cpt + "_qc")},inplace=True)
                pile_qcs = pd.concat([pile_qcs,df],axis=1,sort=False)
                
        else:
            continue 
    fig.suptitle(f"Pile: {pile}; Range: {limit} metres", y=0.9,fontweight="semibold")
    pile_qcs["min"] = pile_qcs.min(axis=1)
    pile_qcs["mean"] = pile_qcs.mean(axis=1)
    pile_qcs["max"] = pile_qcs.max(axis=1)
    pile_qcs.reset_index(inplace=True)

    ax[1].plot(pile_qcs["min"], pile_qcs.z,c='b',label=r"Min. & Max. $q_c$")
    ax[1].plot(pile_qcs["max"], pile_qcs.z,c='b',label="")
    ax[1].plot(pile_qcs["mean"], pile_qcs.z,c='r',label=r"Mean $q_c$")
    plt.fill_betweenx(pile_qcs.z, pile_qcs["min"],pile_qcs["max"], alpha=0.5)
    plt.minorticks_on()
    ax[0].set_ylabel(r"Depth [mNAP]")
    
    for axis in [0,1]:
        ax[axis].set_xlabel(r"Cone resistance $q_c$ [MPa]")
        
        ax[axis].legend()
        ax[0].set_xlim(0,70)
        ax[0].set_ylim(-35,5)
        
    return fig, ax    

#%% Put location coords into database
def makeLocDb(dataFolder):
    df = pd.DataFrame(columns=['cpt','x','y','z'])
    for fileName in tqdm(os.listdir(dataFolder)):
        if fileName.split('.')[-1].lower() != "gef":
            continue
        g = GEF()
        g.readFile(dataFolder + fileName)
 
        df = df.append(pd.Series([g.cpt,g.x,g.y,g.z], index=df.columns), ignore_index=True)
    
    # Change commas to decimals
#    df.x = df.x.replace(',','.')
#    df.y = df.y.replace(',','.')
    
    return df

#%%
def plot_qc(dataFolder, cpt_list1, style=None,  
            cpt_list2=[], list1_name = "List1", list2_name = "List2", with_Isbt=False,
            average=False):
    """
    Main plotter for GEF files using matplotlib 
    """    

    
    if style != None:
        plt.style.use(phdDir + "CC_Work\\01_generic\\stylelib\\" + style + ".mplstyle")
        
    from cycler import cycler
    import matplotlib as mpl
    
    mpl.rcParams['axes.prop_cycle'] = cycler(color=["#BC3C29FF","#0072B5FF","black","#20854EFF"])
    # mpl.rcParams['axes.prop_cycle'] = cycler(color=["red","green","blue","black"])
    fig = plt.figure()
    ax = fig.gca()
    # fontsize=12
    i=0
    for cpt in cpt_list1 + cpt_list2: 
        for fileName in os.listdir(dataFolder):
            if (fileName.split(".gef")[0] == cpt) | (fileName.split(".GEF")[0] == cpt):
                g = GEF()
                g.readFile(dataFolder + fileName)
                df = g.asDataFrame()
                # label_dict = {"P04": "SI1", "P05": "SI2","P07": "SI3","P08": "SI4",
                #               "P01": "DCIS1", "P03": "DCIS2","P06": "DCIS3","P11": "DCIS4"}
                # label_dict = {"CPTU2_0": "Delft","P01": "Maasvlakte"}
                # label = label_dict[g.cpt]
                # label = g.cpt
                
                ax.plot(df.qc, (df.z-df.z.iloc[0])*-1, label=g.cpt)
                i+=1
                ax.set_xlabel(r"Cone resistance $q_c$ [MPa]")
                ax.set_ylabel('Depth [m]')
                
    # # For DFI 
    ax.axhline(37.02,color='#1f77b4',c="red",linestyle="--",alpha=0.7,zorder=99)
    # ax.axhline(37.06,color='#1f77b4',c="blue",linestyle="--")
    ax.axhline(34.06,color='#1f77b4',c="black",linestyle="--",alpha=0.7,zorder=99)
    # ax.text(80, 37.3,"P04 + P05", color="#D00404",alpha=0.8,fontsize=12)
    # ax.text(80, 34.25,"P08", color="black",alpha=0.9,fontsize=12)

                
    ax.set_xlim(0,80)
    ax.set_ylim(30,40)
    # legend = ax.legend(loc="upper right")
    # legend.get_frame().set_linewidth(0.5)
    ax.tick_params(labelbottom=False,labeltop=True)
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_visible(False)

    ax.invert_yaxis()

    return fig

#%% Cross-Correlate
def crossCorr(dataFolder,cpt1,cpt2):
    sys.path.insert(1, phdDir +"\\CC_Work\\")
    from cptCleaner import cptCleaner
    from scipy import signal
    
    for fileName in os.listdir(dataFolder):
        if fileName.split('_')[1] == cpt1 + ".gef" or fileName.split('_')[1] == cpt1 + ".GEF":
            g = GEF()
            g.readFile(dataFolder + fileName)
            df1 = g.asDataFrame()
        elif fileName.split('_')[1] == cpt2 + ".gef" or fileName.split('_')[1] == cpt2 + ".GEF":
            g = GEF()
            g.readFile(dataFolder + fileName)
            df2 = g.asDataFrame()
    
    if df1.z.min() > df2.z.min():
        max_z = df1.z.min()
    else:
        max_z = df2.z.min()
        
    if df1.z.max() < df2.z.max():
        min_z = df1.z.max()
    else:
        min_z = df2.z.max()
    x=1
        
    df1=cptCleaner(df1,min_z,max_z)
    df2=cptCleaner(df2,min_z,max_z)
        
    sig = signal.correlate(df1.qc,df2.qc,mode='same')
    
#%%
#%%
def plotGEF_minimal(dataFolder, cpts,imgFolder):
    """
    For plotting vector functions to be overlaid over picture 
    """    
    from cycler import cycler
    import matplotlib as mpl
    
    plt.style.use(phdDir + "CC_Work\\01_generic\\stylelib\\plotGEF_minimal.mplstyle")
    mpl.rcParams['axes.prop_cycle'] = cycler(color=["blue","green","black","red","grey","brown",])
    fig = plt.figure(figsize=(1,6.375))
    ax = fig.gca()
    
    for cpt in cpts: 
        for fileName in os.listdir(dataFolder):
            if (fileName.split(".gef")[0] == cpt) | (fileName.split(".GEF")[0] == cpt):
                g = GEF()
                g.readFile(dataFolder + fileName)
                df = g.asDataFrame()
                
                print(g.z)
                
                ax.plot(df.qc, df.z-g.z + 1, label=g.cpt)
                # ax.set_xlabel(r"Cone resistance $q_c$ [MPa]")
                # ax.set_ylabel('Depth [mNAP]')
   
    # ax.set_title(g.cpt)
    ax.set_xticks([0,30,60])
    ax.tick_params(labelbottom=False,labeltop=True,labelleft=False)
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_visible(False)
    ax.invert_yaxis()
    
    # ax.set_yal(None)
    ax.set_xlim(0,60)
    ax.set_ylim(-30, 0)
        
    saveGraph(fig, imgFolder, (" ").join(cpts),transparent=True)
    return fig

#%% Export
if __name__ == "__main__":
    locFolder=phdDir + "\\BB_Data\\01_locations\\"
    
    # -------- Data Folders --------
    # dataFolder = phdDir + "BB_Data\\26_ts3\\00_ss\\received BMNED_incorrect incl\\GEF\\"
    dataFolder = phdDir + "BB_Data\\23_Delft Pile Test\\cpts\\GEF\\"
    # dataFolder = phdDir + "BB_Data\\mv2 pile test\\site investigation\\test site_gef files\\"
    # dataFolder = phdDir + "BB_Data\\50_Euromax\\02_SI DINO\\CPT\\"
    # dataFolder = phdDir + "BB_Data\\60_Rdam CPTs_with u2\\"
    # dataFolder = phdDir + "BB_Data\\99_Other\\CPTs Deltares\\"
    # dataFolder = phdDir + "04_Publications\\02_Conferences\\Geotechniekdag\\cpts to plot\\"
    # ------------------------------
    
    # -------- Image Folders --------
    # imgFolder = phdDir + "\\delete\\"
    imgFolder = phdDir + "BB_Data\\23_Delft Pile Test\\cpts\\"
    # imgFolder = phdDir + "BB_Data\\26_ts3\\site investigation\\"
    # imgFolder = phdDir + "CC_Work\\cpt-3d-model\\scripts\\ts3-2d\\"
    # dataFolder = phdDir + "BB_Data\\mv2 pile test\\for npr\\cpts\\"
    # imgFolder = phdDir + "BB_Data\\50_Euromax\\02_SI DINO\\"
    # imgFolder = phdDir + "04_Publications\\02_Conferences\\Geotechniekdag\\cpts to plot\\"
    # imgFolder = phdDir + "BB_Data\\99_Other\\CPTs Deltares\\"
    # ------------------------------

    
    # fig = plot_qc(dataFolder, ["I.001067_P04","I.001067_P05","I.001067_P08"], style="journal",  d
    #         cpt_list2=[], list1_name = "List1", list2_name = "List2", with_Isbt=False,
    #         average=False)
    
    # GEFtoJPG(dataFolder, imgFolder)
    # from getCptData import getCptData
    # fnames = getCptData(61666.53, 443954.9, what_cpt="within_50", 
    #                     cptFolder=dataFolder, wt=1, cpt=None, fname_output=True)
    
    # cpt_list = []
    # for filename in fnames:
    #     g=GEF()
    #     g.readFile(dataFolder+filename+".gef")
    #     cpt_list.append(g.cpt)
    
    pile_to_cpts = {"T1": ["CPT1_4","CPT1_1","CPT1_2","CPT1_3"],
                "T2": ["CPT2_3","CPT3_1","CPT3_2","CPT3_3"],
                "T3": ["CPT4_3","CPT5_1","CPT5_2","CPT5_3"],
                "F1": ["CPT1_3","CPT2_1","CPT2_2","CPT2_3"],
                "F2": ["CPT3_3","CPT4_1","CPT4_2","CPT4_3"],
                "F3": ["CPT5_3","CPT6_1","CPT6_2","CPT6_3"]}
    
    # imgFolder = "C:\\Users\\kduffy\\Downloads\\"
    # for cpt in ['I.001067_SS100','I.001067_SS102','I.001067_SS104','I.001067_SS106','I.001067_SS108',
    #             'I.001067_SS110','I.001067_SS112','I.001067_SS114','I.001067_SS116','I.001067_SS118',]:
    #     plotGEF_minimal(dataFolder, cpts=[cpt],imgFolder=imgFolder)
    # pile = "F1"
    # fnames = pile_to_cpts[pile]
    # fig = plotGEF(dataFolder, fnames, style="report_a4",  
    #         cpt_list2=[], depth_indicator=-23.13,  
    #         list1_name = "list1", list2_name = "List2", with_Isbt=False,
    #         average=False)
    

    
    # pile = "F3"
    plotlyGEF(dataFolder, ["CPTU3_0","CPTU3_1","CPTU3_2","CPTU3_3"],fig_title="P2")
    # saveGraph(fig, imgFolder, pile)
    # plotlyGEF(dataFolder, ["CPT4_3","CPT5_1","CPT5_3"])
    # GEFtoJPG(dataFolder, imgFolder)
