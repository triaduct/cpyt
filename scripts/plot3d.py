# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 11:41:58 2021

@author: kduffy
"""

#%% Install Dependencies
import pandas as pd, numpy as np
import time, os, sys
from readGEF import GEF
from tqdm import tqdm


import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot, iplot


phdDir = "\\".join(os.getcwd().split("\\")[0:7]) + "\\"   # Root directory i.e. the PhD folder

pd.options.mode.chained_assignment = None  # default='warn'
sys.path.insert(1, phdDir + "CC_Work\\01_generic\\")
from graphing import saveGraph,getLine_color

start_time = time.time() 

imgFolder = phdDir + "BB_Data\\90_CPT vis\\" # Where the images will be saved to 

#%%
def cylinder(r, h, a =0, nt=100, nv =50):
    """
    parametrize the cylinder of radius r, height h, base point a
    """
    theta = np.linspace(0, 2*np.pi, nt)
    v = np.linspace(a, a+h, nv )
    theta, v = np.meshgrid(theta, v)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = v
    return x, y, z

def boundary_circle(r, h, nt=100):
    """
    r - boundary circle radius
    h - height above xOy-plane where the circle is included
    returns the circle parameterization
    """
    theta = np.linspace(0, 2*np.pi, nt)
    x= r*np.cos(theta)
    y = r*np.sin(theta)
    z = h*np.ones(theta.shape)
    return x, y, z

def cuboid(r, h, a =0, nt=100, nv =50):
    """
    parametrize the cylinder of radius r, height h, base point a
    """
    theta = np.linspace(0, 2*np.pi, nt)
    v = np.linspace(a, a+h, nv )
    theta, v = np.meshgrid(theta, v)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = v
    return x, y, z

def boundary_circle(r, h, nt=100):
    """
    r - boundary circle radius
    h - height above xOy-plane where the circle is included
    returns the circle parameterization
    """
    theta = np.linspace(0, 2*np.pi, nt)
    x= r*np.cos(theta)
    y = r*np.sin(theta)
    z = h*np.ones(theta.shape)
    return x, y, z

#%%

def plot3d(dataFolder,imgFolder,origin_x=np.nan,origin_y=np.nan, objects=False,rotation_xy=0):
    """
    :rotation_xy:   Rotation of the x-y axis of the CPT relative to a North-East axis (i.e. 
                    typical of global coordinate systems). Isn't typically provided in a GEF
                    file so needs to be manually input.
    """

    fig = go.Figure()
    
    for file in tqdm(os.listdir(dataFolder)):
        if file.split(".")[-1].lower() == "gef":
            g=GEF()
            g.readFile(dataFolder + file)
            df = g.asDataFrame()
            
            if g.corr_z == False:
                raise ValueError("CPT depths have not been corrected. Need to do more adjustment to readGEF")
            
            if g.cpt in ["CPTU2_4","CPTU2_5","CPTU2_6","CPTU2_7","CPTU2_8","CPTU3_4"]:
                colorscale = "teal_r"
                note = " (post install)"
            else:
                colorscale = "reds_r"
                note = ""
                
            df["delta_z"] = abs(df.z.diff())
            df["pen"] = df.delta_z.cumsum()
            df["x"] = np.nan
            df["y"] = np.nan
            df.x.iloc[0] = g.x
            df.y.iloc[0] = g.y
            
            for ix, row in df.iterrows():
                if ix == 0:
                    continue
                delta_x = row.delta_z*np.tan(np.deg2rad(row.incl_x))
                df.x.iloc[ix] = df.x.iloc[ix-1] + delta_x
                delta_y = row.delta_z*np.tan(np.deg2rad(row.incl_y))
                df.y.iloc[ix] = df.y.iloc[ix-1] + delta_y
            
        fig.add_trace(
            go.Scatter3d(
                x=df.x - origin_x,
                y=df.y - origin_y,
                z=df.z,
                customdata=df.qc,
                mode='markers',
                name=g.cpt + note,
                marker=dict(
                    size=2,
                    color=df.qc,                # set color to an array/list of desired values
                    colorscale=colorscale,   # choose a colorscale
                    opacity=0.5,
                    symbol="square"),
                hovertemplate = r"x = %{x:.2f} m<br>"+
                                 "y = %{y:.2f} m<br>"+
                                 "z = %{z:.2f} m<br>"+
                                 "q<sub>c</sub> = %{customdata:.2f} MPa"
                # color=df.qc
            ))
        
    #%% Plot other objects
    if objects:
        # Piles
        pile_df = pd.read_excel(phdDir + "BB_Data\\23_Delft Pile Test\\test_proc\\pile_details_post install.xlsx",
                              sheet_name="depths",index_col=0)
    
            
        # for pile in ["P1","P2","P3"]:
        #     fig.add_trace(go.Scatter3d(
        #         x=[pile_df[pile].x-origin_x, pile_df[pile].x-origin_x],
        #         y=[pile_df[pile].y-origin_y, pile_df[pile].y-origin_y],
        #         z=[pile_df[pile].pile_top, pile_df[pile].pile_depth],
        #         mode='lines',
        #         name=pile,
        #         line=dict(color="darkgray",width=8)
        #     ))    
        
        for pile in ["P1","P2","P3"]:
            fig.add_trace(go.Scatter3d(
                x=[pile_df[pile].x-origin_x, pile_df[pile].x-origin_x],
                y=[pile_df[pile].y-origin_y, pile_df[pile].y-origin_y],
                z=[pile_df[pile].pile_top, pile_df[pile].pile_depth],
                mode='lines',
                name=pile,
                line=dict(color="darkgray",width=8)
            ))  
        
        # Borehole
        # fig.add_trace(go.Scatter3d(
        # x=[85980.7-origin_x,85980.7-origin_x],
        # y=[444500.1-origin_y,444500.1-origin_y],
        # z=[-2.20,-27],
        # mode='lines',
        # name="Borehole B1",
        # line=dict(color="green",width=8)))
        
        # Borehole 2
        r1 = 0.125
        h1 = -2.20+27
        a1 = -27
        x1, y1, z1 = cylinder(r1, h1, a1)
        
        colorscale = [[0, 'green'],
                      [1, 'green']]
        
        fig.add_trace(go.Surface(x=x1, y=y1, z=z1,
                          colorscale = colorscale,
                          showscale=False,name="Borehole B1",
                          opacity=0.5))
        xb_low, yb_low, zb_low = boundary_circle(r1, h=a1)
        xb_up, yb_up, zb_up = boundary_circle(r1, h=a1+h1)
        
        fig.add_trace(go.Scatter3d(x = xb_low.tolist()+[None]+xb_up.tolist(),
                            y = yb_low.tolist()+[None]+yb_up.tolist(),
                            z = zb_low.tolist()+[None]+zb_up.tolist(),
                            mode ='lines',
                            line = dict(color='green', width=2),
                            opacity =0.55, showlegend=False))

    #%%
    fig.update_layout(xaxis = dict(tickformat="digits"),title="Deltares: CPT Probe Investigation")

    plot(fig, filename= imgFolder + "cpts 3d" + ".html", auto_open=True)
        #%%
if __name__ == "__main__":
    dataFolder = phdDir + "BB_Data\\23_Delft Pile Test\\cpts\\GEF\\" # Location of .gef files
    # dataFolder = phdDir + "BB_Data\\99_Other\\CPTs Deltares\\" # Location of .gef files
    # dataFolder = phdDir + "BB_Data\\26_ts3\\cpts_vervallen\\proc\\"
    plot3d(dataFolder,origin_x = 86440, origin_y = 444565, imgFolder=dataFolder,
           rotation_xy = 5)

    """
    -------------------------------------------------
    """
    print("13/09/21: See BB_Data/99_Other/CPTs Deltares for more correct script")
    """
    -------------------------------------------------
    """