# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 12:04:40 2022

@author: kduffy
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch

def plot_borehole_log(fig, layers, x=32, width=2,hlines=False):
    ax = fig.gca()
    
    plt.rcParams['hatch.linewidth'] = 0.3
    
    # x = 32
    # w = 2
            
    def rectangle(layer_name,top,bottom,x,width):
        if "sand" in layer_name:
            hatch = "....."
            c = "#fffbc2"
        elif "clay" in layer_name:
            hatch = "/////"
            c = "#c7ff8f"
        elif "interlaminated" in layer_name:
            hatch = "xxxx...."
            c = "#de8d02"
        else:
            hatch = "xxxx"
            c = "#de8d02"
        r = mpatch.Rectangle((x,bottom), width = width, height = top-bottom,
                             fc = c, hatch = hatch, clip_on = False, ec = "black")
        
        ax.add_artist(r)
        rx, ry = r.get_xy()
        cx = rx + r.get_width()/2.0
        cy = ry + r.get_height()/2.0
    
        ax.text(cx+3, cy,layer_name, color='black', clip_on=False,zorder=5,rotation=90,
                    fontsize=8, ha='center', va='center')
        
    for layer in layers:
        print(layer)
        rectangle(layer,top = layers[layer][0],bottom = layers[layer][-1],
                  x = x,width = width)
        if hlines:
            ax.axhline(layers[layer][0],c="k",linestyle="--")
            ax.axhline(layers[layer][-1],c="k",linestyle="--")
        
