# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 09:48:47 2020

@author: harshitha meda
"""


import pandas as pd
from geopy.distance import great_circle
import networkx as nx
from geopy.point import Point



def close_apts(G,forecast_loc,radius,dt,output_file):
    
    """Returns close airports to a forecast location at a specific NHC cone radius
    
    Takes the 4  arguments: graph, forecast location, forecast radius, dictionary with location data
                                
    
    Args:
        G= US airport network
        forecast_loc= Location of the forecast
        radius= NHC cone radius
        dt=dictionary with US airports location data
        output_file=output filename
        
                            
    """
    new_dict={}

    for i in G.nodes():
        dist=great_circle(forecast_loc,dt[i])/(1.151*1.609)
        if dist <= radius:
            new_dict[i] = dist

    print(new_dict)   
    df4=pd.DataFrame.from_dict(new_dict, orient='index',columns=[1]).sort_values(by=1,ascending=True)
    df4.to_excel(output_file + '.xlsx') 
    
    

    
dt={}
with open('location_data.txt','r') as inf:
    
    dt = eval(inf.read())


df=pd.read_excel(r'C:\Users\harsh\OneDrive - aggies.ncat.edu\Phd Research\CATM\codes\filtered_data.xlsx')
output_file='close_airports_example'
G=nx.from_pandas_edgelist(df, 'Source_Apt', 'Destination_Apt')
"""Need to call function: close_apts(G,forecast_loc,radius,dt,output_file) to
    get the closest airports to a forecast location at a specific NHC cone radius   
  
    The function run for an example forecast location at a NHC cone radius 196 is shown below
"""
close_apts(G,(37,-72),196,dt,'close_airports')

 