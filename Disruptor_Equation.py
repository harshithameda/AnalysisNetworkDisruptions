# -*- coding: utf-8 -*-
"""
Created on Tue May 18 12:00:35 2021

@author: harshitha meda
"""


# from geopy import distance
from math import radians, sqrt
from geopy.point import Point
# from networkx.utils.decorators import not_implemented_for
from scipy.special import entr
import itertools
import pandas as pd
import numpy as np
# import operator
import math
import time
import networkx as nx
# import matplotlib.pyplot as plt
import geopy


def Disruptor_equation_three_components(G,i,dt,mindist, normalized=True):
    
    """Returns list of  normalized intermediate connection component, 
    normalized immediate connection component, and normalized geospatial component
    
    Takes the 4  arguments:Graph/network of airports, specific airport to quantify, 
                            dictionary with location data of airports, hurricane impact distance
                            
                                
    
    Args:
        G=Graph/network of airports
        i= Specific airport for quantification 
        dt= dictionary of location (lat and long) data of every airport present in the  network
        mindist = hurricane impact distance from forecast data of a hurricane
        
                     
    """  

    V=set(G)
    bet_cen=0
    # bet_geo=0
    bet_geo_no=0
    norm_geo=0
    queue=[]

    for pair in itertools.combinations(V, 2):
        if i != pair[0] and i!= pair[1]:
            try:
                paths_list=[]
                bet_paths=0
                geo_paths=0
                paths=0
                paths_geo=0
         
                if [pair[0], pair[1] ] not in queue and [pair[1], pair[0]] not in queue:
                    queue.append([pair[0], pair[1]])
         
                    for path in nx.all_simple_paths(G, source=pair[0], target=pair[1], cutoff=3):
                        paths_list.append(path)
#                
            
                    for p in paths_list:
#           
                        paths+=1
                        if  i in p[1:len(p)-1]:

                            bet_paths+=1
                    
                        else: 

                            paths_geo+=1
                            dist=[]
                            for j,k in enumerate(p[:-1]):                            
                                d = abs(geopath(dt[p[j]],dt[p[j+1]],dt[i]))
#                     
                                dist.append(d) 
                                
                            if min(dist) <= mindist:

                                geo_paths+=1 
            
                        
                    if paths > 0:

                        bet_cen+=bet_paths/paths
                        # bet_geo+=geo_paths/paths
                        
                    if paths_geo>0:
                        
                        bet_geo_no+=geo_paths/paths_geo
                        norm_geo+=1
            
            except nx.exception.NetworkXNoPath:
                bet_cen += 0
                # bet_geo += 0
                bet_geo_no += 0
    
    if normalized:
        scale = 1 / ((len(G) - 1) * (len(G) - 2))
       
        if not G.is_directed():
            scale *= 2
           
    else:
        scale = None
        
    if scale is not None:

        bet_cen *= scale
        # bet_geo *= scale

    
    return [bet_cen, nx.degree_centrality(G)[i], bet_geo_no/norm_geo]


def weighted_combination_direct(G,k,dt,mindist, normalized=True):
    
    """Returns dictionary of airports and disruptor equation values 
    
    Takes the 4  arguments:Graph/network of airports, specific airport to quantify, 
                            dictionary with location data of airports, hurricane impact distance
                            
                                
    
    Args:
        G=Graph/network of airports
        i= Specific airport for quantification 
        dt= dictionary of location (lat and long) data of every airport present in the  network
        mindist = hurricane impact distance from forecast data of a hurricane
        
                     
    """  
    A1=[[0,0,0]]
    nodelist=['na']
    for i in G.nodes():
        nodelist.append(i)
        y=Disruptor_equation_three_components(G,i,dt,mindist, normalized=True)
        print(i,y)
        A1=np.vstack((A1, np.array(y)))
#        print(A1)
#    print(A1)

    A = np.copy(A1)
    row_sums = A.sum(axis=0)
    A[:] = A / row_sums 

    x=entr(A).sum(axis=0)/math.log(len(G.nodes()))
#    print(x)
    x[np.isnan(x)]=0
    print(x)
    x=(1-x)/(k-np.sum(x))
    print("weights")
    print(x)
    
    v = len(G)
    if normalized:
        scale = 1 / ((v - 1) * (v - 2))
        scale1 = 1 / ((v - 1))
        if not G.is_directed():
            scale *= 2
    else:
        scale = None
        scale1= None
        
    if scale is not None:
        A1[:,0] *= (1*x[0])
        A1[:,1] *= (1*x[1])
        A1[:,2] *= (1*x[2])
    
    A1=np.sum(A1,axis=1)
    
    f_dict=dict(zip(nodelist,A1.tolist()))
    del f_dict['na']
    return f_dict


def weighted_combination_from_dataframe(df,k,c1, c2,c3,normalized=True):
    
    """Returns dictionary of airports and weighted combination disruptor equation values 
    
    Takes the 5  arguments: dataframe with three component values, number of components, 
    column 1 name, column 2 name, column 3 name
                                
    
    Args:
        df=dataframe with three component values 
        k= number of components       
        c1= name of first column in dataframe
        c2= name of second column in dataframe
        c3= name of third column in dataframe             
    """ 

    A1= df[[c1,c2,c3]].to_numpy()
    print(A1)
    A = np.copy(A1)
    row_sums = A.sum(axis=0)
    A[:] = A / row_sums 
    print(A)
    x=entr(A).sum(axis=0)/math.log(354)
    x[np.isnan(x)]=0
    print(x)
    x=(1-x)/(k-np.sum(x))
    print("weights")
    print(x)
    v = 354
    if normalized:
        scale = 1 / ((v - 1) * (v - 2))
        scale1 = 1 / ((v - 1))
        # if not G.is_directed():
        #     scale *= 2
    else:
        scale = None
        scale1 = None
        
    if scale is not None:
        A1[:,0] *= (1*x[0])
        A1[:,1] *= (1*x[1])
        A1[:,2] *= (1*x[2])
    
    A1=np.sum(A1,axis=1)
    
    f_dict=dict(zip(df['Airport'],A1.tolist()))
#    del f_dict['na']
    return f_dict


def geopath(a,b,c):
    """Returns distance from an airport to a path between other two airports 
    
    Takes the 3  arguments: Three airports
                                
    
    Args:
        a= origin airport on one path
        b= destination airport on one path
        c= airport from where you have to calculate the distance
                            
    """
#    print(a)
#    print(b)
#    print(c)
    d13 = great_circle_distance__haversine(a,c)
    b12 = np.radians(initial_bearing(a,b))
    b13 = np.radians(initial_bearing(a,c))
#    print(b13-b12)
    diff = abs(b13-b12)
    if diff > np.radians(180):
        diff = ((2*np.radians(180))-diff)
        
    if diff > np.radians(90):
        dxa=d13
    else:
        dxt=np.arcsin(np.sin(d13 / 3958.8) * np.sin(b13 - b12)) * 3958.8
        d12=great_circle_distance__haversine(a,b)
        d14=np.arccos(np.cos(d13/3958.8) / np.cos(dxt/3958.8) ) * 3958.8

#        print(great_circle_distance__haversine(b,c))
        if d14>d12:
#            print("yes")
            dxa=great_circle_distance__haversine(b,c)
    
        else:
            dxa=dxt

    return dxa



def cross_track_distance(a,b,c):
#                         units=ut.METER):
    
    """Returns cross track distance from an airport to a path between other two airports 
    
    Takes the 3  arguments: Three airports
                                
    
    Args:
        a= origin airport on one path
        b= destination airport on one path
        c= airport from where you have to calculate the distance
                            
    """

    d13 = great_circle_distance__haversine(a,c)
    b12 = np.radians(initial_bearing(a,b))
    b13 = np.radians(initial_bearing(a,c))
    value = np.arcsin(np.sin(d13 / 3958.8) * np.sin(b13 - b12)) * 3958.8
    return value
#    return ut.convert(value, ut.METER, units)


def great_circle_distance__haversine(a,b):
    
    """Returns great circle distance between airports 
    
    Takes the 2  arguments: Two airports
                                
    
    Args:
        a= origin airport on one path
        b= destination airport on one path
        
                            
    """
# units=ut.METER):

    
    a_lat, a_lon = radians(a.latitude), radians(a.longitude)
    b_lat, b_lon = radians(b.latitude), radians(b.longitude)
    sdlat2 = np.sin((b_lat - a_lat) / 2.) ** 2
    sdlon2 = np.sin((b_lon - a_lon) / 2.) ** 2
    a = sdlat2 + sdlon2 * np.cos(a_lat) * np.cos(b_lat)
#    value = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) * 3958.8
    value = 2 * np.arcsin(sqrt(a)) *3958.8
    return value
#    return ut.convert(value, ut.METER, units)

def initial_bearing(a,b):
    
    """Returns initial bearing between airports 
    
    Takes the 2  arguments: Two airports
                                
    
    Args:
        a= origin airport on one path
        b= destination airport on one path
        
                            
    """
    

    a_lat, a_lon = radians(a.latitude), radians(a.longitude)
    b_lat, b_lon = radians(b.latitude), radians(b.longitude)
    delta_lon = b_lon - a_lon
    y = np.sin(delta_lon) * np.cos(b_lat)
    x = np.cos(a_lat) * np.sin(b_lat) - np.sin(a_lat) * np.cos(b_lat) * np.cos(delta_lon)
    return np.degrees(np.arctan2(y, x)) % 360
#    return np.arctan2(y,x)         

def geopoint(i):
    
    """Returns latitude and longitude location data of an airport 
    
    Takes the 1  arguments: Airport whose latitude and longitude needs to be obtained
                                
    
    Args:
        i= Airport whose latitude and longitude needs to be obtained
        
                            
    """
    geolocator=geopy.geocoders.Nominatim(user_agent="specify_your_app_name_here",timeout=10)
    location=geolocator.geocode(i)
    print(location.address)
    return Point(latitude=location.latitude,longitude=location.longitude)

""" Dictionary of airports and latitude, longitude location data """
dt={}  
with open('location_data.txt','r') as inf:
    
    dt = eval(inf.read())   
    
    
df2=pd.read_excel('filtered_data.xlsx')
G1=nx.from_pandas_edgelist(df2, 'Source_Apt', 'Destination_Apt')


start_time = time.time()

"""Need to call function: Disruptor_equation_three_components(G,i,dt,mindist, normalized=True) 
    for obtaining normalized components of the the disruptor equation"""


print(Disruptor_equation_three_components(G1,'MCO',dt,173.801, normalized=True))
print("--- %s seconds ---" % (time.time() - start_time))   
   
"""Need to call function: weighted_combination_direct(G,k,dt,mindist, normalized=True) to
    obtained weighted combination of disruptor equation value   
 
"""
print('Airport, Int_con_comp, Imm_con_comp,Geo_comp:' )
# print(weighted_combination_direct(G1, 3, dt, 173.801, normalized=True))
"""Need to call function: weighted_combination_from_dataframe(df,k,c1, c2,c3,normalized=True) to
    obtained weighted combination of disruptor equation value from dataframe of 
    three component values present in the disruptor equation
 
"""