# -*- coding: utf-8 -*-
"""
Created on Tue May 18 13:15:22 2021

@author: harshitha meda
"""

import networkx as nx
# import matplotlib.pyplot as plt
import numpy as np
import geopy
# from geopy import distance
from math import sqrt, radians
from geopy.point import Point
import pandas as pd
# from geopy.distance import great_circle
# import googlemaps
import time


def disruptee_equation_values(G,i,k, mindist,dic):
    
    """Returns normalization factors, unnormalized  values of the components 
    present in the disruptee equation, normalized disruptee equation value 
    
    Takes the 5 arguments:Graph/network of airports, specific airport to quantify as a disruptee, 
                            airport quantified as a disruptor, hurricane impact distance,
                            dictionary of location data
                            
                                
    
    Args:
        G=Graph/network of airports
        i= Specific airport to quanitfy as a disrutpee
        k= Airport quatified and classified as a disruptee
        dic= dictionary of location (lat and long) data of every airport present in the  network
        mindist = hurricane impact distance from forecast data of a hurricane
        
                     
    """  
    dis=0
    x=0
    y=0
    z=0
    n1=0
    n2=0
    n3=0
    simple_paths=[]
    
    for j in G.nodes():
        for path in nx.all_simple_paths(G, source=i,target=j,cutoff=3):
            simple_paths.append(path)
    
    # print(simple_paths)
    single_paths=0
    total_single_paths=0
    total_multiple_paths = 0
    multiple_paths = 0
    geopaths=0
    total_geopaths=0
    remove_paths=[]
    
    for path in simple_paths:
        
        if len(path) == 2:            

            total_single_paths+=1    
            
            if k in path[1]:
                remove_paths.append(path)

                single_paths+=1
        
        
        if len(path) > 2:
            total_multiple_paths += 1
            
            if k in path[1:len(path)]:
                remove_paths.append(path)
                multiple_paths+=1
       

    if single_paths < total_single_paths or multiple_paths < total_multiple_paths:
            for path in simple_paths:
                if path not in remove_paths:
                    total_geopaths+=1
                    
                    dist=[]
                    for u,v in enumerate(path[:-1]):
#                        
                       d = abs(geopath(dt[path[u]],dt[path[u+1]],dt[k]))
                       # print(d)
                       dist.append(d)
                       
                    if min(dist) <= mindist:
##                          print(path)                           
                        geopaths+=1   

    
    
    if total_single_paths > 0:    
        x= single_paths/total_single_paths
        n1 += 1/total_single_paths
        
                
    if total_multiple_paths >0:
        y= multiple_paths/total_multiple_paths
        n2 += 1
    
  
    if total_geopaths > 0:
        z=geopaths/total_geopaths        
        n3 += 1
    
    norm = n1+n2+n3
    # print(norm)
    dis = (x)+(y)+(z)
    
    return [i,norm,x,y,z,dis/norm]



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
    
    
    
if __name__=='__main__':   
    df2=pd.read_excel('filtered_data.xlsx')
    G=nx.from_pandas_edgelist(df2, 'Source_Apt', 'Destination_Apt')

 
    """Need to call function: disruptee_equation_values(G,i,k, mindist,dic): 
    for obtaining normalization factor,unnormalized, normalized components 
    of the disruptee equation for a single airport in the network (provided i should not be same as k)"""
    print('[airport,'
    'normalization_factor,' 'Dest_comp,' 'Trans_comp,''Geo-comp,' 'Dis_equ_val]:')
   
    print(disruptee_equation_values(G,'ORH','MCO', 173.801,dt))
      
    start_time = time.time()
    """Use the below code to explore disruptee equation values of all airports for a disruptor as an excel file. """
    disruptor='MCO'
    filename='MCO_disruptees'
    mindist=173.801
    All_list=[]    
    for j in G.nodes():
        if j!=disruptor:
            All_list.append(disruptee_equation_values(G,  j,       disruptor, mindist,dt))

    df=pd.DataFrame(All_list, columns=['airport',
    'normalization  factor', 'Dest_comp', 'Trans_comp','Geo-comp','Dis_equ_val'])
        
    df.to_excel(filename+'.xlsx')  

    print("--- %s seconds ---" % (time.time() - start_time))

    