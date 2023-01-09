# -*- coding: utf-8 -*-
"""
Created on Tue May 18 13:45:02 2021

@author: harsh
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
import googlemaps
import re


def des_nodes(G,i):
    
    """Returns destination airports for an airport in a graph/ network of airports
    
    Takes the 2 arguments: graph/network of a airports, airport whose destinations needs to 
    identified
    
    
    Args:
        G= graph/network of airports
        i=airport whose destinations needs to identified
        
        
    """
    
    lst=[]
    for j in G.nodes():
        for path in nx.all_simple_paths(G, source=i, target=j, cutoff=3):
            lst.append(path)
    
    des_nodes=[]
    for path in lst:
        des_nodes.append(path[len(path)-1])
        
    return list(set(des_nodes))     

def airline_proportion(df, apt, re_apt):
    """Returns unnormalized airline proportion value, normalization factor, 
    normalized airline proportion value
    
    Takes the 2 arguments: dataframe with all airports information, disruptee airport, 
    rerouting airport that satisfied cutoffs
    
    
    Args:
        df= dataframe with all airports present in entire airport network information
        apt=disruptee airport
        re_apt=rerouting airport that satisfied cutoffs
        
    """
    n=0
    norm=0
    airlines=list(df.Airline.unique())
    print(airlines)
    for i in airlines:
        df1=df[df['Airline']== i]
        print(df1)
        G=nx.from_pandas_edgelist(df1, 'Source_Apt', 'Destination_Apt')
#        nx.draw(G)
#        plt.show()
        if apt in set(G):
            list1=[]
            list1.extend(des_nodes(G,apt))
            print(list1)
            norm+=1
            if re_apt in set(G):
                list2=[]
                list2.extend(des_nodes(G,re_apt))
                print(list2)
                n += len([x for x in des_nodes(G,re_apt) if x in des_nodes(G,apt)])/len(des_nodes(G,apt))
                print(n)

                
    return n, norm, n/norm

def disruptee_equation_values_rerouting_airport(G,i,k,p, mindist,dic):
    
    """Returns normalization factors, unnormalized  values of the components 
    present in the disruptee equation, normalized disruptee equatiom value 
    
    Takes the 5 arguments:Graph/network of airports, airport that met rerouting cutoffs, 
                            airport quantified as a disruptor,disruptee airport from where rerouting 
                            airport is beign estimated,hurricane impact distance, dictionary of location data
                            
                                
    
    Args:
        G=Graph/network of airports
        i= airport that cutoffs for rerouting 
        k= Airport quatified and classified as a disruptee
        p=disruptee airport from where rerouting airport is beign estimated
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
    
    for j in des_nodes(G, p):
        for path in nx.all_simple_paths(G, source=i,target=j,cutoff=3):
            simple_paths.append(path)
    

    single_paths=0
    total_single_paths=0
    total_multiple_paths = 0
    multiple_paths = 0
    geopaths=0
    total_geopaths=0
    remove_paths=[]
    
    for path in simple_paths:
        
        if len(path) == 2:            
#            print(path)
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
    
    return norm,x,y,z,dis/norm



def rerouting_options(df,df1,i,j,G1,c1, c2, p, dis, dt, filename):
    
    """Returns excel file with rerouting options for a disruptee and the reouting metric
    component values
    
    Takes the 10  arguments: two dataframes, disruptee, disruptee equation value,graph/ network of airports
      two cutoff values, disruptor, hurricane impact distance, dictionary of location data
    
    Args:
        df= dataframe with all disruptees and their equation values for a disruptor
        df1=dataframe with all airports present in entire airport network information
        i= disruptee for which we want to identify reoruting airports
        j= disruptee equation value
        G1=graph/network of airports
        c1=threshold disruptee values
        c2= proximity distance
        p=disruptor airport
        dis=hurricane impact distance
        df= dictionary of location data
        filename= name for the output file
                            
    """
    list1=[]
    list3=[]
    list1.extend(des_nodes(G1,i))
    print(len(list1))
    for k in G1.nodes():
        if i != k:
            for s,t in df.loc[df['Airport'] == k , [4]].iterrows(): 
                u = (j -  t[4])/j
                if u > c1 :
                    d = great_circle_distance__haversine(dt[i],dt[k])  
                    # d = google_distance(i,k,dt) 
                    if d  <= c2:
                        list2=[]
                        list2.extend(des_nodes(G1,k))
                        print(len(list2))
                        v = len([x for x in list2 if x in list1])/len(list1)
                        x=list(disruptee_equation_values_rerouting_airport(G1, k, p, i, dis, dt))[4]
                        DV=(j-x)/j
                        AP=airline_proportion(df1,i, k)[2]
                        list3.append([i,k, DV, v, AP,DV+v+AP])
                        print(list3)
    df2= pd.DataFrame(list3, columns=['Disruptee', 'Close_Apt','Change_in _Dis_val','Net_Proportion','airline_prop','Rerouting_metric'])               
     
    return df2.to_excel(filename+'.xlsx')  


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


def google_distance(a,b,dt):
    
    """Returns road distance between two airports 
    
    Takes the 3 arguments: two airports, dictionary of location data
                                
    
    Args:
        a= airport1
        b=airport2
        dt=dictionary of location data
                                  
    """
    
    gmaps = googlemaps.Client(key='AIzaSyCBY6rYdRvWTtQtwS5JOoRotQ9wV4PwaHE') 
    distance = gmaps.distance_matrix([str(dt[a][0]) + " " + str(dt[a][1])], [str(dt[b][0]) + " " + str(dt[b][1])], mode='driving')['rows'][0]['elements'][0]
    # print(distance)
    if distance['status'] != 'OK':
        dis = 10000
        
    else:
        dis = float(''.join(re.findall(r"[-+]?\d*\.\d+|\d+", distance['distance']['text'])))/1.60934
    
    return dis
    
    

# def euclidean_distance(a,b,dt):
#     x= dt[b][0]-dt[a][0]
#     y=(dt[b][1] - dt[a][1])* np.cos(radians(dt[a][0]))
#     return (110.25*np.sqrt(x**2 + y**2))/1.60934
    


""" Dictionary of airports and latitude, longitude location data """
dt={}  
with open('location_data.txt','r') as inf:
    
    dt = eval(inf.read())   

if __name__=='__main__':    
    file_dis_values='MCO_4_3_173.801'
    df=pd.read_excel(file_dis_values+'.xlsx')
    df1=pd.read_excel('filtered_data.xlsx')
    G1=nx.from_pandas_edgelist(df1, 'Source_Apt', 'Destination_Apt')


    
    """Need to call function: rerouting_options(df, df1, i, j, G1, c1, c2, p, dis, dt, filename)
    for complete information of reouting airports for a disrutpee"""
    print(rerouting_options(df, df1, 'ORH', 0.8177, G1, 0.5, 50, 'MCO', 173.801, dt, 'MCO-RA_TEST'))