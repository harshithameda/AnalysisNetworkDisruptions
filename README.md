# AnalysisNetworkDisruptions
This Github repository provides the link to access the code to replicate one of our research works where we try to analyze airport network disruptions while integrating geospatial information.

Routes.txt file has the raw flight routes data; filtered_data.txt file has filtered flight route data for 354 airports on 60 airlines in USA.
location_data.txt file has the location data for all the 354 US airports.

The closest_airports file has required functions to identify the closest airports to a forecast location at a specific NHC cone radius. It uses the location data of 354 U.S. airports.

The disruptor equation file has required functions to compute the disruptor equation value including weighted combination value for a single airport and also weighted combination disruptor  equation values
for an entire airport network. It reads the filtered routes and location data of 354 U.S. airports.

The disruptee equation file has required functions to compute the disruptee equation values of either a single airport/all airports in a network when a particular airport is classified as a disruptor at a particular
distance. It reads the filtered routes data of 354 U.S airports

The Rerouting metric components file has required functions to compute the rerouting metric values when a particular airport is classified as a disruptee.
It reads the filtered routes and location data of 354 U.S. airports; disruptee equation values file.

