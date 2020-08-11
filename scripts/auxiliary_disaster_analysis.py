"""
This script is used to produce all the plots for Data description section (Humanitarian Events Data)
@zeal
"""
from shapely.geometry import Point, Polygon
import shapely
import pandas as pd
import geopandas as gpd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import code
import datetime
from yemen_plotting_utils import *

def getRadius(buffer):
    lat_degree = 110.54 * 1000
    lon_degree = 111.32 * 1000
    lat_radius = buffer / lat_degree
    lon_radius = buffer / lon_degree
    radius = max(lat_radius,lon_radius)
    return radius

def basic_preprocessing(df):
    df = df[["data_id","event_date","region","admin1","admin2","location","latitude","longitude","geo_precision","source","source_scale","fatalities","geometry"]]
    df_locs = df[["location","latitude","longitude","geo_precision"]].drop_duplicates()
    df1 = df.groupby(["event_date","location","latitude","longitude"]).sum()[["fatalities"]].reset_index()
    df1 = df1.rename(columns={"latitude":"Latitude", "longitude":"Longitude"}) #for sake of consistency with NL data
    return df1

# def get_data():
#     de = pd.read_pickle("../extra_datasets/conflict_data_filtered.pck")
#     de = de[(de.fatalities != 0)]
#     de = basic_preprocessing(de)
#     return de

def create_geodataframe(df, buffered, radius, cap_style):
    geom = [Point(x,y) for x,y in zip(df["Longitude"], df["Latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs={"init":"epsg:4326"})
    if buffered == True:
        distance = getRadius(radius)
        gdf["geometry"] = gdf["geometry"].apply(lambda x: x.buffer(distance/2, cap_style=cap_style))
    return gdf

def query_event_locations_by_date(dates_arr, buffered, radius, cap_style):
    # function can be used by any program to query bombing event locations and coordinates
    # user has option to query a buffered dataframe or a dataframe with just location coordinates
    df = get_data()
    df = df[(df.event_date.isin(dates_arr))]
    gdf = create_geodataframe(df, buffered, radius, cap_style)
    return gdf

def query_event_locations_by_monthyear(month_arr, year_arr, buffered, radius, cap_style):
    # function can be used by any program to query bombing event locations and coordinates
    # user has option to query a buffered dataframe or a dataframe with just location coordinates
    df = get_data()
    df = df[(df.event_date.dt.month.isin(month_arr)) & (df.event_date.dt.year.isin(year_arr))]
    gdf = create_geodataframe(df, buffered, radius, cap_style)
    return gdf

def get_data():
    NL_data_path = "./"
    extra_data_path = "../extra_datasets/"
    output_path = "./"

    #--------read & extract just coordinates for NL data--------------
    dg = pd.read_hdf(NL_data_path + "filtered_data.h5", key="zeal")
    dg = dg[["id","Latitude","Longitude"]].drop_duplicates().reset_index(drop=True)
    # create buffered geodataframe
    dg_gdf = create_geodataframe(dg, radius=462, cap_style=3, buffered=True)

    # extract bounding box of our gridded data
    b_coords = dg_gdf.geometry.total_bounds #Ref: https://geopandas.org/reference.html
    bbox = shapely.geometry.box(b_coords[0], b_coords[1], b_coords[2], b_coords[3])
    # code.interact(local=locals())
    #--------------Get conflict data----------------------------------
    df = pd.read_csv("../extra_datasets/Conflict Data for Yemen.csv")

    #--------------Filter conflict data-------------------------------
    df = df[(df.admin1.isin(["Amanat al Asimah","Sanaa"]))]
    df["event_date"] = df["event_date"].apply(lambda x: pd.to_datetime(x))
    df = df[(df.event_date>="2012-04-01") & (df.event_date<="2019-06-01")]
    geom = [Point(x,y) for x,y in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs={"init":"epsg:4326"})
    gdf["inside_bounds"] = gdf["geometry"].apply(lambda x: x.within(bbox))
    gdf = gdf[(gdf.inside_bounds == True)]
    gdf = gdf[(gdf.sub_event_type.isin(["Air/drone strike","Shelling/artillery/missile attack","Grenade","Remote explosive/landmine/IED","Suicice bomb"]))]
    gdf = gdf[["data_id","latitude","longitude","event_id_cnty","event_date","timestamp","event_type","sub_event_type","admin1","admin2","location","fatalities","geometry","notes"]]
    gdf = gdf.rename(columns={"latitude":"Latitude", "longitude":"Longitude"})
    gdf = gdf.drop(columns="geometry")
    gdf["timestamp"] = gdf["timestamp"].apply(lambda x: pd.to_datetime(x, unit="s"))
    gdf["hour_of_day"] = gdf["timestamp"].dt.hour
    return gdf