from shapely.geometry import Point, Polygon
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pylab import *
import datetime
import code
import pickle

def getRadius(buffer):
    lat_degree = 110.54 * 1000
    lon_degree = 111.32 * 1000
    lat_radius = buffer / lat_degree
    lon_radius = buffer / lon_degree
    radius = max(lat_radius,lon_radius)
    return radius

def stretch_data_for_plotting(df, column, max_rad_limit):
    # This function stretches radiance values and brings them between 0 and 255 for plotting purposes
    # Inputs: column - column that requires values to be stretched
    #         max_rad_limit - all radiance values beyond this will be assigned 255
    df[column]= df[column].apply(lambda x: 255 * np.sqrt(np.clip(x, a_min=None, a_max=max_rad_limit)/max_rad_limit))
    return df

#converts a dataframe to geodataframe with buffers instead of points.
#This is used to produced visualizations very similar to NL visualizations available online.
def create_geodataframe(df, radius, cap_style, buffered=True):
    """
    radius - in meters
    cap_style - 1 for round buffer and 3 for square buffer
    """
    geom = [Point(x,y) for x,y in zip(df["Longitude"], df["Latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs={"init":"epsg:4326"})
    if buffered == True:
        distance = getRadius(radius)
        gdf["geometry"] = gdf["geometry"].apply(lambda x: x.buffer(distance/2, cap_style=cap_style))
    return gdf

def plot_geospatial_heatmap_subplots(geo_df, col_name, title, cmap, cmap_type, with_streetmap=False, with_sites=False, add_title=False, ax=None):
    ax = ax

    if with_streetmap == True:
        # read Yemen's shapefile and filter out shapes corresponding to regions of interest
        street_map = gpd.read_file("../yemen_shp_files/yem_admbnda_adm2_govyem_mola_20181102.shp")
        street_map = street_map[(street_map.ADM1_EN.isin(["Amanat Al Asimah","Sana'a"]))]
        street_map['coords'] = street_map['geometry'].apply(lambda x: x.representative_point().coords[:])
        street_map['coords'] = [coords[0] for coords in street_map['coords']]

    if with_sites == True:
        # read damaged structures dataset released by UNITAR
        sites_city = gpd.read_file("./extra_datasets/unitar_unisat_data/Damage_Sites_Sanaa_City_20150910.shp")
        sites_airport = gpd.read_file("./extra_datasets/unitar_unisat_data/Damage_Sites_Sanaa_Airport_20150515.shp")

    # piece of code to standardize pixel cmap (Reference: https://stackoverflow.com/questions/28752727/map-values-to-colors-in-matplotlib)
    lst = np.arange(256)
    minima = 0
    maxima = 255
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    geo_df["mapped_values"] = geo_df[col_name].apply(lambda x: mapper.to_rgba(x)[0])

    # use legend = True to add colorbar and legend = False to remove colorbar
    final_plot = geo_df.plot(ax=ax, column="mapped_values", markersize=20, cmap=cmap_type, legend=False)

    if with_streetmap == True:
        street_map.plot(ax = ax, color = 'white', edgecolor = 'black')

    if with_sites == True:
        city.plot(ax=ax, color="black", markersize=10, marker="x", alpha=0.6)
        airport.plot(ax=ax, color="black", markersize=10, marker="x", alpha=0.6)

    ax.set_axis_off()
    if add_title==True:
        ax.set_title(title)

    return final_plot

def plot_geospatial_heatmap_with_event_locs(geo_df, col_name, events_data, title, cmap, cmap_type, marker_color, events_data_type, max_stretch=255, needs_colormapping=True, add_title=False, event_locs_included=False, include_colorbar=False, with_streetmap=False, ax=None):
    ax = ax

    if with_streetmap == True:
        # read Yemen's shapefile and filter out shapes corresponding to regions of interest
        street_map = gpd.read_file("../yemen_shp_files/yem_admbnda_adm2_govyem_mola_20181102.shp")
        street_map = street_map[(street_map.ADM1_EN.isin(["Amanat Al Asimah","Sana'a"]))]
        street_map['coords'] = street_map['geometry'].apply(lambda x: x.representative_point().coords[:])
        street_map['coords'] = [coords[0] for coords in street_map['coords']]

    if needs_colormapping == True:
        # piece of code to standardize pixel cmap (Reference: https://stackoverflow.com/questions/28752727/map-values-to-colors-in-matplotlib)
        lst = np.arange(max_stretch+1)
        minima = min(lst)
        maxima = max(lst)
        norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        geo_df["mapped_values"] = geo_df[col_name].apply(lambda x: mapper.to_rgba(x)[0])
    else:
        geo_df["mapped_values"] = geo_df[col_name]

    # use legend = True to add colorbar and legend = False to remove colorbar
    # add argument legend_kwds={'label': "Z-score values"}, to add colorbar label
    final_plot = geo_df.plot(ax=ax, column="mapped_values", markersize=30, cmap=cmap_type, legend=include_colorbar, zorder=0)
    # plt.tick_params(labelsize=200)

    #NOTE: For polygons, use facecolor="none" to get transparent fill
    if with_streetmap == True:
        street_map.plot(ax=ax, facecolor="none", edgecolor='black', zorder=1, alpha=0.4)
        xlim = ([geo_df.total_bounds[0],  geo_df.total_bounds[2]])
        ylim = ([geo_df.total_bounds[1],  geo_df.total_bounds[3]])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    if event_locs_included == True:
        if events_data_type == "locations_points":
            events_data.plot(ax=ax, color="yellow", markersize=100, marker="x", zorder=20)
        elif events_data_type == "locations_buffered":
            events_data.plot(ax=ax, facecolor="none", edgecolor="yellow", linewidth=2, zorder=20)

    ax.set_axis_off()
    if add_title==True:
        ax.set_title(title)

    return final_plot