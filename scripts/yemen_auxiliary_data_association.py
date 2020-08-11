"""
This script is used to associate auxiliary/extra information with sana'a grid.
Aux information includes infrastructure and population related information
@zeal
"""
import shapely
from shapely.geometry import Point, Polygon, box
import pandas as pd
import geopandas as gpd
import rasterio as rio
import rasterio.mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import code
import datetime
from yemen_plotting_utils import *

def reproject_rasters_to_epsg4326(data_path, input_filename, updated_filename):
    dst_crs = "EPSG:4326"

    with rasterio.open(data_path + input_filename) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(data_path + updated_filename, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
    return None

def crop_raster(bbox, data_path, filename, updated_filename):
    df = rio.open(data_path + filename)
    out_image, out_transform = rasterio.mask.mask(df, [bbox], crop=True)
    out_meta = df.meta

    out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

    with rasterio.open(data_path + updated_filename, "w", **out_meta) as dest:
        dest.write(out_image)
    # code.interact(local=locals())
    return None

def extract_education_sites_data(dg, bbox, extra_data_path):
    # extract all educational sites that lie inside the bounding box
    db = gpd.read_file(extra_data_path + "hotosm_yem_education_facilities_points_shp/hotosm_yem_education_facilities_points.shp")
    db["lies_inside_bbox"] = db.geometry.apply(lambda x: x.within(bbox))
    db = db[(db.lies_inside_bbox == True)]
    list_of_ed_sites = db.geometry.values

    # Now that we know which educational sites lie inside the box
    # we calculate how many schools lie inside each grid cell
    dg["ed_sites"] = dg["geometry"].apply(lambda x: np.sum(list_of_ed_sites.within(x)))
    return dg

def extract_health_sites_data(dg, bbox, extra_data_path):
    # extract all health sites that lie inside the bounding box
    db = gpd.read_file(extra_data_path + "healthsites_shapefiles/healthsites.shp")
    db["lies_inside_bbox"] = db.geometry.apply(lambda x: x.within(bbox))
    db = db[(db.lies_inside_bbox == True)]
    list_of_health_sites = db.geometry.values

    # Now that we know which health sites lie inside the box
    # we calculate how many health sites lie inside each grid cell
    dg["health_sites"] = dg["geometry"].apply(lambda x: np.sum(list_of_health_sites.within(x)))
    return dg

def extract_road_data(dg, bbox, extra_data_path):
    # extract all roads that lie inside the bounding box
    db = gpd.read_file(extra_data_path + "ymn-roads/Ymn-Roads.shp")
    db["road_id"] = db.index.map(lambda x: "road_{}".format(x))

    # create a dictionary that uses road geometry as key and road_id as value
    dict_roads = {}
    for rid in db.road_id.unique():
        db1 = db[(db.road_id == rid)]
        dict_roads[str(db1.geometry.values[0])] = rid

    list_of_roads = db.geometry.unique()
    list_of_roads = list_of_roads[list_of_roads.within(bbox)]

    # Now that we know which roads lie inside the box
    # we calculate how many roads lie intersect each grid cell
    dg["total_roads"] = dg["geometry"].apply(lambda x: np.sum(list_of_roads.intersects(x)))
    dg["which_roads"] = dg["geometry"].apply(lambda x: list_of_roads[list_of_roads.intersects(x)])
    # replace road geometries with road_id instead for ease of use
    dg["which_roads"] = dg["which_roads"].apply(lambda x: [dict_roads[str(r)] for r in x])
    return dg

def extract_street_data(dg, bbox, extra_data_path):
    # extract all streets that lie inside the bounding box
    db = gpd.read_file(extra_data_path + "yem_trs_streets_osm/yem_trs_streets_osm.shp")
    db["street_id"] = db.index.map(lambda x: "street_{}".format(x))

    # create a dictionary that uses road geometry as key and road_id as value
    dict_streets = {}
    for sid in db.street_id.unique():
        db1 = db[(db.street_id == sid)]
        dict_streets[str(db1.geometry.values[0])] = sid

    list_of_streets = db.geometry.unique()
    list_of_streets = list_of_streets[list_of_streets.within(bbox)]

    # Now that we know which streets lie inside the box
    # we calculate how many and which streets intersect each grid cell
    dg["total_streets"] = dg["geometry"].apply(lambda x: np.sum(list_of_streets.intersects(x)))
    dg["which_streets"] = dg["geometry"].apply(lambda x: list_of_streets[list_of_streets.intersects(x)])

    # replace road geometries with road_id instead for ease of use
    dg["which_streets"] = dg["which_streets"].apply(lambda x: [dict_streets[str(s)] for s in x])
    return dg

def extract_builtup_area(dg, extra_data_path, filename):
    """
    Multi-temporal classification of built-up presence.
    30m Spatial resolution
    0 = no data
    1 = water surface
    2 = land no built-up in any epoch
    3 = built-up from 2000 to 2014 epochs
    4 = built-up from 1990 to 2000 epochs
    5 = built-up from 1975 to 1990 epochs
    6 = built-up up to 1975 epoch
    """
    df = rio.open(extra_data_path + filename)
    dg["land_area"] = dg.geometry.apply(lambda x: np.count_nonzero(rasterio.mask.mask(df, [x], crop=True)[0] == 2))
    dg["builtup_area"] = dg.geometry.apply(lambda x: np.count_nonzero(rasterio.mask.mask(df, [x], crop=True)[0] == 3))
    dg["builtup_area"] = dg["builtup_area"] * 30 * 30 #builtup area in m^2 because each pixel is 30x30m
    dg["land_area"] = dg["land_area"] * 30 * 30
    return dg

def extract_pop_data(dg, extra_data_path, filename):
    """
    Population density for epoch 2015 with spatial resolution of 250m
    Values are expressed as decimals (Float) from 0 to 442591
    NoData [-200]
    """
    df = rio.open(extra_data_path + filename)
    dg["total_pop"] = dg.geometry.apply(lambda x: rasterio.mask.mask(df, [x], crop=True)[0])
    dg["total_pop"] = dg.total_pop.apply(lambda x: np.sum(x[x>0]))
    return dg

def extract_worldpop_data(dg, extra_data_path, filename):
    reproject_rasters_to_epsg4326(extra_data_path + "GHS_POP_E2015_GLOBE_R2019A_54009_250_V1_0_22_7/", "GHS_POP_E2015_GLOBE_R2019A_54009_250_V1_0_22_7.tif", "GHS_POP_E2015_GLOBE_R2019A_54009_250_V1_0_22_7_reprojected.tif")
    crop_raster(bbox, extra_data_path + "GHS_POP_E2015_GLOBE_R2019A_54009_250_V1_0_22_7/", "GHS_POP_E2015_GLOBE_R2019A_54009_250_V1_0_22_7_reprojected.tif", "GHS_POP_E2015_GLOBE_R2019A_54009_250_V1_0_22_7_cropped.tif")
    dg_pop = extract_pop_data(dg_gdf, extra_data_path+"GHS_POP_E2015_GLOBE_R2019A_54009_250_V1_0_22_7/", "GHS_POP_E2015_GLOBE_R2019A_54009_250_V1_0_22_7_cropped.tif")
    return None

if __name__=='__main__':
    NL_data_path = "./"
    extra_data_path = "../extra_datasets/"
    output_path = "./"

    # data_path = "/home/zshah/yemen_files/data/"
    # output_path = "/home/zshah/yemen_files/outputs/"

    #--------read & extract just coordinates for NL data----------
    dg = pd.read_hdf(NL_data_path + "filtered_data.h5", key="zeal")
    dg = dg[["id","Latitude","Longitude"]].drop_duplicates().reset_index(drop=True)
    # create buffered geodataframe
    dg_gdf = create_geodataframe(dg, radius=462, cap_style=3, buffered=True)

    # extract bounding box of our gridded data
    b_coords = dg_gdf.geometry.total_bounds #Ref: https://geopandas.org/reference.html
    bbox = shapely.geometry.box(b_coords[0], b_coords[1], b_coords[2], b_coords[3])

    # #-----------Count education sites per gridcell---------------------------
    # dg_education = extract_education_sites_data(dg_gdf, bbox, extra_data_path)
    # print("Education sites done")

    # #-----------Count health sites per gridcell------------------------------
    # dg_health = extract_health_sites_data(dg_gdf, bbox, extra_data_path)
    # print("Health sites done")

    # #-----------Count roads per gridcell-------------------------------------
    # dg_road = extract_road_data(dg_gdf, bbox, extra_data_path)
    # print("Roads done")

    # #-----------Count streets per gridcell-----------------------------------
    # dg_street = extract_street_data(dg_gdf, bbox, extra_data_path)
    # print("Streets done")

    # #-----------Count built-up area------------------------------------------
    # # reproject built-up area raster to EPSG 4326
    # # reproject_rasters_to_epsg4326(extra_data_path + "GHS_BUILT_LDSMT_GLOBE_R2018A_3857_30_V2_0_16_10/", "GHS_BUILT_LDSMT_GLOBE_R2018A_3857_30_V2_0_16_10.tif", "GHS_BUILT_LDSMT_GLOBE_R2018A_3857_30_V2_0_16_10_reprojected.tif")

    # # once reprojected, crop the raster using the bounding box
    # # crop_raster(bbox, extra_data_path + "GHS_BUILT_LDSMT_GLOBE_R2018A_3857_30_V2_0_16_10/", "GHS_BUILT_LDSMT_GLOBE_R2018A_3857_30_V2_0_16_10_reprojected.tif", "GHS_BUILT_LDSMT_GLOBE_R2018A_3857_30_V2_0_16_10_cropped.tif")

    # # now count the built up area in cropped raster
    # dg_built = extract_builtup_area(dg_gdf, extra_data_path+"GHS_BUILT_LDSMT_GLOBE_R2018A_3857_30_V2_0_16_10/", "GHS_BUILT_LDSMT_GLOBE_R2018A_3857_30_V2_0_16_10_cropped.tif")
    # print("Built up area done")

    # #-----------Count population per gridcell-------------------------------
    # # reproject built-up area raster to EPSG 4326
    # # reproject_rasters_to_epsg4326(extra_data_path + "GHS_POP_E2015_GLOBE_R2019A_54009_250_V1_0_22_7/", "GHS_POP_E2015_GLOBE_R2019A_54009_250_V1_0_22_7.tif", "GHS_POP_E2015_GLOBE_R2019A_54009_250_V1_0_22_7_reprojected.tif")

    # # once reprojected, crop the raster using the bounding box
    # # crop_raster(bbox, extra_data_path + "GHS_POP_E2015_GLOBE_R2019A_54009_250_V1_0_22_7/", "GHS_POP_E2015_GLOBE_R2019A_54009_250_V1_0_22_7_reprojected.tif", "GHS_POP_E2015_GLOBE_R2019A_54009_250_V1_0_22_7_cropped.tif")

    # # now count the built up area in cropped raster
    # dg_pop = extract_pop_data(dg_gdf, extra_data_path+"GHS_POP_E2015_GLOBE_R2019A_54009_250_V1_0_22_7/", "GHS_POP_E2015_GLOBE_R2019A_54009_250_V1_0_22_7_cropped.tif")
    # print("Population done")

    #-------------Count pop using worldpop data-----------------------------
    extract_worldpop_data()

    code.interact(local=locals())
