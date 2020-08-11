from shapely.geometry import Point, Polygon
import pandas as pd
import geopandas as gpd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import code
import datetime

def check_dates(df, baseline_date, baseline_year, comparison_date, comparison_year, baseline_strict=False, comparison_strict=False):
    # check if given dates exist in the dataset and contains majority of the grid points.
    # if not, give the nearest date possible with all the grid points present.
    dk = df.groupby("date_c").nunique()[["id"]].reset_index()
    dk.columns = ["date_c","num_points"]

    print("Date Check......")
    if baseline_strict == False:
        # find dates present in the dataset 20 days in past from the given date (For Baseline)
        dbase = dk[(dk.date_c <= baseline_date)]
        dbase = dbase[(dbase.date_c >= pd.to_datetime(baseline_date,format="%Y-%m-%d").date() - pd.offsets.Day(20))]
        # select the closest date with maximum points covered
        # sometimes max points are same so we have multiple dates - we resort to selecting maximum or closest date
        dbase = dbase[(dbase.num_points == dbase.num_points.max())].date_c.max()
    else:
        # if strict, we just return the exact same input baseline date
        dbase = pd.to_datetime(baseline_date,format="%Y-%m-%d")
    dbase_date = str(dbase.date())
    dbase_year = int(dbase.year)
    print("Baseline date = {}".format(dbase_date))

    if comparison_strict == False:
        # find dates present in the dataset 20 days in future from the given date (For Comparison)
        dcomp = dk[(dk.date_c >= comparison_date)]
        dcomp = dcomp[(dcomp.date_c <= pd.to_datetime(comparison_date,format="%Y-%m-%d").date() + pd.offsets.Day(20))]
        # select the closest date with maximum points covered
        # sometimes max points are same so we have multiple dates - we resort to selecting minimum or closest date
        dcomp = dcomp[(dcomp.num_points == dcomp.num_points.max())].date_c.min()
    else:
        # if strict, we just return the exact same input comparison date
        dcomp = pd.to_datetime(comparison_date,format="%Y-%m-%d")
    dcomp_date = str(dcomp.date())
    dcomp_year = int(dcomp.year)
    print("Comparison date = {}".format(dcomp_date))
    print("************************************************")
    return dbase_date, dbase_year, dcomp_date, dcomp_year

def baseline_db(df, baseline_date, baseline_year, include_exact_baseline_date):
    # Inputs: Origina dataframe, date/year of crisis
    # Outputs: Creates a baseline dataframe of mean radiance for same day-of-the-week 20 weeks pre-crisis
    df["dow"] = df.date_c.apply(lambda x: x.dayofweek)
    day_of_week = df[(df.date_c == baseline_date)]["dow"].unique()[0] #extract day of week corresponding to the crisis day

    if include_exact_baseline_date == True:
        #select all the NL readings corresponding to the same day of week for 20 weeks prior to the crisis
        pre = df[(df.dow == day_of_week) & (df.date_c <= baseline_date)]
    else:
        pre = df[(df.dow == day_of_week) & (df.date_c < baseline_date)]
    pre = pre[(pre.date_c > pd.to_datetime(baseline_date,format="%Y-%m-%d").date() - pd.offsets.Week(20))]
    print("Dates used to create the baseline: {}".format(pre.date_c.unique()))
    pre = pre.groupby(["id","Latitude","Longitude"]).aggregate({"RadE9_Mult_Nadir_Norm":["mean","std"]}).reset_index()
    pre.columns = ["id","Latitude","Longitude","mean_base","std_base"]
    return pre

def zscore(df, d_base, date):
    # Inputs: original data frame, baseline dataframe, date/year for which zscore has to be calculated
    # Outputs: data frame with pre-crisis baseline (constant) and crisis/post-crisis metrics
    #           like zscore, %difference

    #select all NL readings ON the day of crisis or the date for which comparisons have to be made
    post = df[(df.date_c == date)][["id","Latitude","Longitude","RadE9_Mult_Nadir_Norm"]]
    post.columns = ["id","Latitude","Longitude","comparison_rad"]
    print("Selected comparison date: {}".format(date))
    print("************************************************")

    #combine both pre and post dataframes
    db = pd.merge(d_base, post, on=["id","Latitude","Longitude"], how="left")
    epsilon = 1
    sigma_min = 0.1 #used to handle when data doesn't have any variance
    db["perc_difference"] = (db["comparison_rad"] - db["mean_base"])*100/db["mean_base"]
    db["z_score"] = (db["comparison_rad"] - db["mean_base"])/db["std_base"]
    # db["z_score"] = db["z_score"].apply(lambda x: round(x,0))
    return db

def create_disaster_map(df, baseline_date, baseline_year, comparison_date, comparison_year, include_exact_baseline_date=False, strict_baseline_date=False, strict_comparison_date=False):
    # just enter date of crisis and the original NL dataframe
    # as output it gives us dataframe with zscore and %difference
    print("************************************************")
    print("Inputs")
    print("Baseline date: {}".format(baseline_date))
    print("Baseline strictness: {}".format(strict_baseline_date))
    print("Include baseline date:{}".format(include_exact_baseline_date))
    print("Comparison date: {}".format(comparison_date))
    print("Comparison strictness: {}".format(strict_comparison_date))
    print("************************************************")
    base_date, base_year, compare_date, compare_year = check_dates(df, baseline_date, baseline_year, comparison_date, comparison_year, strict_baseline_date, strict_comparison_date, baseline_strict=True, comparison_strict=True)
    db = baseline_db(df, base_date, base_year, include_exact_baseline_date)
    dz = zscore(df, db, compare_date)
    return dz

def yemen_geodb(df):
    # create 2 generic geopandas files for yemen data
    # (1) id and point geometries
    # (2) id and polygon geometries with square buffers around each point

    # Creating (1)
    df = df[["id","Latitude","Longitude"]].drop_duplicates()
    geom = [Point(x,y) for x,y in zip(df["Longitude"], df["Latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs={"init":"epsg:4326"})

    # Creating (2)
    gdf_poly = gdf.copy()
    gdf_poly["geometry"] = gdf_poly["geometry"].apply(lambda x: x.buffer(0.0041670/2, cap_style=3))

    # code.interact(local=locals())
    # gdf_poly.to_file("./yemen_polygons", driver="ESRI Shapefile")
    return None

def associate_unosat_to_yemen_geodb(df):
    list_of_sites = df.geometry.values
    dy = gpd.read_file("./yemen_polygons/yemen_polygons.shp")

    dict_rec = {}
    i = 0
    for site in list_of_sites:
        # code.interact(local=locals())
        dict_rec[i]=dy[(dy.geometry.contains(site))].id.values[0]
        i = i+1

    id_dict = {}
    for y_id in dy.id.unique():
        id_dict[y_id] = 0
    for site, y_id in dict_rec.items():
        id_dict[y_id] = id_dict[y_id] + 1

    dy["unitar_points"] = dy.id.apply(lambda x: id_dict[x])
    dy = dy[["id","geometry","unitar_points"]]
    return dy

if __name__=='__main__':
    #---read pickled dataset------------------
    # # read yemen NL data
    # df = pd.read_hdf("../yemen_filtered_db.h5", key="zeal")
    # df["RadE9_Mult_Nadir_Norm"] = df["RadE9_Mult_Nadir_Norm"].clip(lower=0) #clip negative readings to 0
    # # remove duplicates by taking average values for multiple readings corresponding to same date
    # df = df.groupby(["id","Latitude","Longitude","date_c"]).mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()

    #---------read lunar corrected dataset--------------
    data_path = "./"
    dg = pd.read_hdf(data_path + "yemen_STL_lunar_corrected.h5", key="zeal")
    dg = dg.drop(columns = ["RadE9_Mult_Nadir_Norm"])
    dg = dg.rename(columns={"rad_corr":"RadE9_Mult_Nadir_Norm"})

    # read Yemen's shapefile and filter out shapes corresponding to regions of interest
    street_map = gpd.read_file("../yemen_shp_files/yem_admbnda_adm2_govyem_mola_20181102.shp")
    street_map = street_map[(street_map.ADM1_EN.isin(["Amanat Al Asimah","Sana'a"]))]

    # read damaged structures dataset released by UNITAR
    sites_city = gpd.read_file("../extra_datasets/unitar_unisat_data/Damage_Sites_Sanaa_City_20150910.shp")
    sites_airport = gpd.read_file("../extra_datasets/unitar_unisat_data/Damage_Sites_Sanaa_Airport_20150515.shp")
    sites_damaged = pd.concat([sites_city, sites_airport]).reset_index()
    sites_damaged = sites_damaged.drop(columns = {"index"})

    dz = create_disaster_map(df.copy(), baseline_date="2015-03-26", baseline_year="2015", comparison_date="2015-05-15", comparison_year="2015", include_exact_baseline_date = False, strict_baseline_date=True, strict_comparison_date=False)
    # yemen_geodb(dz.copy()) #Uncomment to create geo databases for Yemen
    dy = associate_unosat_to_yemen_geodb(sites_damaged)
    db = pd.merge(dz, dy, on=["id"], how="left")
    dj = db.groupby("unitar_points").mean()[["z_score"]].reset_index()

    code.interact(local=locals())