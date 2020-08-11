"""
This script is used to produce all the plots and analysis for Results & Evaluation section's Disaster mapping portion.
This script is specifically focused on disaster mapping and analysis for bombing in Sana'a on March 26, 2015.
@zeal
"""
from shapely.geometry import Point, Polygon
import pandas as pd
import geopandas as gpd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import code
import datetime
import seaborn as sns
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL
from yemen_plotting_utils import *
from auxiliary_disaster_analysis import query_event_locations_by_date, query_event_locations_by_monthyear
from shapely.ops import unary_union
import itertools
from scipy.stats import mstats

def associate_NLgrid_with_areas(df):
    # Convert Yemen dataframe to geodataframe with just points
    gdf = create_geodataframe(df, radius=462, cap_style=3, buffered=False)
    gm = gdf.geometry.values

    # read Yemen's shapefile and filter out shapes corresponding to regions of interest
    street_map = gpd.read_file("../yemen_shp_files/yem_admbnda_adm2_govyem_mola_20181102.shp")
    street_map = street_map[(street_map.ADM1_EN.isin(["Amanat Al Asimah","Sana'a"]))]
    street_map = street_map[["ADM1_EN","ADM2_EN","ADM0_EN","geometry"]].drop_duplicates()
    sm = street_map.geometry.values

    sm_dict = {}
    for adm2 in street_map.ADM2_EN.unique():
        print("{}".format(adm2))
        try:
            adm_geom = street_map[(street_map.ADM2_EN == adm2)].geometry.values
            id_lst = list(filter(adm_geom.contains, gm))
            print("Length of list: {}".format(len(id_lst)))
            if len(id_lst) == 0:
                sm_dict[adm2] = np.nan
            else:
                sm_dict[adm2] = gdf[(gdf.geometry.isin(id_lst))]["id"].unique()
        except:
            code.interact(local=locals())

    print("Finish!")
    code.interact(local=locals())

    id_dict = {}
    for key, val in sm_dict.items():
        try:
            for i in val:
                print(i)
                id_dict[i] = key
        except:
            continue

    return None

def plot_single_heatmap(df, column, max_rad_limit):
    fig, ax = plt.subplots()
    df = stretch_data_for_plotting(df, column=column, max_rad_limit=max_rad_limit)
    gdf = create_geodataframe(df)
    plot_geospatial_heatmap_subplots(gdf, col_name=column, title=None, cmap=cm.hot, cmap_type="hot", with_sites=False, add_title=False, ax=ax)
    plt.show()
    return None

def baseline_db_march2015(df, baseline_date, baseline_year, include_exact_baseline_date):
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
    # plot_single_heatmap(pre.copy(), "mean_base", max_rad_limit=300)
    return pre

def visually_compare_March_26_27_radiance(df):
    #-----------Crisis and Post Crisis Data--------------------------------
    df1 = df[(df.date_c.dt.year == 2015) & (df.date_c.dt.month == 3) & (df.date_c.dt.day.isin([25,26,27,28]))]
    df1 = stretch_data_for_plotting(df1, column="RadE9_Mult_Nadir_Norm", max_rad_limit=300)
    gdf = create_geodataframe(df1, radius=462, cap_style=3)

    #-----------Pre-crisis baseline----------------------------------------
    pre = baseline_db_march2015(df, baseline_date="2015-03-26", baseline_year=2015, include_exact_baseline_date=False)
    pre = stretch_data_for_plotting(pre, column="mean_base", max_rad_limit=300)
    pre_gdf = create_geodataframe(pre, radius=462, cap_style=3)

    #----------Events data-------------------------------------------------
    de_26 = query_event_locations_by_date(["2015-03-26"], buffered=False, radius=None, cap_style=None)
    de_26_buffered = query_event_locations_by_date(["2015-03-26"], buffered=True, radius=2500, cap_style=1)

    #----------Visualizing data--------------------------------------------
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4)
    # plot_geospatial_heatmap_with_event_locs(geo_df=gdf[(gdf.date_c == "2015-03-25")], col_name="RadE9_Mult_Nadir_Norm", events_data=de_26, title=None, cmap=cm.hot, cmap_type="hot", marker_color="white", add_title=False, event_locs_included=True, ax=ax1)
    plot_geospatial_heatmap_with_event_locs(geo_df=pre_gdf, col_name="mean_base", events_data=de_26_buffered, title=None, cmap=cm.hot, cmap_type="hot", needs_colormapping=True, marker_color="black", events_data_type="locations_buffered", add_title=False, event_locs_included=True, ax=ax1)
    plot_geospatial_heatmap_with_event_locs(geo_df=gdf[(gdf.date_c == "2015-03-26")], col_name="RadE9_Mult_Nadir_Norm", events_data=de_26, title=None, needs_colormapping=True, cmap=cm.hot, cmap_type="hot", marker_color="white", events_data_type="locations_points", add_title=False, event_locs_included=True, ax=ax2)
    plot_geospatial_heatmap_with_event_locs(geo_df=gdf[(gdf.date_c == "2015-03-27")], col_name="RadE9_Mult_Nadir_Norm", events_data=de_26, title=None, needs_colormapping=True, cmap=cm.hot, cmap_type="hot", marker_color="white", events_data_type="locations_points", add_title=False, event_locs_included=True, ax=ax3)
    plot_geospatial_heatmap_with_event_locs(geo_df=gdf[(gdf.date_c == "2015-03-28")], col_name="RadE9_Mult_Nadir_Norm", events_data=de_26, title=None, needs_colormapping=True, cmap=cm.hot, cmap_type="hot", marker_color="white", events_data_type="locations_points", add_title=False, event_locs_included=True, ax=ax4)
    plt.show()
    code.interact(local=locals())
    return None

def zscore_percchange_march2015(df, create_plot):

    #--------create baseline and crisis day dataframes-----------------
    dbase = baseline_db_march2015(df, baseline_date="2015-03-26", baseline_year=2015, include_exact_baseline_date=False)
    dcrisis = df[(df.date_c == "2015-03-26")]

    #--------calculate percentage change and z-scores------------------
    dz = pd.merge(dbase, dcrisis[["id","date_c","RadE9_Mult_Nadir_Norm"]], left_on=["id"], right_on=["id"], how="left")
    dz["std_base"] = dz["std_base"].apply(lambda x: max(x, 0.1))
    dz["perc_change"] = (dz["RadE9_Mult_Nadir_Norm"] - dz["mean_base"]) * 100.0/(dz["mean_base"] + 1)
    dz["z_score"] = (dz["RadE9_Mult_Nadir_Norm"] - dz["mean_base"])/dz["std_base"]
    dz_output = dz.copy() #to be used for all analysis. dz to be used for visualization purposes only

    if create_plot == True:
        dz["z_score"] = dz["z_score"].apply(lambda x: round(x,1))
        dz["z_score"] = dz["z_score"].clip(lower=-2)
        dz["z_score"] = dz["z_score"].clip(upper=2)
        dz_gdf = create_geodataframe(dz, radius=462, cap_style=3)

        #---------get disaster location data-------------------------------
        # de_26_buffered = query_event_locations_by_date(["2015-03-26"], buffered=True, radius=5000, cap_style=1) #buffered
        de_26_buffered = query_event_locations_by_date(["2015-03-26"], buffered=False, radius=2500, cap_style=1) #just locations

        #---------create plots---------------------------------------------
        fig, ax1 = plt.subplots(figsize=(5,5))
        plot_geospatial_heatmap_with_event_locs(geo_df=dz_gdf, col_name="z_score", events_data=de_26_buffered, title=None, cmap=cm.seismic, cmap_type="seismic", marker_color=None, events_data_type="locations_points", needs_colormapping=False, add_title=False, event_locs_included=True, include_colorbar=True, with_streetmap=True, ax=ax1)
        plt.rc('font', size=16)
        plt.tight_layout()
        plt.show()

    # code.interact(local=locals())
    return dz_output

def calculate_zscore_changes_march2015(df):
    de_26 = query_event_locations_by_date(["2015-03-26"], buffered=True, radius=2500, cap_style=1)
    de_26["event_id"] = de_26.index.map(lambda x: "event_{}".format(x))

    event_dict = {}
    for eid in de_26.event_id.unique():
        de1 = de_26[(de_26.event_id == eid)]
        event_dict[str(de1.geometry.values[0])] = eid

    zones_union = de_26.geometry.unique()
    gdf = create_geodataframe(df, radius=462, cap_style=3)

    gdf["zone"] = gdf.geometry.apply(lambda x: zones_union[zones_union.contains(x)])
    gdf["zone"] = gdf["zone"].apply(lambda x: [event_dict[str(r)] for r in x])
    gdf["len_zone"] = gdf["zone"].apply(lambda x: len(x))
    dk = gdf[(gdf.len_zone!=0)]
    dk["zone"] = dk["zone"].apply(lambda x: x[0])
    dk = dk.groupby("zone").mean()[["perc_change"]]
    dk1 = dk.groupby("zone").sum()[["RadE9_Mult_Nadir_Norm","mean_base"]].reset_index()
    dk1["perc"] = (dk1["RadE9_Mult_Nadir_Norm"] - dk1["mean_base"])*100.0/dk1["mean_base"]

    dn = gdf[(gdf.len_zone==0)]
    dn = dn[["RadE9_Mult_Nadir_Norm","mean_base"]].sum()
    (dn["RadE9_Mult_Nadir_Norm"] - dn["mean_base"])*100.0/dn["mean_base"]


    # #--------------------associate area data----------------------------------------------
    # da = pd.read_hdf("yemen_grid_with_admin.h5",key="zeal")
    # gdf = pd.merge(gdf, da[["id","adm2"]].drop_duplicates(), left_on=["id"], right_on=["id"], how="left")
    # gdf["adm2"] = gdf["adm2"].apply(lambda x: "Maain" if x=="Ma'ain" else "Aththaorah" if x=="Ath'thaorah" else "Shuaub" if x=="Shu'aub" else "Assafiyah" if x=="Assafi'yah" else "Azzal" if x=="Az'zal" else x)
    # #-------------------Obtain list of events that happened during the said interval------
    # de_locs = query_event_locations_by_monthyear([3],[2015],buffered=False,radius=None,cap_style=None)
    # de_locs = de_locs[(de_locs.event_date=="2015-03-26")]
    # de_locs = de_locs.sort_values(by="event_date")
    # de_locs = de_locs.groupby(["admin2","event_date"]).count()[["data_id"]].reset_index()

    # #------------------Combine admin2 TNL and event data----------------------------------
    # dm = pd.merge(gdf, de_locs[["data_id","admin2","event_date"]].drop_duplicates(), left_on=["adm2","date_c"], right_on=["admin2","event_date"], how="left")

    # print("Present in zone: {}".format(gdf[(gdf.present_in_zone == True)].perc_change.mean()))
    # print("Not present in zone: {}".format(gdf[(gdf.present_in_zone == False)].perc_change.mean()))

    # print("Present in zone: {}".format(gdf[(gdf.present_in_zone == True)].z_score.mean()))
    # print("Not present in zone: {}".format(gdf[(gdf.present_in_zone == False)].z_score.mean()))

    code.interact(local=locals())
    return None

def march2015_and_infra(df, di):
    print("Hi")
    """
    Information obtained in this function:
    (1) Basic population based grouping information on infrastructure and radiance
    (2) Generic damage information (how many roads, etc. were present in low zscore cells)
    (3) Structured damage information summary (table with groups of population density and impact of damage categories)
    """
    #-------------------Obtain list of events that happened during the said interval------
    de_locs = query_event_locations_by_monthyear([1,2,3,4,5],[2015],buffered=False,radius=None,cap_style=None)
    de_locs = de_locs[(de_locs.event_date=="2015-03-26")]
    de_locs = de_locs.sort_values(by="event_date")
    # de_locs = de_locs.groupby(["admin2","event_date"]).count()[["data_id"]].reset_index()

    #--------obtain baseline and crisis radiance and percentage change/z-score-----------------
    dz = zscore_percchange_march2015(df, create_plot="False")

    #---------combine zscore and infra+pop datasets------------------------------------
    dc = pd.merge(dz, di, left_on=["id"], right_on=["id"], how="left")
    dc["pop_group"] = dc["total_pop"].apply(lambda x: "Very low" if x<=20 else "Low" if 20<x<=600 else "Medium" if 600<x<=3000 else "High")

    #--------------------Infra Damage assessment---------------------------------------
    dm = dc.copy()
    dm = dm[["id","ed_sites","health_sites","which_roads","which_streets","builtup_area","total_pop","z_score","perc_change","RadE9_Mult_Nadir_Norm","mean_base"]]
    dm["z_group"] = dm.z_score.apply(lambda x: "severe" if x<=-2 else "moderate" if -2<x<=-1 else "low" if -1<x<0 else "increase")
    dmz = dm.groupby("z_group").agg({"ed_sites":"sum","health_sites":"sum","builtup_area":"sum","total_pop":"sum","RadE9_Mult_Nadir_Norm":"sum","mean_base":"sum","id":"count"})
    # .sum()[["ed_sites","health_sites","builtup_area","total_pop","RadE9_Mult_Nadir_Norm","mean_base"]]
    dmz["tnl_change"] = (dmz["RadE9_Mult_Nadir_Norm"] - dmz["mean_base"])*100.0/dmz["mean_base"]
    dmz["perc_cells"] = dmz["id"]*100.0/3285

    droads = dm.groupby("z_group")["which_roads"].apply(lambda x: np.unique(list(itertools.chain.from_iterable(x)))).reset_index()
    droads["total_roads"] = droads["which_roads"].apply(lambda x: len(x))
    dstreets = dm.groupby("z_group")["which_streets"].apply(lambda x: np.unique(list(itertools.chain.from_iterable(x)))).reset_index()
    dstreets["total_streets"] = dstreets["which_streets"].apply(lambda x: len(x))

    dmz = pd.merge(dmz, droads, on=["z_group"], how="left")
    dmz = pd.merge(dmz, dstreets, on=["z_group"], how="left")
    dmz = dmz.drop(columns=["which_streets","which_roads"])

    total_roads = len(np.unique(dm["which_roads"].sum()))
    total_streets = len(np.unique(dm["which_streets"].sum()))

    #calculate percentages of each infra type
    dm2 = dmz.copy()
    dm2["ed_sites"] = dm2["ed_sites"]*100.0/dm2["ed_sites"].sum()
    dm2["health_sites"] = dm2["health_sites"]*100.0/dm2["health_sites"].sum()
    dm2["builtup_area"] = dm2["builtup_area"]*100.0/dm2["builtup_area"].sum()
    dm2["total_pop"] = dm2["total_pop"]*100.0/dm2["total_pop"].sum()
    dm2["total_roads"] = dm2["total_roads"]*100.0/total_roads
    dm2["total_streets"] = dm2["total_streets"]*100.0/total_streets
    dm2["perc_cells"] = dm2["id"]*100.0/3285

    #---------Basic assessments comparing pop groups----------------------------------------------
    dc_pop_basic = dc.groupby("pop_group").agg({"ed_sites":"sum","health_sites":"sum","total_roads":"sum","total_streets":"sum","builtup_area":"sum","total_pop":"sum","mean_base":"mean"})
    print("*********************************************************")
    print("Mean Base Mean")
    print(dc_pop_basic)

    dc_pop_tnl = dc.groupby("pop_group").sum()[["RadE9_Mult_Nadir_Norm","mean_base"]].reset_index()
    dc_pop_tnl["perc_change"] = (dc_pop_tnl["RadE9_Mult_Nadir_Norm"] - dc_pop_tnl["mean_base"])*100.0/dc_pop_tnl["mean_base"]
    print("*********************************************************")
    print("TNL Change per Pop")
    print(dc_pop_tnl)

    #---------Structured damage information summary table----------------------------------
    # Each row will represent a pop category/baseline rad category since both are the same
    # Each column will represent Impact of damage:
    #   severe decrease(<-50%), moderate decrease(-50 to -25), slight decrease (-25 to 0), increase (> 0)
    ds = dc[["id","geometry","perc_change","z_score","pop_group"]]
    # ds["damage_group"] = ds.perc_change.apply(lambda x: "High" if x<-50 else "moderate" if -50<=x<-25 else "slight" if -25<=x<0 else "increase")
    ds["damage_group"] = ds.z_score.apply(lambda x: "severe" if x<=-2 else "moderate" if -2<x<=-1 else "low" if -1<x<0 else "increase")
    code.interact(local=locals())
    da = pd.read_hdf("yemen_grid_with_admin.h5",key="zeal")
    ds = pd.merge(ds, da[["id","adm2"]], left_on=["id"], right_on=["id"], how="left")

    # code.interact(local=locals())

    ds = ds.groupby(["pop_group","damage_group"]).count()[["id"]].reset_index()
    ds = ds.rename(columns = {"id":"sites"})
    ds_totals = ds.groupby(["pop_group"]).sum()[["sites"]].reset_index()
    ds_totals = ds_totals.rename(columns = {"sites":"total_sites"})
    ds = pd.merge(ds, ds_totals, left_on=["pop_group"], right_on=["pop_group"], how="left")
    ds["perc_cells"] = ds["sites"] * 100.0/ds["total_sites"]
    print("*********************************************************")
    print("Structured Damage Summary:")
    print("*********************************************************")
    print("{}".format(ds))

    #---------Visualize pop based grouping---------------------------------------------------
    dv = dc[["id","Latitude_x","Longitude_x","RadE9_Mult_Nadir_Norm","pop_group"]]
    dv.columns = ["id","Latitude","Longitude","RadE9_Mult_Nadir_Norm","pop_group"]
    dv_gdf = create_geodataframe(dv, radius=462, cap_style=3, buffered=True)

    # fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2)
    # plot_geospatial_heatmap_with_event_locs(geo_df=dv_gdf, col_name="RadE9_Mult_Nadir_Norm", events_data=None, title=None, cmap=cm.hot, cmap_type="hot", needs_colormapping=True, marker_color="black", events_data_type="locations_buffered", add_title=False, event_locs_included=False, include_colorbar=True, ax=ax1)

    fig, ax2 = plt.subplots(nrows=1, ncols=1)
    plot_geospatial_heatmap_with_event_locs(geo_df=dv_gdf, col_name="pop_group", events_data=None, title=None, cmap=cm.Set1, cmap_type="Set1", needs_colormapping=False, marker_color="black", events_data_type="locations_buffered", add_title=False, with_streetmap=True, event_locs_included=False, include_colorbar=True, ax=ax2)
    plt.show()

    #---------check if we can find shapefile of areas for yemen------------------------------

    code.interact(local=locals())
    return None

def zscore_percchange_may2015(df, create_plot):
    #Baseline used by UNOSAT was December 31, 2014 and crisis date was May 15, 2015

    #--------create baseline and crisis day dataframes-----------------
    dbase = baseline_db_march2015(df, baseline_date="2014-12-31", baseline_year=2014, include_exact_baseline_date=True)
    dcrisis = df[(df.date_c == "2015-05-15")]

    #--------calculate percentage change and z-scores------------------
    dz = pd.merge(dbase, dcrisis[["id","date_c","RadE9_Mult_Nadir_Norm"]], left_on=["id"], right_on=["id"], how="left")
    dz["std_base"] = dz["std_base"].apply(lambda x: max(x, 0.1))
    dz["perc_change"] = (dz["RadE9_Mult_Nadir_Norm"] - dz["mean_base"]) * 100.0/(dz["mean_base"] + 1)
    dz["z_score"] = (dz["RadE9_Mult_Nadir_Norm"] - dz["mean_base"])/dz["std_base"]
    dz_output = dz.copy() #to be used for all analysis. dz to be used for visualization purposes only
    return dz_output

def verify_using_unitar_2015(df, create_plot=True):
    # read damaged structures dataset released by UNITAR
    sites_city = gpd.read_file("../extra_datasets/unitar_unisat_data/Damage_Sites_Sanaa_City_20150910.shp")
    sites_airport = gpd.read_file("../extra_datasets/unitar_unisat_data/Damage_Sites_Sanaa_Airport_20150515.shp")
    sites_damaged = pd.concat([sites_city, sites_airport]).reset_index()
    sites_damaged = sites_damaged.drop(columns = {"index"})
    sites_damaged = sites_damaged[(sites_damaged.SiteID != "Transport Vehicle")]
    sites_damaged.SiteID = sites_damaged.SiteID.apply(lambda x: "Building" if x=="Building (General / Default)" else "Building" if x=="Greenhouse" else "Building" if x=="Industrial Facility" else x)
    # sites_damaged = sites_damaged[(sites_damaged.Main_Damag.isin(['Severe Damage', 'Destroyed']))]

    #--------obtain baseline and crisis radiance and percentage change/z-score-----------------
    dz = zscore_percchange_may2015(df, create_plot="False")
    # classify each pixel by damage
    dz["damage_group"] = dz.perc_change.apply(lambda x: "severe" if x<-50 else "moderate" if -50<=x<-25 else "slight" if -25<=x<0 else "increase")
    dz_gdf = create_geodataframe(dz.copy(), radius=462, cap_style=3)

    #--------obtain zscores corresponding to every unitar location-----------------------------
    dz_gdf["damaged_buildings"] = dz_gdf["geometry"].apply(lambda x: len(list(filter(x.contains, sites_damaged[(sites_damaged.SiteID == "Building")].geometry.values))))
    dz_gdf["damaged_roads"] = dz_gdf["geometry"].apply(lambda x: len(list(filter(x.contains, sites_damaged[(sites_damaged.SiteID == "Road")].geometry.values))))
    dz_gdf["damaged_fields"] = dz_gdf["geometry"].apply(lambda x: len(list(filter(x.contains, sites_damaged[(sites_damaged.SiteID == "Field")].geometry.values))))

    # associate admin2 with data
    da = pd.read_hdf("yemen_grid_with_admin.h5",key="zeal")
    dz_gdf = pd.merge(dz_gdf, da, left_on=["id"], right_on=["id"])

    #--------visualize grid with unitar points-------------------------------------------------
    if create_plot == True:
        dz_gdf["z_score"] = dz_gdf["z_score"].apply(lambda x: round(x,1))
        dz_gdf["z_score"] = dz_gdf["z_score"].clip(lower=-2)
        dz_gdf["z_score"] = dz_gdf["z_score"].clip(upper=2)
        fig, ax1 = plt.subplots(nrows=1, ncols=1)
        plot_geospatial_heatmap_with_event_locs(geo_df=dz_gdf, col_name="z_score", events_data=sites_damaged, title=None, cmap=cm.seismic, cmap_type="seismic", marker_color=None, events_data_type="locations_points", needs_colormapping=False, add_title=False, event_locs_included=True, include_colorbar=True, with_streetmap=True, ax=ax1)
        plt.show()

    code.interact(local=locals())
    return None

def track_unitar_sites(df):
    dk = df.copy()
    #---------Tracking between Dec 12 and May 15--------------------
    # df = df[(df.date_c>="2014-12-12") & (df.date_c<="2015-05-15")]
    gdf = create_geodataframe(df[["id","Latitude","Longitude"]].drop_duplicates(), radius=462, cap_style=3)

    #------UNOSAT Data----------------------------------------------
    # read damaged structures dataset released by UNITAR
    sites_city = gpd.read_file("../extra_datasets/unitar_unisat_data/Damage_Sites_Sanaa_City_20150910.shp")
    sites_airport = gpd.read_file("../extra_datasets/unitar_unisat_data/Damage_Sites_Sanaa_Airport_20150515.shp")
    sites_damaged = pd.concat([sites_city, sites_airport]).reset_index()
    sites_damaged = sites_damaged.drop(columns = {"index"})
    sites_damaged = sites_damaged[(sites_damaged.SiteID != "Transport Vehicle")]
    sites_damaged.SiteID = sites_damaged.SiteID.apply(lambda x: "Building" if x=="Building (General / Default)" else "Building" if x=="Greenhouse" else "Building" if x=="Industrial Facility" else x)

    #-------combine UNOSAT and NL data-----------------------------
    gdf["damaged_buildings"] = gdf["geometry"].apply(lambda x: len(list(filter(x.contains, sites_damaged[(sites_damaged.SiteID == "Building")].geometry.values))))
    gdf["damaged_roads"] = gdf["geometry"].apply(lambda x: len(list(filter(x.contains, sites_damaged[(sites_damaged.SiteID == "Road")].geometry.values))))
    gdf["damaged_fields"] = gdf["geometry"].apply(lambda x: len(list(filter(x.contains, sites_damaged[(sites_damaged.SiteID == "Field")].geometry.values))))

    #---combine dataframe with it's trend data and damaged buildings data---------
    df = dk[(dk.date_c>="2015-03-20") & (dk.date_c<="2015-05-15")]

    ds = pd.read_hdf("STL_and_bandpass_yemen_updated.h5", key="zeal")
    df = pd.merge(df, ds[["id","date_c","trend"]], left_on=["id","date_c"], right_on=["id","date_c"], how="left")
    df = pd.merge(df, gdf[["id","damaged_buildings","geometry"]], left_on=["id"], right_on=["id"], how="left")

    #-------------------Obtain list of events that happened during the said interval------
    de_locs = query_event_locations_by_monthyear([3,4,5],[2015],buffered=False,radius=None,cap_style=None)
    de_locs = de_locs[(de_locs.event_date<="2015-05-15") & (de_locs.event_date>="2015-03-26")]

    # de_locs = query_event_locations_by_monthyear([1,2,3,4,5],[2015],buffered=False,radius=None,cap_style=None)
    # de_locs = de_locs[(de_locs.event_date<="2015-05-15")]

    de_locs = de_locs.sort_values(by="event_date")
    print(de_locs)
    de = de_locs.groupby("event_date").count()[["location"]].reset_index()
    de_locs = de_locs.drop(columns = ["Latitude","Longitude","event_id_cnty","geometry"])

    #=============================================================================================================
    #-------------------calculate changes in radiance-----------------------------
    # df = df.sort_values(by=["id","date_c"])
    # df["rad_past"] = df.groupby("id")["RadE9_Mult_Nadir_Norm"].shift(1)
    # df["perc_rad"] = (df["RadE9_Mult_Nadir_Norm"] - df["rad_past"])*100.0/df["rad_past"]

    # #-------------------calculate changes respective to Jan 2015----------------
    df = df.sort_values(by=["id","date_c"])
    ddec = dk[(dk.date_c.dt.month == 12) & (dk.date_c.dt.year == 2014)].groupby("id").mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()
    ddec = ddec.rename(columns={"RadE9_Mult_Nadir_Norm":"rad_past"})

    df = pd.merge(df, ddec[["id","rad_past"]], left_on=["id"], right_on=["id"], how="left")
    df["perc_rad"] = (df["RadE9_Mult_Nadir_Norm"] - df["rad_past"])*100.0/df["rad_past"]

    #=============================================================================================================

    #------------------ Analyze drops in radiance over time---------------------------------
    # dp = df[["id","date_c","perc_rad","damaged_buildings"]]
    # dp["build_status"] = dp["damaged_buildings"].apply(lambda x: 0 if x==0 else 1)

    # # #Reference: https://stackoverflow.com/questions/40978059/min-and-max-row-from-pandas-groupby
    # dp_rad = dp.loc[dp.groupby(["id","build_status"]).perc_rad.idxmin()]
    # dg_rad = dp_rad.groupby(["date_c","build_status"]).count()[["id"]].reset_index()
    # dg_rad = pd.merge(dp[["date_c","build_status"]].drop_duplicates(), dg_rad, left_on=["date_c","build_status"], right_on=["date_c","build_status"], how="left")
    # dg_rad = dg_rad.fillna(0)
    # dg_rad = dg_rad.sort_values(by=["date_c","build_status"])
    # dg_rad.set_index("date_c",inplace=True)

    #------------------ Analyze FIRST TIME drastic drops in radiance over time--------------
    dp = df[["id","date_c","perc_rad","damaged_buildings"]]
    dp["build_status"] = dp["damaged_buildings"].apply(lambda x: 0 if x==0 else 1)

    #Reference: https://stackoverflow.com/questions/40978059/min-and-max-row-from-pandas-groupby
    dp_rad = dp.loc[dp[(dp.perc_rad <= -90)].groupby(["id","build_status"]).date_c.idxmin()]
    # code.interact(local=locals())
    dg_rad = dp_rad.groupby(["date_c","build_status"]).count()[["id"]].reset_index()
    dg_rad = pd.merge(dp[["date_c","build_status"]].drop_duplicates(), dg_rad, left_on=["date_c","build_status"], right_on=["date_c","build_status"], how="left")
    dg_rad = dg_rad.fillna(0)
    dg_rad = dg_rad.sort_values(by=["date_c","build_status"])
    dg_rad.set_index("date_c",inplace=True)

    #----------------------Visualization of the above analysis------------------------
    sites0 = dp[(dp.build_status == 0)].id.nunique()
    sites1 = dp[(dp.build_status == 1)].id.nunique()

    dgr0 = dg_rad[(dg_rad.build_status == 0)].reset_index()
    dgr1 = dg_rad[(dg_rad.build_status == 1)].reset_index()

    dgr0["id"] = dgr0["id"].cumsum()
    dgr1["id"] = dgr1["id"].cumsum()

    dgr0["pixel_prop"] = dgr0["id"]*100/sites0
    dgr1["pixel_prop"] = dgr1["id"]*100/sites1

    # code.interact(local=locals())

    fig, ax = plt.subplots()
    ax.plot(dgr0.date_c, dgr0.pixel_prop, zorder=0, marker="*", markersize=4, label="Not containing a damage site (3087 pixels)")
    ax.plot(dgr1.date_c, dgr1.pixel_prop, zorder=1, color="red", marker="s", markersize=4, label="Containing atleast one damage site (198 pixels)")
    ax.vlines(de.event_date, ymin=0, ymax=100, linestyles='dashed', alpha=0.4)
    ax.set_xticks(de.event_date)
    ax.set_xticklabels(de.event_date.dt.strftime("%d-%b-%y"), rotation=90, ha="center")
    ax.set_ylabel("Proportion of pixels (%)")
    ax.set_xlabel("Date")
    ax.set_ylim(0,100)
    ax.legend(loc="upper left")
    # plt.title("Proportion of pixels that experienced a sudden drop in radiance (<-90%) relative to the baseline for the first time.")
    plt.tight_layout()
    plt.show()

    # #-------------------------Verification------------------------------------------------
    dv = dp.loc[dp[(dp.perc_rad <= -90)].groupby(["id","build_status"]).date_c.idxmin()]
    da = pd.read_hdf("yemen_grid_with_admin.h5",key="zeal")
    dv = pd.merge(dv, da, left_on=["id"], right_on=["id"], how="left")
    dv = pd.merge(dv, de_locs[["event_date","sub_event_type","admin2"]], left_on=["date_c"], right_on=["event_date"], how="left")
    dv["adm2"] = dv["adm2"].apply(lambda x: "Maain" if x=="Ma'ain" else "Aththaorah" if x=="Ath'thaorah" else "Shuaub" if x=="Shu'aub" else "Assafiyah" if x=="Assafiyah" else "Azzal" if x=="Az'zal" else x)

    dv1 = dv.groupby("date_c").adm2.apply(set).reset_index()
    dv1 = dv1.rename(columns={"adm2":"detected_adm2"})
    # dv1["detected_adm2"] = dv1["detected_adm2"].apply(lambda x: np.unique(x))

    dv2 = dv.groupby("date_c").admin2.apply(set).reset_index()
    dv2 = dv2.rename(columns={"admin2":"actual_adm2"})
    # dv2["actual_adm2"] = dv2["actual_adm2"].apply(lambda x: np.unique(x))

    dvf = pd.merge(dv1, dv2, left_on=["date_c"], right_on=["date_c"], how="left")
    dvf["missed_detec"] = dvf["actual_adm2"] - dvf["detected_adm2"]
    dvf["misses"] = dvf["missed_detec"].apply(lambda x: len(x))
    dvf["actual"] = dvf["actual_adm2"].apply(lambda x: len(x))
    dvf["error"] = dvf["misses"]*100/dvf["actual"]
    doe = dvf[(dvf.actual_adm2 != {np.nan})]

    code.interact(local=locals())
    return None

def track_sites_for_march_to_may2015(df):
    dk = df.copy()
    #---------Tracking between Dec 12 and May 15--------------------
    # df = df[(df.date_c>="2014-12-12") & (df.date_c<="2015-05-15")]
    gdf = create_geodataframe(df[["id","Latitude","Longitude"]].drop_duplicates(), radius=462, cap_style=3)

    #------UNOSAT Data----------------------------------------------
    # read damaged structures dataset released by UNITAR
    sites_city = gpd.read_file("../extra_datasets/unitar_unisat_data/Damage_Sites_Sanaa_City_20150910.shp")
    sites_airport = gpd.read_file("../extra_datasets/unitar_unisat_data/Damage_Sites_Sanaa_Airport_20150515.shp")
    sites_damaged = pd.concat([sites_city, sites_airport]).reset_index()
    sites_damaged = sites_damaged.drop(columns = {"index"})
    sites_damaged = sites_damaged[(sites_damaged.SiteID != "Transport Vehicle")]
    sites_damaged.SiteID = sites_damaged.SiteID.apply(lambda x: "Building" if x=="Building (General / Default)" else "Building" if x=="Greenhouse" else "Building" if x=="Industrial Facility" else x)

    #-------combine UNOSAT and NL data-----------------------------
    gdf["damaged_buildings"] = gdf["geometry"].apply(lambda x: len(list(filter(x.contains, sites_damaged[(sites_damaged.SiteID == "Building")].geometry.values))))
    gdf["damaged_roads"] = gdf["geometry"].apply(lambda x: len(list(filter(x.contains, sites_damaged[(sites_damaged.SiteID == "Road")].geometry.values))))
    gdf["damaged_fields"] = gdf["geometry"].apply(lambda x: len(list(filter(x.contains, sites_damaged[(sites_damaged.SiteID == "Field")].geometry.values))))

    #---combine dataframe with it's damaged buildings data---------
    df = dk[(dk.date_c>="2015-03-20") & (dk.date_c<="2015-05-15")]
    df = pd.merge(df, gdf[["id","damaged_buildings","geometry"]], left_on=["id"], right_on=["id"], how="left")

    #-------------------Obtain list of events that happened during the said interval------
    de_locs = query_event_locations_by_monthyear([3,4,5],[2015],buffered=False,radius=None,cap_style=None)
    de_locs = de_locs[(de_locs.event_date<="2015-05-15") & (de_locs.event_date>="2015-03-15") & (de_locs.sub_event_type=="Air/drone strike")]
    de_locs = de_locs.sort_values(by="event_date")
    print(de_locs)
    de = de_locs.groupby("event_date").count()[["location"]].reset_index()
    de_locs = de_locs.drop(columns = ["Latitude","Longitude","event_id_cnty","geometry"])

    #=============================================================================================================
    #-------------------calculate changes in radiance-----------------------------
    # df = df.sort_values(by=["id","date_c"])
    # df["rad_past"] = df.groupby("id")["RadE9_Mult_Nadir_Norm"].shift(1)
    # df["perc_rad"] = (df["RadE9_Mult_Nadir_Norm"] - df["rad_past"])*100.0/df["rad_past"]

    # #-------------------calculate changes respective to March 26, 2015----------------
    df = df.sort_values(by=["id","date_c"])
    dmar = zscore_percchange_march2015(dk.copy(), create_plot=False)
    dmar = dmar.rename(columns={"mean_base":"rad_past"})

    df = pd.merge(df, dmar[["id","rad_past"]], left_on=["id"], right_on=["id"], how="left")
    df["perc_rad"] = (df["RadE9_Mult_Nadir_Norm"] - df["rad_past"])*100.0/df["rad_past"]

    #=============================================================================================================

    #------------------ Analyze drops in radiance over time---------------------------------
    # dp = df[["id","date_c","perc_rad","damaged_buildings"]]
    # dp["build_status"] = dp["damaged_buildings"].apply(lambda x: 0 if x==0 else 1)

    # # #Reference: https://stackoverflow.com/questions/40978059/min-and-max-row-from-pandas-groupby
    # dp_rad = dp.loc[dp.groupby(["id","build_status"]).perc_rad.idxmin()]
    # dg_rad = dp_rad.groupby(["date_c","build_status"]).count()[["id"]].reset_index()
    # dg_rad = pd.merge(dp[["date_c","build_status"]].drop_duplicates(), dg_rad, left_on=["date_c","build_status"], right_on=["date_c","build_status"], how="left")
    # dg_rad = dg_rad.fillna(0)
    # dg_rad = dg_rad.sort_values(by=["date_c","build_status"])
    # dg_rad.set_index("date_c",inplace=True)

    #------------------ Analyze FIRST TIME drastic drops in radiance over time--------------
    dp = df[["id","date_c","perc_rad","damaged_buildings"]]
    dp["build_status"] = dp["damaged_buildings"].apply(lambda x: 0 if x==0 else 1)

    #Reference: https://stackoverflow.com/questions/40978059/min-and-max-row-from-pandas-groupby
    dp_rad = dp.loc[dp[(dp.perc_rad <= -92)].groupby(["id","build_status"]).date_c.idxmin()]
    # code.interact(local=locals())
    dg_rad = dp_rad.groupby(["date_c","build_status"]).count()[["id"]].reset_index()
    dg_rad = pd.merge(dp[["date_c","build_status"]].drop_duplicates(), dg_rad, left_on=["date_c","build_status"], right_on=["date_c","build_status"], how="left")
    dg_rad = dg_rad.fillna(0)
    dg_rad = dg_rad.sort_values(by=["date_c","build_status"])
    dg_rad.set_index("date_c",inplace=True)

    #----------------------Visualization of the above analysis------------------------
    sites0 = dp[(dp.build_status == 0)].id.nunique()
    sites1 = dp[(dp.build_status == 1)].id.nunique()

    dgr0 = dg_rad[(dg_rad.build_status == 0)].reset_index()
    dgr1 = dg_rad[(dg_rad.build_status == 1)].reset_index()

    dgr0["id"] = dgr0["id"].cumsum()
    dgr1["id"] = dgr1["id"].cumsum()

    dgr0["pixel_prop"] = dgr0["id"]*100/sites0
    dgr1["pixel_prop"] = dgr1["id"]*100/sites1

    dgr0["pixel_prop"] = dgr0["pixel_prop"].clip(upper=100) #to avoid issues caused at decimal level in calculations
    dgr1["pixel_prop"] = dgr1["pixel_prop"].clip(upper=100)
    # code.interact(local=locals())

    fig, ax = plt.subplots()
    ax.plot(dgr0.date_c, dgr0.pixel_prop, zorder=0, marker="*", markersize=4, label="Not containing a damage site (3087 pixels)")
    ax.plot(dgr1.date_c, dgr1.pixel_prop, zorder=1, color="red", marker="s", markersize=4, label="Containing atleast one damage site (198 pixels)")
    ax.vlines(de.event_date, ymin=0, ymax=100, linestyles='dashed', alpha=0.4)
    ax.set_xticks(de.event_date)
    ax.set_xticklabels(de.event_date.dt.strftime("%d-%b-%y"), rotation=90, ha="center")
    ax.set_ylabel("Proportion of pixels (%)")
    ax.set_xlabel("Date")
    ax.set_ylim(0,101)
    ax.legend(loc="upper left")
    # plt.title("Proportion of pixels that experienced a sudden drop in radiance (<-90%) relative to the baseline for the first time.")
    plt.tight_layout()
    plt.show()

    # # #-------------------------Verification------------------------------------------------
    # dv = dp.loc[dp[(dp.perc_rad <= -90)].groupby(["id","build_status"]).date_c.idxmin()]
    # da = pd.read_hdf("yemen_grid_with_admin.h5",key="zeal")
    # dv = pd.merge(dv, da, left_on=["id"], right_on=["id"], how="left")
    # dv = pd.merge(dv, de_locs[["event_date","sub_event_type","admin2"]], left_on=["date_c"], right_on=["event_date"], how="left")
    # dv["adm2"] = dv["adm2"].apply(lambda x: "Maain" if x=="Ma'ain" else "Aththaorah" if x=="Ath'thaorah" else "Shuaub" if x=="Shu'aub" else "Assafiyah" if x=="Assafiyah" else "Azzal" if x=="Az'zal" else x)

    # dv1 = dv.groupby("date_c").adm2.apply(set).reset_index()
    # dv1 = dv1.rename(columns={"adm2":"detected_adm2"})

    # dv2 = dv.groupby("date_c").admin2.apply(set).reset_index()
    # dv2 = dv2.rename(columns={"admin2":"actual_adm2"})

    # dvf = pd.merge(dv1, dv2, left_on=["date_c"], right_on=["date_c"], how="left")
    # dvf["missed_detec"] = dvf["actual_adm2"] - dvf["detected_adm2"]
    # dvf["misses"] = dvf["missed_detec"].apply(lambda x: len(x))
    # dvf["actual"] = dvf["actual_adm2"].apply(lambda x: len(x))
    # dvf["error"] = dvf["misses"]*100/dvf["actual"]
    # doe = dvf[(dvf.actual_adm2 != {np.nan})]

    code.interact(local=locals())
    return None

def admin2_TNL(dk, da):
    df = pd.merge(dk, da[["id","adm2","adm1"]].drop_duplicates(), left_on=["id"], right_on=["id"], how="left")
    df = df[(df.date_c > "2015-01-01") & (df.date_c < "2015-05-15")]
    df = df.groupby(["adm2","date_c"]).mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()
    df["adm2"] = df["adm2"].apply(lambda x: "Maain" if x=="Ma'ain" else "Aththaorah" if x=="Ath'thaorah" else "Shuaub" if x=="Shu'aub" else "Assafiyah" if x=="Assafi'yah" else "Azzal" if x=="Az'zal" else x)

    #-------------------Obtain list of events that happened during the said interval------
    de_locs = query_event_locations_by_monthyear([1,2,3,4,5],[2015],buffered=False,radius=None,cap_style=None)
    de_locs = de_locs[(de_locs.event_date<="2015-05-15")]
    de_locs = de_locs.sort_values(by="event_date")
    de_locs = de_locs.groupby(["admin2","event_date"]).count()[["data_id"]].reset_index()

    #------------------Combine admin2 TNL and event data----------------------------------
    dm = pd.merge(df, de_locs[["data_id","admin2","event_date"]].drop_duplicates(), left_on=["adm2","date_c"], right_on=["admin2","event_date"], how="left")

    # df.set_index("date_c", inplace=True)
    # df.groupby("adm2").RadE9_Mult_Nadir_Norm.plot(legend=True)
    # plt.show()

    code.interact(local=locals())
    return None

def march2015_and_infra2(df, di):
    """
    Information obtained in this function:
    (1) Basic population based grouping information on infrastructure and radiance
    (2) Generic damage information (how many roads, etc. were present in low zscore cells)
    (3) Structured damage information summary (table with groups of population density and impact of damage categories)
    """
    #-------------------Obtain list of events that happened during the said interval------
    # de_locs = query_event_locations_by_monthyear([1,2,3,4,5],[2015],buffered=False,radius=None,cap_style=None)
    # de_locs = de_locs[(de_locs.event_date=="2015-03-26")]
    # de_locs = de_locs.sort_values(by="event_date")
    # # de_locs = de_locs.groupby(["admin2","event_date"]).count()[["data_id"]].reset_index()

    #--------obtain baseline and crisis radiance and percentage change/z-score-----------------
    dz = zscore_percchange_march2015(df, create_plot="False")

    #---------combine zscore and infra+pop datasets------------------------------------
    dc = pd.merge(dz, di, left_on=["id"], right_on=["id"], how="left")


    #---------Basic assessments comparing pop groups----------------------------------------------
    # dc_pop_basic = dc.groupby("j_group").agg({"ed_sites":"sum","health_sites":"sum","total_roads":"sum","total_streets":"sum","builtup_area":"sum","total_pop":"sum","mean_base":"mean"})
    # print("*********************************************************")
    # print("Mean Base Mean")
    # print(dc_pop_basic)

    # dc_pop_tnl = dc.groupby("j_group").sum()[["RadE9_Mult_Nadir_Norm","mean_base"]].reset_index()
    # dc_pop_tnl["perc_change"] = (dc_pop_tnl["RadE9_Mult_Nadir_Norm"] - dc_pop_tnl["mean_base"])*100.0/dc_pop_tnl["mean_base"]
    # print("*********************************************************")
    # print("TNL Change per j_group")
    # print(dc_pop_tnl)

    # #--------------------Infra Damage assessment---------------------------------------
    # dm = dc.copy()
    # dm = dm[["id","ed_sites","health_sites","which_roads","which_streets","builtup_area","total_pop","z_score","perc_change","RadE9_Mult_Nadir_Norm","mean_base"]]
    # dm["z_group"] = dm.z_score.apply(lambda x: "severe" if x<=-2 else "moderate" if -2<x<=-1 else "low" if -1<x<0 else "increase")
    # dmz = dm.groupby("z_group").agg({"ed_sites":"sum","health_sites":"sum","builtup_area":"sum","total_pop":"sum","RadE9_Mult_Nadir_Norm":"sum","mean_base":"sum","id":"count"})
    # # .sum()[["ed_sites","health_sites","builtup_area","total_pop","RadE9_Mult_Nadir_Norm","mean_base"]]
    # dmz["tnl_change"] = (dmz["RadE9_Mult_Nadir_Norm"] - dmz["mean_base"])*100.0/dmz["mean_base"]
    # dmz["perc_cells"] = dmz["id"]*100.0/3285

    # droads = dm.groupby("z_group")["which_roads"].apply(lambda x: np.unique(list(itertools.chain.from_iterable(x)))).reset_index()
    # droads["total_roads"] = droads["which_roads"].apply(lambda x: len(x))
    # dstreets = dm.groupby("z_group")["which_streets"].apply(lambda x: np.unique(list(itertools.chain.from_iterable(x)))).reset_index()
    # dstreets["total_streets"] = dstreets["which_streets"].apply(lambda x: len(x))

    # dmz = pd.merge(dmz, droads, on=["z_group"], how="left")
    # dmz = pd.merge(dmz, dstreets, on=["z_group"], how="left")
    # dmz = dmz.drop(columns=["which_streets","which_roads"])

    # total_roads = len(np.unique(dm["which_roads"].sum()))
    # total_streets = len(np.unique(dm["which_streets"].sum()))

    # #calculate percentages of each infra type
    # dm2 = dmz.copy()
    # dm2["ed_sites"] = dm2["ed_sites"]*100.0/dm2["ed_sites"].sum()
    # dm2["health_sites"] = dm2["health_sites"]*100.0/dm2["health_sites"].sum()
    # dm2["builtup_area"] = dm2["builtup_area"]*100.0/dm2["builtup_area"].sum()
    # dm2["total_pop"] = dm2["total_pop"]*100.0/dm2["total_pop"].sum()
    # dm2["total_roads"] = dm2["total_roads"]*100.0/total_roads
    # dm2["total_streets"] = dm2["total_streets"]*100.0/total_streets
    # dm2["perc_cells"] = dm2["id"]*100.0/3285

    #---------Structured damage information summary table----------------------------------
    # Each row will represent a pop category/baseline rad category since both are the same
    # Each column will represent Impact of damage:
    #  severe decrease(<-50%), moderate decrease(-50 to -25), slight decrease (-25 to 0), increase (> 0)
    ds = dc[["id","perc_change","z_score","j_group"]]
    # ds["damage_group"] = ds.perc_change.apply(lambda x: "High" if x<-50 else "moderate" if -50<=x<-25 else "slight" if -25<=x<0 else "increase")
    ds["damage_group"] = ds.z_score.apply(lambda x: "severe" if x<=-2 else "moderate" if -2<x<=-1 else "low" if -1<x<0 else "increase")
    da = pd.read_hdf("yemen_grid_with_admin.h5", key="zeal")
    ds = pd.merge(ds, da[["id","adm2"]], left_on=["id"], right_on=["id"], how="left")

    # code.interact(local=locals())

    ds = ds.groupby(["j_group","damage_group"]).count()[["id"]].reset_index()
    ds = ds.rename(columns = {"id":"sites"})
    ds_totals = ds.groupby(["j_group"]).sum()[["sites"]].reset_index()
    ds_totals = ds_totals.rename(columns = {"sites":"total_sites"})
    ds = pd.merge(ds, ds_totals, left_on=["j_group"], right_on=["j_group"], how="left")
    ds["perc_cells"] = ds["sites"] * 100.0/ds["total_sites"]
    print("*********************************************************")
    print("Structured Damage Summary:")
    print("*********************************************************")
    print("{}".format(ds))

    #---------Visualize pop based grouping---------------------------------------------------
    # dv = dc[["id","Latitude_x","Longitude_x","RadE9_Mult_Nadir_Norm","j_group"]]
    # dv.columns = ["id","Latitude","Longitude","RadE9_Mult_Nadir_Norm","j_group"]
    # dv_gdf = create_geodataframe(dv, radius=462, cap_style=3, buffered=True)

    # # fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2)
    # # plot_geospatial_heatmap_with_event_locs(geo_df=dv_gdf, col_name="RadE9_Mult_Nadir_Norm", events_data=None, title=None, cmap=cm.hot, cmap_type="hot", needs_colormapping=True, marker_color="black", events_data_type="locations_buffered", add_title=False, event_locs_included=False, include_colorbar=True, ax=ax1)

    # fig, ax2 = plt.subplots(nrows=1, ncols=1)
    # plot_geospatial_heatmap_with_event_locs(geo_df=dv_gdf, col_name="j_group", events_data=None, title=None, cmap=cm.Set1, cmap_type="Set1", needs_colormapping=False, marker_color="black", events_data_type="locations_buffered", add_title=False, with_streetmap=True, event_locs_included=False, include_colorbar=True, ax=ax2)
    # plt.show()

    #---------check if we can find shapefile of areas for yemen------------------------------

    code.interact(local=locals())
    return None




if __name__=='__main__':
    #--------read & process necessary datasets----------
    # dg = pd.read_hdf("filtered_data.h5", key="zeal")
    # #clip negative radiance values to 0
    # dg["RadE9_Mult_Nadir_Norm"] = dg["RadE9_Mult_Nadir_Norm"].clip(lower=0)
    # # combine multiple readings for same day and same id
    # dg = dg.groupby(["id","Latitude","Longitude","date_c"]).mean()[["RadE9_Mult_Nadir_Norm","LI"]].reset_index()

    #---------read lunar corrected dataset--------------
    data_path = "./"
    dg = pd.read_hdf(data_path + "yemen_STL_lunar_corrected.h5", key="zeal")
    dg = dg.drop(columns = ["RadE9_Mult_Nadir_Norm"])
    dg = dg.rename(columns={"rad_corr":"RadE9_Mult_Nadir_Norm"})

    #---------Associate areas in Sanaa with NL grid cells----------
    # output has been pickled
    # da = associate_NLgrid_with_areas(dg[["id","Latitude","Longitude"]].drop_duplicates())
    da = pd.read_hdf("yemen_grid_with_admin.h5",key="zeal")

    #query_event_locations_by_monthyear([3,4],[2015],buffered=False,radius=None,cap_style=None)
    #-------------create baseline for March 2015 event-------------
    # baseline_db_march2015(dg, baseline_date="2015-03-26", baseline_year=2015, include_exact_baseline_date=False)

    #-------visually compare radiance on March 26 and March 27-----
    # visually_compare_March_26_27_radiance(dg.copy())

    #-------------calculate perc change and zscore for March 2015 event----
    dz = zscore_percchange_march2015(dg, create_plot=False)
    # now calculate z-score changes inside the bombing zones and outside them
    # calculate_zscore_changes_march2015(dz)

    #-------------2015 baseline and infrastructure---------"
    extra_data_path = "../extra_datasets/"
    di = pd.read_pickle(extra_data_path + "yemen_infra_pop_data_combined_2.pck")
    dbase = pd.read_hdf("yemen_groups.h5",key="zeal")
    march2015_and_infra(dg.copy(), di.copy())
    # march2015_and_infra2(dg.copy(), dbase.copy())

    #-------------verification using UNITAR--------------------------------
    # verify_using_unitar_2015(dg.copy(), create_plot=False)

    #-------------track unitar sites---------------------------------------
    # track_unitar_sites(dg.copy())

    #--------------track sites from march to may---------------------------
    # track_sites_for_march_to_may2015(dg.copy())

    #-----------track TNL changes at admin level---------------------------
    # admin2_TNL(dg.copy(), da.copy())