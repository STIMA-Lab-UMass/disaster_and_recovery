"""
This script is used to produce all the plots and analysis for recovery tracking section's Disaster mapping portion.
This script is specifically focused on recovery mapping and analysis post March 26, 2015.
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
import ruptures as rpt
import statsmodels.api as sm
import jenkspy

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
        fig, ax1 = plt.subplots()
        plot_geospatial_heatmap_with_event_locs(geo_df=dz_gdf, col_name="z_score", events_data=de_26_buffered, title=None, cmap=cm.seismic, cmap_type="seismic", marker_color=None, events_data_type="locations_points", needs_colormapping=False, add_title=False, event_locs_included=True, include_colorbar=True, with_streetmap=True, ax=ax1)
        plt.show()
    # code.interact(local=locals())
    return dz_output

def march2015_and_infra(df, di):
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
    # dc["pop_group"] = dc["total_pop"].apply(lambda x: "Very low" if x<=20 else "Low" if 20<x<=600 else "Medium" if 600<x<=2000 else "High")

    dc["pop_group"] = dc["total_pop"].apply(lambda x: "Very low" if x<=20 else "Low" if 20<x<=600 else "Medium" if 600<x<=3000 else "High")
    # dc["pop_group"] = dc["total_pop"].apply(lambda x: "Very low" if x<=300 else "Low" if 300<x<=800 else "Medium" if 800<x<=3000 else "High")

    #--------------------Infra Damage assessment---------------------------------------
    # dm = dc.copy()
    # dm = dm[["id","ed_sites","health_sites","which_roads","which_streets","builtup_area","total_pop","z_score","perc_change","RadE9_Mult_Nadir_Norm","mean_base"]]
    # dm["z_group"] = dm.z_score.apply(lambda x: "severe" if x<=-2 else "moderate" if -2<x<=-1 else "low" if -1<x<0 else "increase")
    # dmz = dm.groupby("z_group").sum()[["ed_sites","health_sites","builtup_area","total_pop","RadE9_Mult_Nadir_Norm","mean_base"]]
    # dmz["tnl_change"] = (dmz["RadE9_Mult_Nadir_Norm"] - dmz["mean_base"])*100.0/dmz["mean_base"]

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

def precrisis_baseline_data(df, di):
    #--------obtain baseline and crisis radiance and percentage change/z-score-----------------
    dz = zscore_percchange_march2015(df, create_plot="False")

    #---------combine zscore and infra+pop datasets------------------------------------
    dc = pd.merge(dz, di, left_on=["id"], right_on=["id"], how="left")
    # classify each cell by population
    dc["pop_group"] = dc["total_pop"].apply(lambda x: "Very low" if x<=20 else "Low" if 20<x<=600 else "Medium" if 600<x<=3000 else "High")
    # classify each cell by damage
    dc["damage_group"] = dc.z_score.apply(lambda x: "severe" if x<=-2 else "moderate" if -2<x<=-1 else "low" if -1<x<0 else "increase")
    # create groups by pre-radiance levels
    dc["prerad_group"] = dc.mean_base.apply(lambda x: "Very Low" if x<15 else "Low" if 15<=x<50 else "Medium" if 50<=x<100 else "High")
    # add admin2 data to the table
    da = pd.read_hdf("yemen_grid_with_admin.h5",key="zeal")
    dc = pd.merge(dc, da[["id","adm2"]], left_on=["id"], right_on=["id"], how="left")
    # dc = dc.drop(columns=["which_roads","which_streets","total_roads","total_streets","land_area","builtup_area","Longitude_y","Latitude_y"])
    # code.interact(local=locals())
    return dc

def postcrisis_holiday_lighting_correction(df):
    """
    2015: June 17 to July 16
    2016: June 6 to July 5
    2017: May 26 to June 24
    2018: May 16 to June 14
    2019: May 5 to June 3

    So we can basically consider months of April and August for correction purposes.
    """
    ddates = df[["date_c"]].drop_duplicates()
    ddates["holiday"] = ddates["date_c"].apply(lambda x: True if (pd.to_datetime("2015-06-17",format="%Y-%m-%d")<=x.date()<=pd.to_datetime("2015-07-16",format="%Y-%m-%d")) | (pd.to_datetime("2016-06-06",format="%Y-%m-%d")<=x.date()<=pd.to_datetime("2016-07-05",format="%Y-%m-%d"))| (pd.to_datetime("2017-05-26",format="%Y-%m-%d")<=x.date()<=pd.to_datetime("2017-06-24",format="%Y-%m-%d"))| (pd.to_datetime("2018-05-16",format="%Y-%m-%d")<=x.date()<=pd.to_datetime("2018-06-14",format="%Y-%m-%d")) | (pd.to_datetime("2019-05-05",format="%Y-%m-%d")<=x.date()<=pd.to_datetime("2019-06-03",format="%Y-%m-%d")) else False)
    ddates["correction_dates"] = ddates["date_c"].apply(lambda x: True if (pd.to_datetime("2015-05-17",format="%Y-%m-%d")<=x.date()<=pd.to_datetime("2015-06-16",format="%Y-%m-%d")) | (pd.to_datetime("2015-07-17",format="%Y-%m-%d")<=x.date()<=pd.to_datetime("2015-08-16",format="%Y-%m-%d")) |
                                                        (pd.to_datetime("2016-05-06",format="%Y-%m-%d")<=x.date()<=pd.to_datetime("2016-06-05",format="%Y-%m-%d")) | (pd.to_datetime("2016-07-06",format="%Y-%m-%d")<=x.date()<=pd.to_datetime("2016-08-05",format="%Y-%m-%d")) |
                                                        (pd.to_datetime("2017-04-26",format="%Y-%m-%d")<=x.date()<=pd.to_datetime("2017-05-25",format="%Y-%m-%d")) | (pd.to_datetime("2017-06-25",format="%Y-%m-%d")<=x.date()<=pd.to_datetime("2017-07-24",format="%Y-%m-%d")) |
                                                        (pd.to_datetime("2018-04-16",format="%Y-%m-%d")<=x.date()<=pd.to_datetime("2018-05-15",format="%Y-%m-%d")) | (pd.to_datetime("2018-06-15",format="%Y-%m-%d")<=x.date()<=pd.to_datetime("2018-07-14",format="%Y-%m-%d")) |
                                                        (pd.to_datetime("2019-04-05",format="%Y-%m-%d")<=x.date()<=pd.to_datetime("2019-05-04",format="%Y-%m-%d")) | (pd.to_datetime("2019-06-04",format="%Y-%m-%d")<=x.date()<=pd.to_datetime("2019-07-03",format="%Y-%m-%d")) else False
                                                        )


    df = pd.merge(df, ddates, left_on=["date_c"], right_on=["date_c"], how="left")
    df["year"] = df.date_c.dt.year
    dz = df[(df.correction_dates == True)].groupby(["id","year"]).agg({"RadE9_Mult_Nadir_Norm":["mean","std"]}).reset_index()
    dz.columns = ["id","year","mean_base","std_base"]

    dh = pd.merge(df, dz, left_on=["id","year"], right_on=["id","year"], how="left")
    dh = dh[(dh.holiday==True)]
    dh["hol_zscore"] = (dh["RadE9_Mult_Nadir_Norm"] - dh["mean_base"])/dh["std_base"]

    df = pd.merge(df, dh[["id","date_c","hol_zscore"]], left_on=["id","date_c"], right_on=["id","date_c"], how="left")
    df["rad_corr"] = np.where((df.holiday == True) & (df.hol_zscore > 1) & (df.date_c.dt.year>=2015), np.nan, df["RadE9_Mult_Nadir_Norm"])
    df.set_index("date_c", inplace=True)
    df["rad_corr"] = df.groupby(["id"])["rad_corr"].apply(lambda x: x.interpolate(method="time"))
    df = df.reset_index()

    # dk1 = df[(df.id == "sanaa_grid_0_6") & (df.date_c.dt.year == 2013)]
    # dk1 = df[(df.id == "sanaa_grid_20_22") & (df.date_c.dt.year >= 2015)]
    # plt.plot(dk1.date_c, dk1.RadE9_Mult_Nadir_Norm, label="original", c="red")
    # plt.plot(dk1.date_c, dk1.rad_corr, label="corrected", c="blue")
    # plt.title("Holiday Lighting Correction")
    # plt.legend()
    # plt.show()
    # code.interact(local=locals())
    return df

def detect_and_measure_outage_duration(df, dbase):
    # df_out = df[(df.date_c>="2015-03-26")]
    df_out = df.copy()
    groupby_field = "j_group"

    #---------Measuring percent outages every month per cell------------------------------------------------------
    df_out["outage"] = df_out["rad_corr"].apply(lambda x: True if x<1 else False)
    # the following three lines calculate yearly outage rate (%) associated with a single pixel
    df_meas = df_out.groupby(["id", pd.Grouper(key="date_c",freq="1M")]).agg({"LI":"count", "outage":"sum"}).reset_index()
    df_meas.columns = ["id","date_c","total_readings","outage_count"]
    df_meas["perc_outage"] = df_meas["outage_count"]*100.0/df_meas["total_readings"]

    #--------------Per pixel outages per year--------------------------------------------------------------------
    fontsize = 14
    figsize = (10,4)
    db = dbase[["id", groupby_field]].drop_duplicates()
    df_agg = pd.merge(df_meas, db, left_on=["id"], right_on=["id"])
    df_agg = df_agg.groupby([groupby_field, "date_c"]).agg({"id":"count","outage_count":"sum","perc_outage":"mean"}).reset_index()
    fig, ax = plt.subplots(figsize=figsize)
    markers = ["*","^","s","o"]
    colors = ["r","b","g","k"]
    i = 0
    for grp in df_agg[groupby_field].unique():
        df1 = df_agg[(df_agg[groupby_field] == grp)]
        ax.plot(df1.date_c, df1["perc_outage"], label=grp, marker=markers[i], markersize=6, color=colors[i])
        i = i + 1
    ax.set_xticks(df1.date_c)
    ax.locator_params(tight=True, axis="x", nbins=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    ax.set_xlabel("Month/Year", fontsize=fontsize)
    ax.set_ylabel("Monthly Outage Rate (%)", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    # ax.set_title("{}".format(groupby_field))
    plt.legend(prop=dict(size=fontsize))
    plt.tight_layout()
    plt.show()

    # #-----------Plot corresponding to single ID-------------------------------------------------------
    # id_list = ["sanaa_grid_2_41"]
    # drad = df[(df.id.isin(id_list))][["id","date_c","rad_corr"]].sort_values(by=["date_c","id"])
    # dchange = df_out[(df_out.id.isin(id_list))]
    # dchange["date_temp"] = dchange.date_c
    # dchange = dchange.groupby([pd.Grouper(freq="1M", key="date_c")]).agg({"outage":"sum","date_temp":"nunique"}).reset_index()
    # dchange["perc_change"] = dchange["outage"]*100.0/dchange["date_temp"]
    # fontsize = 14
    # figsize = (10,4)
    # fig, ax = plt.subplots(figsize=figsize)
    # ax.scatter(drad.date_c, drad.rad_corr, s=6, label="Radiance")
    # ax.hlines(1, drad.date_c.min(), drad.date_c.max(), color="red", linestyles="dashed", label="Detection limit")
    # ax.set_xlabel("Date", fontsize=fontsize)
    # ax.set_ylabel("Nadir Normalized Radiance", fontsize=fontsize)
    # ax.tick_params(axis='both', which='major', labelsize=fontsize)
    # ax2 = ax.twinx()
    # ax2.plot(dchange.date_c, dchange.perc_change, color="black", label="Monthly Outage Rate")
    # ax2.set_ylabel("Monthly Outage Rate (%)", fontsize=fontsize)
    # # added these three lines to include all legends in one box
    # lines, labels = ax.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4, prop=dict(size=fontsize))
    # ax2.tick_params(axis='both', which='major', labelsize=fontsize)
    # plt.tight_layout()
    # plt.show()

    #-----------Proportion of pixels in every damage group (>80% outage)----------------------------------
    # dp = df_out.copy()
    # dp = dp.groupby(["id", pd.Grouper(key="date_c",freq="1M")]).agg({"LI":"count", "outage":"sum"}).reset_index()
    # dp.columns = ["id","date_c","total_readings","outage_count"]
    # dp["perc_outage"] = dp["outage_count"]*100.0/dp["total_readings"]
    # dp = dp[(dp.perc_outage >= 80)]

    # db = dbase[["id", groupby_field]].drop_duplicates()
    # id_counts = db.groupby(groupby_field).nunique()[["id"]].reset_index()
    # id_counts.columns = [groupby_field, "total_ids"]

    # dp_agg = pd.merge(dp, db, left_on=["id"], right_on=["id"])
    # dp_agg = dp_agg.groupby([groupby_field, "date_c"]).agg({"id":"nunique"}).reset_index()
    # dp_agg.columns = [groupby_field,"date_c","id_outage"]
    # dp_agg = pd.merge(dp_agg, id_counts, left_on=[groupby_field], right_on=[groupby_field], how="left")
    # dp_agg["perc_ids"] = dp_agg["id_outage"]*100.0/dp_agg["total_ids"]

    # dp_agg = dp_agg.groupby(groupby_field).resample("1M",on="date_c").mean()[["perc_ids"]].reset_index()
    # # dp_agg = pd.merge(dp[["date_c"]].drop_duplicates(), dp_agg, left_on=["date_c"], right_on=["date_c"], how="left")
    # dp_agg = dp_agg.fillna(0)
    # dp_agg = dp_agg.sort_values(by=["date_c",groupby_field])

    # fig, ax = plt.subplots()
    # markers = ["*","^","s","+"]
    # colors = ["r","b","g","m"]
    # i = 0
    # for grp in dp_agg[groupby_field].unique():
    #     df1 = dp_agg[(dp_agg[groupby_field] == grp)]
    #     ax.plot(df1.date_c, df1["perc_ids"], label=grp, marker=markers[i], markersize=4, color=colors[i])
    #     i = i + 1
    # ax.set_xticks(dp_agg.date_c.unique())
    # ax.locator_params(tight=True, axis="x", nbins=10)
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
    # ax.set_xlabel("Month/Year")
    # ax.set_ylabel("Proportion of pixels (%)")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    #-----------Proportion of pixels overall (>80% outage)----------------------------------
    # dp = df_out.copy()
    # dp = dp.groupby(["id", pd.Grouper(key="date_c",freq="1M")]).agg({"LI":"count", "outage":"sum"}).reset_index()
    # dp.columns = ["id","date_c","total_readings","outage_count"]
    # dp["perc_outage"] = dp["outage_count"]*100.0/dp["total_readings"]
    # dp = dp[(dp.perc_outage >= 80)]

    # dp_agg = dp.groupby(["date_c"]).nunique()[["id"]].reset_index()
    # dp_agg["perc_cells"] = dp_agg["id"]*100.0/3285

    # # fig, ax = plt.subplots()
    # ax.plot(dp_agg.date_c, dp_agg.perc_cells, marker="o", markersize=4, color="black", label="Overall")
    # ax.set_xticks(dp_agg.date_c)
    # ax.set_xlabel("Month/Year")
    # ax.locator_params(tight=True, axis="x", nbins=10)
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
    # ax.set_xlabel("Month/Year")
    # ax.set_ylabel("Proportion of pixels (%)")
    # plt.legend()
    # plt.grid()
    # plt.tight_layout()
    # plt.show()

    code.interact(local=locals())
    return None

def changepoint_detection_samples(df, penalty=15):
    # 3_41 (2), 37_6 (3), 32_18 (3), 21_36 (1)
    """
    penalty = 10, model=rbf usually gives around 3 change points and performs decent for points that actually have 3 changepoints.
    penalty = 15 worked almost perfectly for all points but it detected an extra change point in 3_41
    """
    gt_dict = {}
    gt_dict["sanaa_grid_3_41"] = 2
    gt_dict["sanaa_grid_37_6"] = 3
    gt_dict["sanaa_grid_32_18"] = 3
    gt_dict["sanaa_grid_21_36"] = 1
    gt_dict["sanaa_grid_30_17"] = 4
    gt_dict["sanaa_grid_24_29"] = 2

    df = df[(df.id.isin(["sanaa_grid_3_41","sanaa_grid_37_6","sanaa_grid_32_18","sanaa_grid_21_36","sanaa_grid_30_17","sanaa_grid_24_29"]))]

    for cellid in df.id.unique():
        print("cellid")
        df1 = df[(df.id == cellid)].sort_values(by=["date_c"])
        signal = df1.rad_corr.values
        dates = df1.date_c.values
        algo = rpt.Pelt(model="rbf").fit(signal)
        result = algo.predict(pen=penalty)

        # we exclude the last value of result as it is irrelevant
        result = result[0:-1]
        print("Result = {}".format(dates[result]))
        fig, ax = plt.subplots(figsize=(15,6))
        ax.scatter(dates, signal, s=6)
        ax.vlines(dates[result], signal.min(), signal.max(), linestyles="dashed")
        ax.set_title("{} Ground truth by EOG: {}".format(cellid, gt_dict[cellid]))
        ax.set_ylabel("Nadir Normalized Radiance")
        for i in result:
            date = pd.to_datetime(dates[i])
            date = date.strftime("%d-%b-%Y")
            ax.text(dates[i], signal.max(), date, color="red")
        # plt.show()
        # code.interact(local=locals())
        plt.savefig("{}_changepoint.pdf".format(cellid))
    return None

def changepoint_detection_singlecell(df, cellid, penalty=12, fontsize=14, figsize=(10,4), create_plot=False):
    df1 = df.sort_values(by=["date_c"])
    signal = df1.rad_corr.values
    dates = df1.date_c.values
    algo = rpt.Pelt(model="rbf").fit(signal)
    result = algo.predict(pen=penalty)
    # we exclude the last value of result as it is irrelevant
    result = result[0:-1]
    if create_plot == True:
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
        ax.scatter(dates, signal, s=6)
        ax.vlines(dates[result], signal.min(), signal.max(), linestyles="dashed")
        for i in result:
            date = pd.to_datetime(dates[i])
            date = date.strftime("%d-%b-%Y")
            ax.text(dates[i], signal.max()-0.5, date, color="red", fontsize=fontsize)
        ax.set_title("{}".format(cellid), fontsize=fontsize)
        ax.set_xlabel("Date", fontsize=fontsize)
        ax.set_ylabel("Nadir Normalized Radiance", fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.show()
    return dates[result]

def percentage_recovery(df):
    #-----------------------Create baseline for recovery tracking---------------------------------------
    dbase = df[(df.date_c<"2015-03-26")].groupby(["id","Latitude","Longitude"]).mean()[["rad_corr"]].reset_index()
    dbase = dbase.rename(columns={"rad_corr":"rad_base"})

    #---------------------Snapshot of % recovery by mean Jan/Feb/March/April 2019------------------------------------------
    drec = df[(df.date_c.dt.year == 2019) & (df.date_c.dt.month.isin([1,2,3,4]))].groupby("id").mean()[["rad_corr"]].reset_index()
    drec = drec.rename(columns={"rad_corr":"rad_present"})
    drec = pd.merge(drec, dbase, left_on=["id"], right_on=["id"], how="left")
    drec["recovery"] = drec["rad_present"]*100.0/drec["rad_base"]
    drec_gdf = create_geodataframe(drec, buffered=True, radius=462, cap_style=3)

    # # Visualize this
    # fig, ax1 = plt.subplots()
    # plot_geospatial_heatmap_with_event_locs(geo_df=drec_gdf, col_name="recovery", events_data=None, title=None, cmap=cm.autumn, cmap_type="autumn", marker_color=None, events_data_type="locations_points", needs_colormapping=False, add_title=False, event_locs_included=False, include_colorbar=True, with_streetmap=True, ax=ax1)
    # plt.show()
    #-----------------Merging of dataframe with db with groups-------------------------------------------------
    ds = drec.copy()
    ds["recovery_group"] = ds["recovery"].apply(lambda x: "low" if 0<x<=20 else "moderate" if 20<x<=50 else "high" if 50<x<=100 else "expansion")
    di = pd.read_hdf("yemen_groups.h5",key="zeal")
    ds = pd.merge(ds, di, left_on=["id"], right_on=["id"], how="left")

    #----------------Grouping base summary------------------------------------------------
    groupby_field = "damage_group"
    dm = ds.groupby(groupby_field).mean()[["rad_base","rad_present"]]
    dm["perc_recovery"] = dm["rad_present"]*100.0/dm["rad_base"]

    #-----------------Structure summary of recovery as of May 2019 of pixels in different groups------------------------
    dp = ds.copy()
    dp = dp.groupby(groupby_field)["recovery_group"].value_counts().reset_index(name="pixels")
    dp_totals = dp.groupby(groupby_field).sum()[["pixels"]].reset_index()
    dp_totals = dp_totals.rename(columns={"pixels":"total_pixels"})
    dp = pd.merge(dp, dp_totals, left_on=[groupby_field], right_on=[groupby_field])
    dp["perc_pixels"] = dp["pixels"]*100.0/dp["total_pixels"]
    # dp1 = dp.groupby(groupby_field).apply(lambda x: x["pixels"]*100.0/sum(x["pixels"])).reset_index()

    #-----------------Boxplots of recovery by damage groups-------------------------------
    # db = ds.copy()
    # db = db[(db.rad_base>1) & (db.rad_present>1)]
    # # db = db[(db.prerad_group != "Very Low")]
    # # db["infra_type"] = db.apply(lambda x: "education" if (x["ed_sites"]>0) & (x["health_sites"]==0) else "health" if (x["health_sites"]>0) & (x["ed_sites"]==0) else "both" if (x["health_sites"]>0 & x["ed_sites"]>0) else "None", axis=1)
    # fig, ax = plt.subplots()
    # db.boxplot(column=["recovery"], by=["damage_group"], ax=ax)
    # plt.xlabel("Damage Groups")
    # plt.ylabel("Recovery (%)")
    # plt.title("Variation in recovery of pixels of different damage groups")
    # plt.show()

    #-----------------Boxplots of recovery by prerad groups-------------------------------
    # db = ds.copy()
    # db = db[(db.rad_base>1) & (db.rad_present>1)]
    # # db = db[(db.prerad_group != "Very Low")]
    # # db["infra_type"] = db.apply(lambda x: "education" if (x["ed_sites"]>0) & (x["health_sites"]==0) else "health" if (x["health_sites"]>0) & (x["ed_sites"]==0) else "both" if (x["health_sites"]>0 & x["ed_sites"]>0) else "None", axis=1)
    # fig, ax = plt.subplots()
    # db.boxplot(column=["recovery"], by=["prerad_group"], ax=ax)
    # plt.xlabel("Pre-crisis Radiance Groups")
    # plt.ylabel("Recovery (%)")
    # plt.title("Variation in recovery of pixels belonging to different pre-crisis radiance groups")
    # plt.show()

    code.interact(local=locals())
    return None

def TNL_based_recovery(df):
    di = pd.read_hdf("yemen_groups.h5",key="zeal")
    # groupby_field = "damage_group"
    groupby_field = "j_group"
    #-----------------------Create baseline for recovery tracking---------------------------------------
    dbase = df[(df.date_c<"2015-03-26")].groupby(["id","Latitude","Longitude"]).mean()[["rad_corr"]].reset_index()
    dbase = dbase.rename(columns={"rad_corr":"rad_base"})
    dbase = pd.merge(dbase, di, left_on=["id"], right_on=["id"], how="left")
    dbase_tnl = dbase.groupby(groupby_field).sum()[["rad_base"]].reset_index()

    #-----------------------Calculate mean radiance for every month every year starting 2016--------------------------------
    # dm = df[(df.date_c.dt.year > 2015)]
    dm = df[(df.date_c>="2015-05-01")]
    dm = dm.groupby(["id",pd.Grouper(freq="1M",key="date_c")]).mean()[["rad_corr"]].reset_index()
    dm = dm.rename(columns={"rad_corr":"rad_month"})
    dm = pd.merge(dm, di, left_on=["id"], right_on=["id"], how="left")
    dm_tnl = dm.groupby([groupby_field,"date_c"]).agg({"rad_month":"sum"}).reset_index()
    dm_tnl = pd.merge(dm_tnl, dbase_tnl, left_on=[groupby_field], right_on=[groupby_field], how="left")
    dm_tnl["recovery"] = dm_tnl["rad_month"]*100.0/(dm_tnl["rad_base"])

    #-------get just TNL change wrt to group for tables ----------------
    dd = dm_tnl[(dm_tnl.date_c.dt.year == 2019) & (dm_tnl.date_c.dt.month.isin([1,2,3,4]))]
    dd = dd.groupby(groupby_field).mean()[["rad_month","rad_base","recovery"]].reset_index()

    # visualize this
    fig, ax = plt.subplots()
    markers = ["*","^","o","s"]
    i=0
    for grp in dm_tnl[groupby_field].unique():
        dm1 = dm_tnl[(dm_tnl[groupby_field] == grp)]
        ax.plot(dm1.date_c, dm1.recovery, label=grp, marker=markers[i], markersize=4)
        i = i+1
    plt.legend()
    plt.xlabel("Month/Year")
    plt.ylabel("TNL Recovery Rate(%)")
    # plt.title("Recovery of average daily TNL per month relative to the baseline (grouping by {})".format(groupby_field))
    plt.show()

    code.interact(local=locals())
    return None

def mean_recovery_calculations(df):
    di = pd.read_hdf("yemen_groups.h5",key="zeal")
    groupby_field = "j_group"
    #-----------------------Create baseline for recovery tracking---------------------------------------
    dbase = df[(df.date_c<"2015-03-26")].groupby(["id","Latitude","Longitude"]).mean()[["rad_corr"]].reset_index()
    dbase = dbase.rename(columns={"rad_corr":"rad_base"})
    dbase = pd.merge(dbase, di, left_on=["id"], right_on=["id"], how="left")
    dbase_tnl = dbase.groupby(groupby_field).mean()[["rad_base"]].reset_index()

    #-----------------------Calculate mean radiance for every month every year starting 2016--------------------------------
    # dm = df[(df.date_c.dt.year > 2015)]
    dm = df[(df.date_c>="2015-05-01")]
    dm = dm.groupby(["id",pd.Grouper(freq="1Y",key="date_c")]).mean()[["rad_corr"]].reset_index()
    dm = dm.rename(columns={"rad_corr":"rad_year"})
    dm = pd.merge(dm, di, left_on=["id"], right_on=["id"], how="left")
    dm_tnl = dm.groupby([groupby_field,"date_c"]).agg({"rad_year":"mean"}).reset_index()
    dm_tnl = pd.merge(dm_tnl, dbase_tnl, left_on=[groupby_field], right_on=[groupby_field], how="left")
    dm_tnl["recovery"] = dm_tnl["rad_year"]*100.0/(dm_tnl["rad_base"])
    code.interact(local=locals())
    return None

def recovery_and_rate_single_cell(df, dgf):
    # interesting recovery points: 28_49 and 19_42 and 9_27
    # what works: 8_28 with penalty 15 and no thresholding. Jun 4 2016
    # what works: 8_28 with penalty 10 and no thresholding. Jun 4 2016
    # what works: 8_28 with penalty 10 and w thresholding. Sep 8, 2016
    # what works: 8_28 with penalty 15 and w thresholding. doesn't work
    di = pd.read_hdf("yemen_groups.h5",key="zeal")
    groupby_field = "damage_group"
    cell_id = "sanaa_grid_8_28"
    #------------------------Find ruptures in time------------------------------------------------
    dchange = dgf[(dgf.id == cell_id)][["date_c","rad_corr"]].drop_duplicates()
    dchange2 = df[(df.id == cell_id)][["date_c","rad_corr"]].drop_duplicates()
    dates_rpt = changepoint_detection_singlecell(dchange, cell_id, penalty=12, create_plot=False)
    # code.interact(local=locals())

    #-----------------------Create baseline for recovery tracking---------------------------------------
    dbase = df[(df.date_c<="2015-03-26") & (df.id == cell_id)].groupby(["id","Latitude","Longitude"]).mean()[["rad_corr"]].reset_index()
    dbase = dbase.rename(columns={"rad_corr":"rad_base"})
    dbase = pd.merge(dbase, di, left_on=["id"], right_on=["id"], how="left")
    base_mean = dbase.rad_base.values[0]

    #--------------post-crisis radiance levels----------------------------------------------------------
    dm = df[(df.date_c>="2016-06-04") & (df.id == cell_id)]
    dm = dm.groupby(["id",pd.Grouper(freq="1M",key="date_c")]).mean()[["rad_corr"]].reset_index()
    dm = dm.rename(columns={"rad_corr":"rad_month"})
    dm["perc_recovery"] = dm["rad_month"]*100.0/base_mean

    #-------------Rate of recovery----------------------------------------------------------------
    dr = df[(df.date_c>="2016-06-04") & (df.id == cell_id)].reset_index().drop(columns=["index"])
    dr = dr.groupby(["id",pd.Grouper(freq="1M",key="date_c")]).mean()[["rad_corr"]].reset_index()
    X = dr.index
    y = dr["rad_corr"]
    # X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    dr["predicted_rad"] = predictions
    rate_of_recovery = model.params.values[0]

    code.interact(local=locals())
    #---------visualize this--------------------
    # playing around with dm to make good visualization
    dvis = pd.merge(df[(df.date_c>="2016-06-04") & (df.date_c<="2019-04-30") & (df.id == cell_id)][["date_c"]].drop_duplicates(), dm, left_on=["date_c"], right_on=["date_c"], how="left")
    dvis.set_index("date_c", inplace=True)
    dvis = dvis.bfill()
    dvis = dvis.reset_index()

    #----------------------------
    figsize = (10,4)
    fontsize = 14
    fig, ax = plt.subplots(figsize=figsize)
    # dchange = dchange2.copy()
    ax.scatter(dchange["date_c"], dchange["rad_corr"], s=6, c="blue", zorder=0, label="Radiance")
    ax.vlines(dates_rpt, dchange.rad_corr.min(), dchange.rad_corr.max(), linestyles="dashed", color="green", zorder=10)
    # ax.hlines(base_mean, dchange.date_c.min(), dates_rpt[0], linestyles="dashed", color="black", linewidth=3, zorder=10, label="Baseline mean")
    ax.plot(dr["date_c"], dr["predicted_rad"], c="black", zorder=20, linewidth=3, label="Best-fitted recovery trajectory")
    ax.set_ylabel("Nadir Normalized Radiance", fontsize=fontsize)
    ax.set_xlabel("Date", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    for i in range(len(dates_rpt)):
        date = pd.to_datetime(dates_rpt[i])
        date = date.strftime("%d-%b-%Y")
        ax.text(dates_rpt[i], dchange.rad_corr.max()-0.5, date, color="black", fontsize=fontsize-1)
    ax2 = ax.twinx()
    ax2.plot(dm["date_c"], dm["perc_recovery"], c="red", marker="o", markersize=6, label="Recovery")
    # ax2.scatter(dvis["date_c"], dvis["perc_recovery"], c="red", label="Monthly recovery", s=4)
    ax2.set_ylim([0,50])
    ax2.set_ylabel("Degree of recovery (%)", fontsize=fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize)
    # added these three lines to include all legends in one box
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4, prop=dict(size=fontsize))
    plt.tight_layout()
    plt.show()

    code.interact(local=locals())

    return None

def run_OLS(dr, column):
    X = dr.index
    y = dr[column]
    # X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    dr["predicted_rad"] = predictions
    rate_of_recovery = model.params.values[0]
    return rate_of_recovery

def recovery_rate_for_all(df, dgf):
    #------------------------Find ruptures in time------------------------------------------------
    # temp_arr = {}
    # for cell_id in dgf.id.unique():
    #     print("******************************")
    #     print("ID = {}".format(cell_id))
    #     dchange = dgf[(dgf.id == cell_id)][["date_c","rad_corr"]].drop_duplicates()
    #     dates_rpt = changepoint_detection_singlecell(dchange, cell_id, penalty=15, create_plot=False)
    #     print("dates = {}".format(dates_rpt))
    #     temp_arr[cell_id] = dates_rpt
    # code.interact(local=locals())

    #-----------------------Create baseline for recovery tracking---------------------------------------
    di = pd.read_hdf("yemen_groups.h5",key="zeal")
    dbase = df[(df.date_c<"2015-03-26")].groupby(["id","Latitude","Longitude"]).mean()[["rad_corr"]].reset_index()
    dbase = dbase.rename(columns={"rad_corr":"rad_base"})
    dbase = pd.merge(dbase, di, left_on=["id"], right_on=["id"], how="left")
    # base_mean = dbase.rad_base.values[0]

    # #---------------------- PICKLED ALREADY----------------
    # dm = df[(df.date_c>="2016-06-04")]
    # dm = dm.groupby(["id",pd.Grouper(freq="1M",key="date_c")]).mean()[["rad_corr"]].reset_index()
    # dm = dm.rename(columns={"rad_corr":"rad_month"})

    # temp_dict = {}
    # for cell_id in df.id.unique():
    #     print("******************************")
    #     print("ID = {}".format(cell_id))
    #     dm_rec = dm[(dm.id == cell_id)].reset_index().drop(columns=["index"])
    #     rate_of_recovery = run_OLS(dm_rec, column=["rad_month"])
    #     temp_dict[cell_id] = rate_of_recovery
    #     print("ROR = {}".format(rate_of_recovery))

    dr = pd.read_hdf("id_ror.h5",key="zeal")
    dr = pd.merge(dr[["id","ror"]].drop_duplicates(), dbase, left_on=["id"], right_on=["id"], how="left")
    dr["ror_norm"] = (dr["ror"] - min(dr["ror"]))/(max(dr["ror"] - min(dr["ror"])))
    breaks = jenkspy.jenks_breaks(dr.ror.values, nb_class=3)
    dr["ror_group"] = dr.ror.apply(lambda x: "Low" if breaks[0]<=x<breaks[1] else "Medium" if breaks[1]<=x<breaks[2] else "High")
    dr_gdf = create_geodataframe(dr, buffered=True, radius=462, cap_style=3)

    #---------create plots---------------------------------------------
    # fig, ax1 = plt.subplots(figsize=(4,5))
    # plot_geospatial_heatmap_with_event_locs(geo_df=dr_gdf, col_name="ror_norm", events_data=None, title=None, cmap=cm.seismic, cmap_type="seismic", marker_color=None, events_data_type="locations_points", needs_colormapping=False, add_title=False, event_locs_included=False, include_colorbar=True, with_streetmap=True, ax=ax1)
    # plt.rc('font', size=14)
    # plt.tight_layout()
    # plt.show()

    code.interact(local=locals())
    return None

if __name__=='__main__':
    #--------read & process filtered NL dataset---------- (FILTERED)
    dgf = pd.read_hdf("filtered_data.h5", key="zeal")
    #clip negative radiance values to 0
    dgf["RadE9_Mult_Nadir_Norm"] = dgf["RadE9_Mult_Nadir_Norm"].clip(lower=0)
    # combine multiple readings for same day and same id
    dgf = dgf.groupby(["id","Latitude","Longitude","date_c"]).mean()[["RadE9_Mult_Nadir_Norm","LI"]].reset_index()

    #---------read lunar corrected dataset-------------- (UNFILTERED)
    data_path = "./"
    dg = pd.read_hdf(data_path + "yemen_STL_lunar_corrected.h5", key="zeal")
    dg = dg.drop(columns = ["RadE9_Mult_Nadir_Norm"])
    dg = dg.rename(columns={"rad_corr":"RadE9_Mult_Nadir_Norm"})

    #----------holiday lighting correction--------------
    dgf = postcrisis_holiday_lighting_correction(dgf.copy())
    dg = postcrisis_holiday_lighting_correction(dg.copy())
    """
    NOTE: In both the dataframes rad_corr will represent data with correction for holidays
    """

    #------------detect change points in data illustration-----------
    # changepoint_detection_samples(dgf.copy())

    #---------Associate areas in Sanaa with NL grid cells----------
    # output has been pickled using "disaster_mapping_march2015.py" file
    # da = associate_NLgrid_with_areas(dg[["id","Latitude","Longitude"]].drop_duplicates())
    da = pd.read_hdf("yemen_grid_with_admin.h5",key="zeal")

    #-------------2015 baseline and infrastructure---------
    extra_data_path = "../extra_datasets/"
    di = pd.read_pickle(extra_data_path + "yemen_infra_pop_data_combined_2.pck")
    # dbase = precrisis_baseline_data(dg.copy(), di.copy())
    # dbase = dbase[["id","damage_group","pop_group","prerad_group","ed_sites","health_sites","total_pop"]].drop_duplicates()
    dbase = pd.read_hdf("yemen_groups.h5",key="zeal")
    # dbase = pd.merge(dbase, di[["id","labels_pre"]].drop_duplicates(), left_on=["id"], right_on=["id"], how="left")

    #-------------Identifying outage duration--------------
    # detect_and_measure_outage_duration(dgf.copy(), dbase.copy()) #we use filtered data to look for outages

    #-----------Percentage Recovery for All----------------
    # percentage_recovery(dg.copy())

    #-----------TNL and grouping based recovery------------
    # TNL_based_recovery(dg.copy())
    # mean_recovery_calculations(dg.copy())

    #-----------rate of recovery - single cell-------------
    recovery_and_rate_single_cell(dg.copy(), dgf.copy())

    #-----------rate of recovery - all cells-------------
    # recovery_rate_for_all(dg.copy(), dgf.copy())
    code.interact(local=locals())

    #-----------create plots for detection demonstration-------
    # cell_id = "sanaa_grid_8_28"
    # dchange = dgf[(dgf.id == cell_id)][["date_c","rad_corr"]].drop_duplicates()
    # changepoint_detection_singlecell(dchange, cell_id, penalty=12, fontsize=14, figsize=(10,4), create_plot=True)
