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

def calculate_jenks_breaks(df):
    rad = df.mean_base.values
    breaks = jenkspy.jenks_breaks(rad, nb_class=3)
    df["j_group"] = df.mean_base.apply(lambda x: "Low" if breaks[0]<=x<breaks[1] else "Medium" if breaks[1]<=x<breaks[2] else "High")
    dgroups = pd.read_hdf("yemen_groups.h5",key="zeal")
    dgroups = pd.merge(dgroups, df[["id","j_group"]].drop_duplicates(), left_on=["id"], right_on=["id"], how="left")
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

    #---------get pre-crisis mean radiance--------------
    dz = zscore_percchange_march2015(dg.copy(), create_plot="False")

    #--------jenks natural breaks-----------------------
    calculate_jenks_breaks(dz.copy())
    code.interact(local=locals())