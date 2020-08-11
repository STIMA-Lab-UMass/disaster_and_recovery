"""
This script is used to produce all the plots for Overview subsection in results & evaluation section
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
from yemen_plotting_utils import plot_geospatial_heatmap_subplots

def getRadius(buffer):
    lat_degree = 110.54 * 1000
    lon_degree = 111.32 * 1000
    lat_radius = buffer / lat_degree
    lon_radius = buffer / lon_degree
    radius = max(lat_radius,lon_radius)
    return radius

def track_TNL_evolution(df):
    # plot: daily_filtered_TNL_timeseries.pdf
    # Daily TNL dataframe
    df1 = df.groupby("date_c").sum()[["RadE9_Mult_Nadir_Norm"]].reset_index()
    df1 = df1.sort_values(by="date_c")
    plt.scatter(df1.date_c, df1.RadE9_Mult_Nadir_Norm, s=4)
    plt.ylabel("TNL Radiance (nW/cm2/sr)")
    plt.xlabel("Day")
    plt.show()
    # code.interact(local=locals())
    return None

def boxplot_weekday_weekend_rad(db):
    # compute sum value of night lights (TNL) in the region on a daily bases
    db = db.groupby("date_c").sum()[["RadE9_Mult_Nadir_Norm"]].reset_index()

    db["dow"] = db.date_c.apply(lambda x: x.dayofweek)
    db["year"] = db.date_c.dt.year
    db["dayname"] = db.dow.apply(lambda x: "Mon" if x==0 else "Tue" if x==1 else "Wed" if x==2 else "Thu" if x==3 else "Fri" if x==4 else "Sat" if x==5 else "Sun")
    db["weekname"] = db.dow.apply(lambda x: "Weekend" if x==4 else "Weekend" if x==5 else "Weekday")

    ax=sns.boxplot(x=db.year, y=db.RadE9_Mult_Nadir_Norm, hue=db.weekname)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["Weekday", "Weekend"])
    plt.xlabel("Year")
    plt.ylabel("TNL Radiance (nW/cm2/sr)")
    # plt.title("Variation in total night lights during weekdays and weekends")
    plt.show()
    # code.interact(local=locals())
    return None

def month_to_month_mean_daily_TNL(db):
    db = db.copy()
    # compute daily TNL
    db = db.groupby([pd.Grouper(key="date_c", freq="1D")]).sum()[["RadE9_Mult_Nadir_Norm"]].reset_index()
    # compute mean daily TNL
    db = db.groupby([pd.Grouper(key="date_c", freq="1M")]).mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()

    # db = db.groupby([pd.Grouper(key="date_c", freq="1M")]).sum()[["RadE9_Mult_Nadir_Norm"]].reset_index()

    # calculate % change month-to-month
    # db["prev_rad"] = db["RadE9_Mult_Nadir_Norm"].shift(1)
    # db["perc_change"] = (db["RadE9_Mult_Nadir_Norm"] - db["prev_rad"]) * 100.0/db["prev_rad"]

    # how much does a value change? if 2 becomes 1 in the next timestep, then the value will go down by 50%
    # or changed by -50% [how: (1-2)/2 = -1/2 = -50%]
    db["next_rad"] = db["RadE9_Mult_Nadir_Norm"].shift(-1)
    db["perc_change"] = (db["next_rad"] - db["RadE9_Mult_Nadir_Norm"]) * 100.0/db["RadE9_Mult_Nadir_Norm"]

    # code.interact(local=locals())

    # plot the results
    fig, ax = plt.subplots()
    lns1 = ax.plot(db.date_c.dt.strftime('%b-%y'), db.RadE9_Mult_Nadir_Norm, marker="o", color="k", zorder=100, label="TNL")
    ax.set_xlabel("Month-Year")
    ax.set_ylabel("Mean Daily TNL")

    ax2 = ax.twinx()
    lns2 = ax2.bar(db.date_c.dt.strftime('%b-%y'), db.perc_change, width=0.8, color="red", edgecolor="black", alpha=0.6, zorder=1, label="Change (%)")
    ax2.set_ylabel("Change (%)")

    # added these three lines to include all legends in one box
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    fig.autofmt_xdate()
    plt.show()
    # code.interact(local=locals())
    return None

def create_pre_post_present_heatmaps(db):
    """
    # Pre data - Jan 2015
    # Post data - April 2015
    # Present data - April 2019
    """
    db_pre = db[(db.date_c.dt.year == 2015) & (db.date_c.dt.month == 1)].groupby(["id", "Latitude", "Longitude"]).mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()
    db_post = db[(db.date_c.dt.year == 2015) & (db.date_c.dt.month == 4)].groupby(["id", "Latitude", "Longitude"]).mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()
    db_present = db[(db.date_c.dt.year == 2019) & (db.date_c.dt.month == 4)].groupby(["id", "Latitude", "Longitude"]).mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()

    max_rad_limit = 255 #ideally max rad is 199 but we can use 300 as stretching value just for sake of better visualization
    db_pre["RadE9_Mult_Nadir_Norm"]= db_pre["RadE9_Mult_Nadir_Norm"].apply(lambda x: 255 * np.sqrt(np.clip(x, a_min=None, a_max=max_rad_limit)/max_rad_limit))
    db_post["RadE9_Mult_Nadir_Norm"]= db_post["RadE9_Mult_Nadir_Norm"].apply(lambda x: 255 * np.sqrt(np.clip(x, a_min=None, a_max=max_rad_limit)/max_rad_limit))
    db_present["RadE9_Mult_Nadir_Norm"]= db_present["RadE9_Mult_Nadir_Norm"].apply(lambda x: 255 * np.sqrt(np.clip(x, a_min=None, a_max=max_rad_limit)/max_rad_limit))

    titles = ["pre", "post", "present"]
    i = 0
    c = 1
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(14, 6)) # Reference: https://stackoverflow.com/questions/48129222/matplotlib-make-plots-in-functions-and-then-add-each-to-a-single-subplot-figure
    axes_arr = [ax1, ax2, ax3]
    for df in [db_pre, db_post, db_present]:
        print(df.head())
        geom = [Point(x,y) for x,y in zip(df["Longitude"], df["Latitude"])]
        gdf = gpd.GeoDataFrame(df, geometry=geom, crs={"init":"epsg:4326"})
        distance = getRadius(462) #462m = 15 arc seconds for Nairobi (Reference: https://www.opendem.info/arc2meters.html)
        gdf["geometry"] = gdf["geometry"].apply(lambda x: x.buffer(distance/2, cap_style=3))
        plot_geospatial_heatmap_subplots(gdf, col_name="RadE9_Mult_Nadir_Norm", title="{}".format(titles[i]), cmap=cm.hot, cmap_type="hot", with_sites=False, add_title=False, ax=axes_arr[i])
        i = i+1
        c = c+1
        del(df)
        del(gdf)
    plt.show()
    # code.interact(local=locals())
    return None

def create_pre_post_present_heatmaps_individually(db):
    """
    # Pre data - Jan 2015
    # Post data - April 2015
    # Present data - April 2019
    """
    db_pre = db[(db.date_c.dt.year == 2015) & (db.date_c.dt.month == 1)].groupby(["id", "Latitude", "Longitude"]).mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()
    db_post = db[(db.date_c.dt.year == 2015) & (db.date_c.dt.month == 4)].groupby(["id", "Latitude", "Longitude"]).mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()
    db_present = db[(db.date_c.dt.year == 2019) & (db.date_c.dt.month == 4)].groupby(["id", "Latitude", "Longitude"]).mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()

    max_rad_limit = 255 #ideally max rad is 199 but we can use 300 as stretching value just for sake of better visualization
    db_pre["RadE9_Mult_Nadir_Norm"]= db_pre["RadE9_Mult_Nadir_Norm"].apply(lambda x: 255 * np.sqrt(np.clip(x, a_min=None, a_max=max_rad_limit)/max_rad_limit))
    db_post["RadE9_Mult_Nadir_Norm"]= db_post["RadE9_Mult_Nadir_Norm"].apply(lambda x: 255 * np.sqrt(np.clip(x, a_min=None, a_max=max_rad_limit)/max_rad_limit))
    db_present["RadE9_Mult_Nadir_Norm"]= db_present["RadE9_Mult_Nadir_Norm"].apply(lambda x: 255 * np.sqrt(np.clip(x, a_min=None, a_max=max_rad_limit)/max_rad_limit))

    titles = ["Jan 2015", "Apr 2015", "Apr 2019"]
    i = 0
    c = 1
     # Reference: https://stackoverflow.com/questions/48129222/matplotlib-make-plots-in-functions-and-then-add-each-to-a-single-subplot-figure
    for df in [db_pre, db_post, db_present]:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        print(df.head())
        geom = [Point(x,y) for x,y in zip(df["Longitude"], df["Latitude"])]
        gdf = gpd.GeoDataFrame(df, geometry=geom, crs={"init":"epsg:4326"})
        distance = getRadius(462) #462m = 15 arc seconds for Nairobi (Reference: https://www.opendem.info/arc2meters.html)
        gdf["geometry"] = gdf["geometry"].apply(lambda x: x.buffer(distance/2, cap_style=3))
        plot_geospatial_heatmap_subplots(gdf, col_name="RadE9_Mult_Nadir_Norm", title="{}".format(titles[i]), cmap=cm.hot, cmap_type="hot", with_sites=False, add_title=True, ax=ax)
        i = i+1
        c = c+1
        del(df)
        del(gdf)
    plt.show()
    # code.interact(local=locals())
    return None

def TNL(df):
    dg = df.groupby(["date_c"]).sum()[["RadE9_Mult_Nadir_Norm"]].reset_index()

    fontsize = 14
    fig, ax = plt.subplots(figsize=(10,4))
    ax.scatter(dg.date_c, dg.RadE9_Mult_Nadir_Norm, s=6)
    ax.set_xlabel("Date", fontsize=fontsize)
    ax.set_ylabel("Nadir Normalized Radiance", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.tight_layout()
    plt.show()

    code.interact(local=locals())
    return None

if __name__=='__main__':
    data_path = "./"

    #--------read & process necessary datasets----------
    # dg = pd.read_hdf("filtered_data.h5", key="zeal")
    # #clip negative radiance values to 0
    # dg["RadE9_Mult_Nadir_Norm"] = dg["RadE9_Mult_Nadir_Norm"].clip(lower=0)
    # # combine multiple readings for same day and same id
    # dg = dg.groupby(["id","Latitude","Longitude","date_c"]).mean()[["RadE9_Mult_Nadir_Norm","LI"]].reset_index()

    # #---------read lunar corrected dataset--------------
    dg = pd.read_hdf(data_path + "yemen_STL_lunar_corrected.h5", key="zeal")
    dg = dg.drop(columns = ["RadE9_Mult_Nadir_Norm"])
    dg = dg.rename(columns={"rad_corr":"RadE9_Mult_Nadir_Norm"})
    code.interact(local=locals())


    #-----track evolution of TNL over time--------------
    # track_TNL_evolution(dg)

    #-----visualize variation in weekdays and weekends---------
    # boxplot_weekday_weekend_rad(dg)

    #-----month to month mean daily TNL and % changes----------
    # month_to_month_mean_daily_TNL(dg)

    #-------plot heatmap timelines-----------------------------
    # create_pre_post_present_heatmaps(dg)
    create_pre_post_present_heatmaps_individually(dg)

    # TNL(dg.copy())

    code.interact(local=locals())