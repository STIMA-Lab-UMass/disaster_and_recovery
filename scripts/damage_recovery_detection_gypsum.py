"""
This script is used to produce all the plots and analysis for recovery tracking section's Disaster mapping portion.
This script is specifically focused on recovery mapping and analysis post March 26, 2015.
@zeal
"""
import shapely
from shapely.geometry import Point, Polygon, box
import pandas as pd
import geopandas as gpd
import numpy as np
import pickle
import os
import matplotlib.dates as mdates
import code
import datetime
from auxiliary_disaster_analysis import query_event_locations_by_date, query_event_locations_by_monthyear
import ruptures as rpt
from shapely.ops import unary_union
from yemen_plotting_utils import *
from math import radians, cos, sin, asin, sqrt
from collections import Counter
import seaborn as sns

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
    return df

def changepoint_detection_singlecell(df, cellid, penalty=12, create_plot=False):
    df1 = df.sort_values(by=["date_c"])
    signal = df1.rad_corr.values
    dates = df1.date_c.values
    algo = rpt.Pelt(model="rbf").fit(signal)
    result = algo.predict(pen=penalty)
    # we exclude the last value of result as it is irrelevant
    result = result[0:-1]

    if len(result) != 0:
        op_arr = [pd.to_datetime(x).date() for x in dates[result]]
        # consider disaster dates only between 2015 january and 2015 may.
        disaster_arr = [y for y in op_arr if datetime.date(2015,1,1) <= y <= datetime.date(2015,5,1)]
        # consider recovery dates only beyong 2015 may.
        recovery_arr = [y for y in op_arr if y > datetime.date(2015,5,1)]
    else:
        print("!!!!!!_______________!!!!!!")
        print("No ruptures")
        print("!!!!!!_______________!!!!!!")
        op_arr = []
        disaster_arr = []
        recovery_arr = []
    return op_arr, disaster_arr, recovery_arr

def date_detection_for_all(dgf, output_path):
    #------------------------Find ruptures in time------------------------------------------------
    dgf = dgf[(dgf.id.isin(["sanaa_grid_3_41", "sanaa_grid_21_36","sanaa_grid_24_29","sanaa_grid_30_17","sanaa_grid_32_18","sanaa_grid_37_6","sanaa_grid_6_6","sanaa_grid_20_22","sanaa_grid_10_6", "sanaa_grid_16_15"]))]
    overall = {}
    disaster = {}
    recovery = {}
    i = 0
    for cell_id in dgf.id.unique():
        print("i=={}".format(i))
        print("******************************", flush=True)
        print("ID = {}".format(cell_id), flush=True)
        dchange = dgf[(dgf.id == cell_id)][["date_c","rad_corr"]].drop_duplicates()
        overall[cell_id], disaster[cell_id], recovery[cell_id] = changepoint_detection_singlecell(dchange, cell_id, penalty=12, create_plot=False)
        print("dates = {}".format(overall[cell_id]), flush=True)
        # if (i%20==0):
        #     pickle.dump(overall, open(output_path + "rupture_all_wo_filtering.pck", "wb"))
        #     pickle.dump(disaster, open(output_path + "rupture_disaster_wo_filtering.pck", "wb"))
        #     pickle.dump(recovery, open(output_path + "rupture_recovery_wo_filtering.pck", "wb"))
        #     print("##################")
        #     print("Dictionary checkpoint saved.", flush=True)
        i=i+1

    df = dgf[["id"]].drop_duplicates()
    df["overall_dt"] = df.id.apply(lambda x: overall[x])
    df["disaster_dt"] = df.id.apply(lambda x: disaster[x])
    df["recovery_dt"] = df.id.apply(lambda x: recovery[x])
    # df.to_hdf(output_path + "rupture_detection_wo_filtering_df.h5", key="zeal")
    # print("DataFrame has been saved.", flush=True)
    return df

def process_detection_output(df):
    df["d_dt"] = df["disaster_dt"].apply(lambda x: x[0] if len(x)>0 else "Not available")
    df["r_dt"] = df["recovery_dt"].apply(lambda x: x[0] if len(x)>0 else "Not available")
    code.interact(local=locals())
    #--------damage--------------------
    dd = df.copy()
    dd = dd.groupby("d_dt").count()[["id"]].reset_index()
    dd["perc_id"] = dd["id"]*100.0/3285
    dd_notavail = dd[(dd.d_dt == "Not available")]

    dd = dd[(dd.d_dt!="Not available")]
    dd["date_c"] = dd["d_dt"].apply(lambda x: str(x))
    dd["date_c"] = pd.to_datetime(dd["date_c"], format="%Y-%m-%d")
    # dd = dd.resample("1D", on="date_c").mean().reset_index() #upsampling
    # dd = dd.fillna(0)
    # plt.bar(dd["date_c"], dd["perc_id"])

    #-------possible recovery----------
    dr = df.copy()
    dr = dr.groupby("r_dt").count()[["id"]].reset_index()
    dr["perc_id"] = dr["id"]*100.0/3285
    dr["date_c"] = dr["r_dt"].apply(lambda x: str(x))

    # code.interact(local=locals())
    return dd, dr

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

def verify_damage(df):
    #----read extra datasets--------
    da = pd.read_hdf("yemen_grid_with_admin.h5",key="zeal")
    de_locs = query_event_locations_by_monthyear([1,2,3,4,5],[2015],buffered=False,radius=None,cap_style=None)
    de_locs = de_locs[["data_id","event_date","sub_event_type","admin2","geometry"]]
    de_locs["sub_event_type"] = de_locs["sub_event_type"].apply(lambda x: "non-aerial" if x!="Air/drone strike" else x)

    #-----process disaster detection dataframe-----
    df["d_dt"] = df["disaster_dt"].apply(lambda x: x[0] if len(x)>0 else "Not available")
    df = df[(df.d_dt!="Not available")]
    df["date_c"] = df["d_dt"].apply(lambda x: str(x))
    df["date_c"] = pd.to_datetime(df["date_c"], format="%Y-%m-%d")
    df = df[["id","date_c"]].drop_duplicates()

    #----merge all dataframes---------------------
    df = pd.merge(df, da[["id","Latitude","Longitude","adm2"]], left_on="id", right_on="id", how="left")
    df["adm2"] = df["adm2"].apply(lambda x: "Maain" if x=="Ma'ain" else "Aththaorah" if x=="Ath'thaorah" else "Shuaub" if x=="Shu'aub" else "Assafiyah" if x=="Assafiyah" else "Azzal" if x=="Az'zal" else x)
    gdf = create_geodataframe(df, radius=462, cap_style=3, buffered=False)

    #-----verification----------------------------
    dm = pd.DataFrame(columns={"detec_date","events","event_type","pixels","mean_distance", "min_distance", "max_distance"})
    # here date_c indicates the detected infliction date
    for event_dt in df.date_c.unique():
        de1 = de_locs[(de_locs.event_date == event_dt)]
        gdf1 = gdf[(gdf.date_c == event_dt)]
        print("#############################################################")
        print("Total events on {}: {}".format(event_dt, de1.data_id.nunique()))
        admins_event = de1.admin2.unique()
        if len(admins_event) != 0:
            print("Admins affected: {}".format(admins_event))
            print("#############################################################")
            de1_geom = de1.geometry.values
            event_type = de1.sub_event_type.values
            event_type,_ = Counter(event_type).most_common(1)[0]
            # gdf1["dist_val"] = gdf1.geometry.apply(lambda x: min(x.distance(k) for k in de1_geom))
            gdf1["dist_val"] = gdf1.geometry.apply(lambda a: min(haversine(a.x, a.y, k.x, k.y) for k in de1_geom))
            print("Minimum distance: {}".format(gdf1.dist_val.min()))
            print("Mean distance: {}".format(gdf1.dist_val.mean()))
            print("Median distance: {}".format(gdf1.dist_val.mean()))
            print("Maximum distance: {}".format(gdf1.dist_val.max()))
            dm = dm.append({"detec_date":event_dt, "events": de1.data_id.nunique(), "event_type":event_type, "pixels":len(gdf1), "mean_distance":gdf1.dist_val.mean(), "min_distance":gdf1.dist_val.min(), "max_distance":gdf1.dist_val.max()}, ignore_index=True)
        else:
            print("Admins affected: 0")
            print("#############################################################")
            dm = dm.append({"detec_date":event_dt, "events": 0, "event_type": "unknown", "pixels":len(gdf1), "mean_distance":0, "min_distance":0, "max_distance":0}, ignore_index=True)

    dm["colors"] = dm.event_type.apply(lambda x: "red" if x=="Air/drone strike" else "green" if x=="unknown" else "blue")
    dm = dm.sort_values(by=["detec_date"])
    dm.set_index("detec_date", inplace=True)
    dm = dm.resample("1D").asfreq()
    dm = dm.fillna(0)

    dm["mean_distance"] = dm["mean_distance"].apply(lambda x: 1 if x==0 else x)
    dm["cumulative_pixels"] = dm["pixels"].cumsum()
    dm["pixel_prop"] = dm["pixels"]*100.0/3285
    dm["cumul_pixel_prop"] = dm["cumulative_pixels"]*100.0/3285
    dm = dm.reset_index()
    # dm = dm[(dm.detec_date >= "2015-03-15")]
    code.interact(local=locals())

    fontsize = 14
    figsize = (12, 4)
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(dm["detec_date"], dm["pixel_prop"], color="blue", label="Proportion", alpha=0.6)
    ax.set_xlabel("Date", fontsize=fontsize)
    ax.set_ylabel("Proportion of pixels (%)", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    ax2 = ax.twinx()
    ax2.plot(dm["detec_date"], dm["cumul_pixel_prop"], color="black", marker="*", markersize=4, label="Cumulative proportion")
    ax2.set_ylim(0,100)
    ax2.set_ylabel("Proportion of pixels (%) \n [Cumulative]", fontsize=fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize)

    ax2.locator_params(tight=True, axis="x", nbins=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b-%Y"))

    # plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)
    # added these three lines to include all legends in one box
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left", prop=dict(size=fontsize))


    plt.tight_layout()
    plt.show()

    # code.interact(local=locals())
    # dm = dm.rename(columns={"mean_distance":"Mean distance (km)", "event_type":"Type of event"})
    # fig, ax = plt.subplots()
    # for ev_type in dm.event_type.unique():
    #     dm1 = dm[(dm.event_type == ev_type)]
    #     ax.scatter(dm1.datec_date.dt.strftime("%d-%b-%y"), dm1.pixels, s=dm1.mean_distance*50, c=dm1.colors, alpha=0.5, label=ev_type)
    # plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)
    # ax.set_xlabel("Date of Event")
    # ax.set_ylabel("Number of affected pixels")
    # ax.legend()
    # plt.tight_layout()
    # plt.show()

    # sns.set()
    # fig, ax = plt.subplots()
    # g = sns.scatterplot(x="detec_date", y="pixels", size="Mean distance (km)", sizes=(50,300), hue="Type of event", data=dm)
    # plt.xticks(rotation=70)
    # plt.xlabel("Date of event")
    # plt.ylabel("Proportion of pixels (%)")
    # plt.tight_layout()
    # plt.show()

    # sns.set(style="whitegrid")
    # fig, ax = plt.subplots()
    # sns.barplot(x=dm.detec_date.dt.strftime("%d-%b-%y"), y=dm.pixels, hue=dm.event_type, width=0.09)
    # plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)
    # plt.legend()
    # ax.set_xlabel("Date of Event")
    # ax.set_ylabel("Proportion of pixels (%)")
    # plt.tight_layout()
    # plt.show()

    code.interact(local=locals())
    return None

def plots_for_recovery(df):
    #-----process recovery detection dataframe-----
    df["r_dt"] = df["recovery_dt"].apply(lambda x: x[0] if len(x)>0 else "Not available")
    df = df[(df.r_dt!="Not available")]
    df["date_c"] = df["r_dt"].apply(lambda x: str(x))
    df["date_c"] = pd.to_datetime(df["date_c"], format="%Y-%m-%d")
    df = df.groupby(pd.Grouper(freq="1M",key="date_c")).count()[["id"]].reset_index()
    dm = df.copy()
    dm["cumul_pixels"] = dm["id"].cumsum()
    dm["pixel_prop"] = dm["id"]*100/3285
    dm["cumul_pixel_prop"] = dm["cumul_pixels"]*100/3285
    # dm["date_c"] = dm["date_c"].dt.strftime("%b-%Y")
    code.interact(local=locals())

    fig, ax = plt.subplots()
    ax.plot(dm["date_c"], dm["cumul_pixel_prop"], color="black", marker="*", markersize=4)
    ax.set_xlabel("Month/Year")
    ax.set_ylabel("Proportion of pixels (%)")
    ax.set_xticks(dm.date_c)
    ax.locator_params(tight=True, axis="x", nbins=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
    plt.tight_layout()
    plt.show()

    # fig, ax = plt.subplots()
    # ax.bar(dm["date_c"], dm["pixel_prop"], color="blue", label="Proportion", alpha=0.6, width=1.2)
    # ax.set_xlabel("Month/Year")
    # ax.set_ylabel("Proportion of pixels (%)")
    # # ax.set_xticks(dm.date_c)
    # # ax.locator_params(tight=True, axis="x", nbins=10)
    # # ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))

    # ax2 = ax.twinx()
    # ax2.plot(dm["date_c"], dm["cumul_pixel_prop"], color="black", marker="*", markersize=4, label="Cumulative")
    # ax2.set_ylim(0,100)
    # ax2.set_ylabel("Proportion of pixels (%) [Cumulative]")
    # plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)
    # # added these three lines to include all legends in one box
    # lines, labels = ax.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax2.legend(lines + lines2, labels + labels2, loc=0)
    # fig.autofmt_xdate()
    # plt.tight_layout()
    # plt.show()
    code.interact(local=locals())
    return None

if __name__=='__main__':
    data_path = "./"
    output_path = "./"

    # data_path = "/home/zshah/yemen_files/data/"
    # output_path = "/home/zshah/yemen_files/outputs/"

    #--------read & process filtered NL dataset---------- (FILTERED)
    dgf = pd.read_hdf(data_path + "filtered_data.h5", key="zeal")
    #clip negative radiance values to 0
    dgf["RadE9_Mult_Nadir_Norm"] = dgf["RadE9_Mult_Nadir_Norm"].clip(lower=0)
    # combine multiple readings for same day and same id
    dgf = dgf.groupby(["id","Latitude","Longitude","date_c"]).mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()

    #----------holiday lighting correction--------------
    dgf = postcrisis_holiday_lighting_correction(dgf.copy())
    print("holiday based correction done.", flush=True)

    #---------date detection----------------------------
    # dd = date_detection_for_all(dgf.copy(), output_path)
    dd = pd.read_hdf("rupture_detection_wo_filtering_df.h5", key="zeal")
    # code.interact(local=locals())
    #--------overall detection output & proportion plot-------------------
    # process_detection_output(dd.copy())

    #---------verify damaged sites----------------------------------------
    verify_damage(dd.copy())

    #------------Recovery plot--------------------------------------------
    # plots_for_recovery(dd.copy())
    # code.interact(local=locals())
