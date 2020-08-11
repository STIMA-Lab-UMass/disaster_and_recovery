"""
This script is used to produce all the plots for Data description section (NL data and NL data pre-processing)
@zeal
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import code
import datetime

def preprocess_df(df):
    # df = df.copy()
    # code.interact(local=locals())
    #------Parse date time--------------
    df["date_c"] = df["Agg_Name"].apply(lambda x: pd.to_datetime(x.split("_")[1].split("d")[1],format="%Y%m%d"))
    df["start_time"] = df["Agg_Name"].apply(lambda x: pd.to_datetime(x.split("_")[2].split("t")[1],format="%H%M%S%f"))
    df["end_time"] = df["Agg_Name"].apply(lambda x: pd.to_datetime(x.split("_")[3].split("e")[1],format="%H%M%S%f"))
    #------Extract bit flag values------
    df['Vflag_bin'] = df['QF_Vflag'].apply(lambda x: '{:032b}'.format(x))
    # flip the array of bits: from left->right to right-> left
    # because binary numbers are indexed from right to left
    ### NOTE: another approach would be to use BITWISE AND
    df['Vflag_bin'] = df['Vflag_bin'].apply(lambda x: x[::-1])
    # extract flags corresponding to each metric using standard python indexing
    df['flag_cloud2'] = df['Vflag_bin'].apply(lambda x: x[3:5])
    df['zero_lunar_illum'] = df['Vflag_bin'].apply(lambda x: x[5])
    # df['day_night_term'] = df['Vflag_bin'].apply(lambda x: x[6:8])
    # df['fire_detect'] = df['Vflag_bin'].apply(lambda x: x[8:14])
    # df['stray_light'] = df['Vflag_bin'].apply(lambda x: x[14:16])
    # df['cloud2_rej'] = df['Vflag_bin'].apply(lambda x: x[18])
    # df['dnb_light'] = df['Vflag_bin'].apply(lambda x: x[22:24])
    # df['dnb_saa'] = df['Vflag_bin'].apply(lambda x: x[24])
    # df['no_data'] = df['Vflag_bin'].apply(lambda x: x[31])
    #------Add Satellite Zenith Angle------
    # df = pd.merge(df, dz, on='Sample_DNB')
    #-----Filter out cloud covered and lunar illum data points-----
    # df = df[(df.flag_cloud2 == '00') & (df.zero_lunar_illum == '1')]
    # df = df.drop(columns = {"flag_cloud2","zero_lunar_illum","Vflag_bin"})
    # Output filtered and unfiltered dataframes
    return df

def create_daily_database(path):
    filelist = os.listdir(path)
    df_to_append = []
    for filename in filelist:
        if filename == ".DS_Store":
            continue
        print("Reading {}".format(filename))
        df = pd.read_csv(path + filename)
        db = preprocess_df(df)
        print("Done")
        df_to_append.append(db)
    apd = pd.concat(df_to_append)
    code.interact(local = locals())
    return appended_data

def yearly_and_monthly_data_summary(do):
    do = do[["id", "date_c", "flag_cloud2", "zero_lunar_illum"]].drop_duplicates()
    do["year"] = do.date_c.dt.year
    do["month"] = do.date_c.dt.month
    do["day"] = do.date_c.dt.day

    ################################################
    ## data for yearly analysis
    dtotal = do.groupby(["year"]).nunique()[["date_c"]]
    dtotal = dtotal.rename(columns={"date_c":"total_days"})

    dc = do[(do.flag_cloud2 == '00')]
    dc = dc.groupby(["year"]).nunique()[["date_c"]]
    dc = dc.rename(columns={"date_c":"no_cloud"})

    dl = do[(do.zero_lunar_illum == '1')]
    dl = dl.groupby(["year"]).nunique()[["date_c"]]
    dl = dl.rename(columns={"date_c":"no_lunar"})

    db = do[(do.flag_cloud2 == '00') & (do.zero_lunar_illum == '1')]
    db = db.groupby(["year"]).nunique()[["date_c"]]
    db = db.rename(columns={"date_c":"no_cloud_lunar"})

    g = [dtotal, dc, dl, db]

    df = pd.concat(g, axis=1, sort=False)
    df["perc_cloud"] = df["no_cloud"]*100.0/df["total_days"]
    df["perc_lunar"] = df["no_lunar"]*100.0/df["total_days"]
    df["perc_both"] = df["no_cloud_lunar"]*100.0/df["total_days"]

    ################################################
    ## data for monthly analysis
    # dtotal = do.groupby(["month"]).nunique()[["date_c"]]
    # dtotal = dtotal.rename(columns={"date_c":"total_days"})

    # dc = do[(do.flag_cloud2 == '00')]
    # dc = dc.groupby(["month"]).nunique()[["date_c"]]
    # dc = dc.rename(columns={"date_c":"no_cloud"})

    # dl = do[(do.zero_lunar_illum == '1')]
    # dl = dl.groupby(["month"]).nunique()[["date_c"]]
    # dl = dl.rename(columns={"date_c":"no_lunar"})

    # db = do[(do.flag_cloud2 == '00') & (do.zero_lunar_illum == '1')]
    # db = db.groupby(["month"]).nunique()[["date_c"]]
    # db = db.rename(columns={"date_c":"no_cloud_lunar"})

    # g = [dtotal, dc, dl, db]

    # df = pd.concat(g, axis=1, sort=False)
    # df["perc_cloud"] = df["no_cloud"]*100.0/df["total_days"]
    # df["perc_lunar"] = df["no_lunar"]*100.0/df["total_days"]
    # df["perc_both"] = df["no_cloud_lunar"]*100.0/df["total_days"]

    # code.interact(local=locals())

    ################################################
    ## Plot type 1
    # plt.figure
    # plt.plot(df.index, df.perc_cloud, marker="o", color="r", label="cloud filter")
    # plt.plot(df.index, df.perc_lunar, marker="^", color="b", label="lunar filter")
    # plt.plot(df.index, df.perc_both, marker="s", color="g", label="cloud + lunar filter")
    # plt.ylim([0,101])
    # plt.xticks(np.arange(1, 13, 1))
    # plt.yticks(np.arange(0, 101, 10))
    # plt.legend()
    # # plt.xlabel("Timeline(year)")
    # plt.xlabel("Month")
    # plt.ylabel("Remaining Days(%)")
    # plt.title("Proportion of days remaining in the dataset post-filtering by month")
    # plt.show()

    ## Plot type 2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(df.index, df.perc_cloud, marker="o", color="r", label="cloud filter")
    ax.plot(df.index, df.perc_lunar, marker="^", color="b", label="lunar filter")
    ax.plot(df.index, df.perc_both, marker="s", color="g", label="cloud + lunar filter")
    ax.set_ylim([0,100])
    ax.set_yticks(np.arange(0, 101, 10))
    ax.legend()
    ax.set_xlabel("Timeline(year)")
    ax.set_ylabel("Remaining Days(%)")
    ax2 = ax.twinx()
    ax2.bar(df.index, df.total_days, alpha=0.2)
    ax2.set_ylabel("Total number of coverage days")
    plt.title("Proportion of days remaining in the dataset after filtering")
    plt.show()

    ## Plot type 3
    ## Plotting reference: https://stackoverflow.com/questions/22833404/how-do-i-plot-hatched-bars-using-pandas
    # df1 = df[["total_days","no_cloud","no_lunar","no_cloud_lunar"]]
    # ax = plt.figure(figsize=(10, 6)).add_subplot(111)
    # df1.plot(ax=ax, kind='bar', legend=False, rot=0, edgecolor="black", color="white")
    # # ax.legend(["Total Coverage", "Clear sky", "Zero lunar illuminance", "Zero lunar illuminance + clear sky"])
    # bars = ax.patches
    # hatches = ''.join(h*len(df) for h in '.-/\\')
    # for bar, hatch in zip(bars, hatches):
    #     bar.set_hatch(hatch)
    # ax.legend(loc='center right', bbox_to_anchor=(1, 1), ncol=4)
    # ax.legend(["Total Coverage", "Clear sky", "Zero lunar illuminance", "Zero lunar illuminance + clear sky"])
    # ax.set_ylabel("Days")
    # ax.set_xlabel("Year")
    # plt.title("Total number of coverage, cloud free, zero lunar illuminance, cloud free/zero lunar illuminance days per year")
    # plt.show()

    code.interact(local = locals())
    return None

def time_of_capture(do):
    do["st_int"] = pd.to_datetime(do["start_time"]).astype(np.int64)
    do["et_int"] = pd.to_datetime(do["end_time"]).astype(np.int64)
    do["duration_int"] = pd.to_timedelta(do["end_time"] - do["start_time"]).astype(np.int64)
    res = do.groupby("id").agg(["mean","std"])
    res.columns = ['_'.join(c) for c in res.columns.values]

    # calculate overall values for complete region
    overall_st_mean = pd.to_datetime(res.st_int_mean.mean())
    overall_st_std = pd.to_timedelta(res.st_int_std.mean())
    overall_et_mean = pd.to_datetime(res.et_int_mean.mean())
    overall_et_std = pd.to_timedelta(res.et_int_std.mean())
    overall_duration_mean = pd.to_datetime(res.duration_int_mean.mean())
    overall_duration_std = pd.to_timedelta(res.duration_int_std.mean())

    # compute spatial values for every grid cell
    res['st_mean'] = pd.to_datetime(res['st_int_mean'])
    res['st_std'] = pd.to_timedelta(res['st_int_std'])
    res['et_mean'] = pd.to_datetime(res['et_int_mean'])
    res['et_std'] = pd.to_timedelta(res['et_int_std'])
    res['duration_mean'] = pd.to_datetime(res['duration_int_mean'])
    res['duration_std'] = pd.to_timedelta(res['duration_int_std'])
    res = res[["st_mean", "st_std", "et_mean", "et_std", "duration_mean", "duration_std"]]

    code.interact(local=locals())

    return None

def proportion_of_points_after_filtering(df):
    def get_proportions(dk):
        # Input: Bad days dataframe
        dk = dk.groupby("date_c").nunique()[["id"]].reset_index()
        dk = dk.rename(columns={"id":"actual_points"})
        dk["perc_points"] = dk["actual_points"]*100.0/3285 #3285 is total no. of points in the grid
        return dk

    def create_cdf(dk):
        dk = dk.sort_values("perc_points")
        dk = dk.reset_index()
        dk = dk.drop(columns={"index"})
        dk["no_of_days"] = dk.index
        return dk

    def plot_cdf(dk,title):
        plt.figure()
        plt.plot(dk.no_of_days, dk.perc_points)
        plt.xlabel("No. of bad days")
        plt.ylabel("Proportion of points contained (%)")
        plt.title(title)
        plt.show()
        return None

    # create dataframes that contain data with clouds/lunar/cloud+lunar
    dc = df[(df.flag_cloud2 != '00')]
    dl = df[(df.zero_lunar_illum != '1')]
    db = df[(df.flag_cloud2 != '00') | (df.zero_lunar_illum != '1')]

    dc = get_proportions(dc)
    dl = get_proportions(dl)
    db = get_proportions(db)

    dc = create_cdf(dc)
    dl = create_cdf(dl)
    db = create_cdf(db)

    plot_cdf(dc, title="CDF - Proportion of grid points by no. of bad (cloud) days")
    plot_cdf(dl, title="CDF - Proportion of grid points by no. of bad (lunar) days")
    plot_cdf(db, title="CDF - Proportion of grid points by no. of bad (cloud + lunar) days")

    code.interact(local=locals())
    return None

def yearly_monthly_daily_boxplot(db, column):
    # take mean of radiance when there are multiple readings for the same date
    db = db.groupby(["id","date_c"]).mean()[[column]].reset_index()
    db = db[(db.date_c.dt.year<2015)]
    dk = db.copy()

    ## compare variation in TNL when we use daily aggregates, monthly aggregates and yearly
    db = db.groupby("date_c").sum()[[column]].reset_index()

    dy = db.resample("1Y",on="date_c").mean()[[column]].reset_index()
    dy = dy[(~dy[column].isnull())]

    dm = db.resample("1M",on="date_c").mean()[[column]].reset_index()
    dm = dm[(~dm[column].isnull())]

    dd = db.resample("1D",on="date_c").mean()[[column]].reset_index()
    dd = dd[(~dd[column].isnull())]

    # code.interact(local=locals())
    fontsize=14
    fig, ax = plt.subplots(figsize=(6,3))
    data = [dy[column], dm[column], dd[column]]
    ax.boxplot(data)
    ax.set_xticklabels(["Annual","Monthly","Nightly"], fontsize=fontsize)
    ax.set_ylabel("Total Nighttime Lights", fontsize=fontsize)
    ax.set_xlabel("NL Composites", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.tight_layout()
    plt.show()
    code.interact(local=locals())
    return None

if __name__=='__main__':
    re = "../sanaa_csv/"

    #---create & pickle monthly database------
    # dg = create_daily_database(re)
    #---------------------------------------------------------
    # read and pre-process unfiltered and unnormalized data
    do = pd.read_hdf("overall_unfiltered_unnormalized_data.hdf5", key="zeal")
    #clip negative radiance values to 0
    do["RadE9_DNB"] = do["RadE9_DNB"].clip(lower=0)
    # combine multiple readings for same day and same id
    do = do.groupby(["id","Latitude","Longitude","date_c","flag_cloud2","zero_lunar_illum"]).mean()[["RadE9_DNB"]].reset_index()

    #---------------------------------------------------------
    # read and pre-process totally filtered and normalized data
    df = pd.read_hdf("filtered_data.h5", key="zeal")
    #clip negative radiance values to 0
    df["RadE9_Mult_Nadir_Norm"] = df["RadE9_Mult_Nadir_Norm"].clip(lower=0)
    # combine multiple readings for same day and same id
    df = df.groupby(["id","Latitude","Longitude","date_c"]).mean()[["RadE9_Mult_Nadir_Norm","LI"]].reset_index()

    #---------------------------------------------------------
    # plot showing importance of daily data over monthly, and yearly data
    yearly_monthly_daily_boxplot(df, column="RadE9_Mult_Nadir_Norm") #we are using filtered dataset for now

    #---------------------------------------------------------
    # plot showing limitation of lunar and cloud based filtering
    # yearly_and_monthly_data_summary(do)


    # time_of_capture(do)
    # proportion_of_points_after_filtering(do)