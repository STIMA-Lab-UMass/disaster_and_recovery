import pandas as pd
import geopandas as gpd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import code
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from scipy.signal import butter, lfilter
from scipy.signal import freqz

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
    # df['fire_detect'] = df['Vflag_bin'].apply(lambda x: x[8:14])
    # df['stray_light'] = df['Vflag_bin'].apply(lambda x: x[14:16])
    #-----Filter out cloud covered and lunar illum data points-----
    # df = df[(df.flag_cloud2 == '00') & (df.zero_lunar_illum == '1')]
    # df = df.drop(columns = {"flag_cloud2","zero_lunar_illum","Vflag_bin"})
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
    dn = pd.concat(df_to_append)
    try:
        dn.to_hdf("sanaa_unfiltered_nadir_norm.h5", key="zeal")
        print("File saved")
    except:
        print("Couldn't save")
        code.interact(local=locals())
    return dn

def find_lowrad_points(df):
    # only select readings with no lunar illuminance
    dk = df[(df.zero_lunar_illum == "1")]
    # only select readings from 2012 to 2014
    dk = dk[(dk.date_c.dt.year <= 2014)]
    # calculate mean radiance per id
    dk = dk.groupby("id").mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()
    # select readings with mean radiance less than 0.2
    dk = dk[(dk.RadE9_Mult_Nadir_Norm < 0.3)]
    # step 3: extract an array of location id
    id_arr = dk.id.unique()
    return id_arr

def visualize_lunar_effect(df, year):
    df = df.sort_values(by="date_c")
    df = df[(df.date_c.dt.year == year)]
    # df = df[(df.date_c.dt.month == 6)]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(df.date_c, df.RadE9_Mult_Nadir_Norm)
    ax1.set_ylabel("Radiance")
    ax2 = ax1.twinx()
    ax2.scatter(df.date_c, df.LI, s=2, color="r")
    ax2.set_ylabel("LI")
    plt.show()
    return None

def seasonal_decomposition(df, year, column):
    df = df[(df.date_c.dt.year == year)]
    # df = df[(df.date_c.dt.month <= 6)]
    df = df.sort_values(by="date_c")
    df = df.resample("1D", on="date_c").mean()[[column]]
    df = df.interpolate(method="time")
    series = df[column]
    result = seasonal_decompose(series, period=29, model="additive")
    print("{}".format(result.trend.mean()))
    result.plot()
    plt.show()
    code.interact(local=locals())
    return None

def STL_decomposition(df, column, year, title):
    df = df[(df.date_c.dt.year == year)]
    df = df.sort_values(by="date_c")
    df = df[["date_c", column]]
    df = df.resample("1D", on="date_c").mean()[[column]]
    df = df.interpolate(method="time")
    series = df[column]

    stl = STL(series, period=29, robust=True)
    res = stl.fit()

    print("Trend mean = {}".format(res.trend.mean()))
    # fig = res.plot()
    # fig.suptitle(title)
    # plt.show()
    # code.interact(local=locals())
    return res

def seasonal_component_for_all(df, id_arr):
    res_comb = []
    # for pt in id_arr:
    for pt in df.id.unique():
        print(pt)
        for year in [2012,2013,2014,2015,2016,2017,2018,2019]:
            print(year)
            res = STL_decomposition(df[(df.id==pt)], column="RadE9_Mult_Nadir_Norm", year=year, title=None)
            res_db= pd.concat([res.observed,res.trend, res.seasonal, res.resid], axis=1).reset_index()
            res_db["id"] = pt
            res_db["year"] = year
            res_comb.append(res_db)

    res_comb = pd.concat(res_comb, axis=0)
    res_comb.to_hdf("STL_decomposition_output_2012_2019_allpoints.h5", key="zeal")
    code.interact(local=locals())
    return None

def relative_radiance_levels_only_lunar_affected(df, dr, year, how):
    dr = dr[(dr.year == year)]
    df = df[(df.date_c.dt.year == year)]
    df_lunar = df[(df.zero_lunar_illum == "0")]
    df_nolunar = df[(df.zero_lunar_illum == "1")]

    # dr["season"] = dr["season"].clip(lower = 0)
    # dr["resid"] = dr["resid"].clip(lower = 0)
    dr = dr.groupby("date_c").agg({"season":"mean", "resid":"mean", "RadE9_Mult_Nadir_Norm":"mean"}).reset_index()
    dr.columns = ["date_c", "season", "resid", "observed"]
    dfr_lunar = pd.merge(df_lunar, dr, left_on=["date_c"], right_on=["date_c"], how="left")

    if how=="STL":
        dfr_lunar["RadE9_Mult_Nadir_Norm"] = dfr_lunar["RadE9_Mult_Nadir_Norm"] - dfr_lunar["season"] - dfr_lunar["resid"]
    elif how=="radiance":
        dfr_lunar["RadE9_Mult_Nadir_Norm"] = dfr_lunar["RadE9_Mult_Nadir_Norm"] - dfr_lunar["observed"]
    else:
        print("Please select a correct radiance correction mechanism!")

    dfr_lunar = dfr_lunar.rename(columns = {"RadE9_Mult_Nadir_Norm":"rad_corr"})
    dfr_lunar["rad_corr"] = dfr_lunar["rad_corr"].clip(lower=0)

    df_new = pd.merge(df, dfr_lunar[["id","date_c","rad_corr"]], on=["id", "date_c"], how="left")
    df_new["rad_corr"] = np.where(df_new["zero_lunar_illum"]=="0", df_new["rad_corr"], df_new["RadE9_Mult_Nadir_Norm"])

    # nolunar_avg_before_corr = df_new[(df_new.zero_lunar_illum == "1")].groupby("id").mean()[["RadE9_Mult_Nadir_Norm"]]
    # nolunar_avg_before_corr = nolunar_avg_before_corr.rename(columns={"RadE9_Mult_Nadir_Norm":"nolunar_nocorr"})

    # lunar_avg_before_corr = df_new[(df_new.zero_lunar_illum == "0")].groupby("id").mean()[["RadE9_Mult_Nadir_Norm"]]
    # lunar_avg_before_corr = lunar_avg_before_corr.rename(columns={"RadE9_Mult_Nadir_Norm":"lunar_nocorr"})

    # nolunar_avg_after_corr = df_new[(df_new.zero_lunar_illum == "1")].groupby("id").mean()[["rad_corr"]]
    # nolunar_avg_after_corr = nolunar_avg_after_corr.rename(columns={"rad_corr":"nolunar_corr"})

    # lunar_avg_after_corr = df_new[(df_new.zero_lunar_illum == "0")].groupby("id").mean()[["rad_corr"]]
    # lunar_avg_after_corr = lunar_avg_after_corr.rename(columns={"rad_corr":"lunar_corr"})

    # overall_avg_before_corr = df_new.groupby("id").mean()[["RadE9_Mult_Nadir_Norm"]]
    # overall_avg_before_corr = overall_avg_before_corr.rename(columns={"RadE9_Mult_Nadir_Norm":"overall_nocorr"})

    # overall_avg_after_corr = df_new.groupby("id").mean()[["rad_corr"]]
    # overall_avg_after_corr = overall_avg_after_corr.rename(columns={"rad_corr":"overall_corr"})

    # df_summary = pd.concat([nolunar_avg_before_corr, nolunar_avg_after_corr, lunar_avg_before_corr, lunar_avg_after_corr, overall_avg_before_corr, overall_avg_after_corr], axis=1, sort=False)

    # code.interact(local=locals())

    # gridcell_id = "sanaa_grid_10_69"
    # title = "{} - Only lunar affected readings [Method={}]".format(gridcell_id, how)
    # STL_decomposition(df_new[(df_new.id == gridcell_id)], "rad_corr", year, title)

    # dk = df_new[(df_new.id == gridcell_id)]
    # dk1 = dk[(dk.zero_lunar_illum == "0")]
    # fig, ax = plt.subplots()
    # plt.scatter(dk.date_c, dk.RadE9_Mult_Nadir_Norm, label="original")
    # plt.scatter(dk.date_c, dk.rad_corr, label="corrected")
    # plt.hlines(dk1.rad_corr.mean(), dk.date_c.min(), dk.date_c.max(), label="mean corr rad")
    # plt.vlines(dk1.date_c, dk.rad_corr.min(), dk.rad_corr.max(), label="LI")
    # # plt.plot(dr.date_c, dr.season, label="seasonal")
    # plt.legend()
    # plt.show()

    return df_new

def relative_radiance_levels_all_readings(df, dr, year, how):
    dr = dr[(dr.year == year)]
    df = df[(df.date_c.dt.year == year)]

    dr = dr.groupby("date_c").agg({"season":"mean", "resid":"mean", "RadE9_Mult_Nadir_Norm":"mean"}).reset_index()
    dr.columns = ["date_c", "season", "resid", "observed"]
    dfr_lunar = pd.merge(df, dr, left_on=["date_c"], right_on=["date_c"], how="left")

    if how=="STL":
        dfr_lunar["rad_corr"] = dfr_lunar["RadE9_Mult_Nadir_Norm"] - dfr_lunar["season"] - dfr_lunar["resid"]
    elif how=="radiance":
        dfr_lunar["rad_corr"] = dfr_lunar["RadE9_Mult_Nadir_Norm"] - dfr_lunar["observed"]
    else:
        print("Please select a correct radiance correction mechanism!")

    dfr_lunar["rad_corr"] = dfr_lunar["rad_corr"].clip(lower=0)
    df_new = dfr_lunar.copy()

    # nolunar_avg_before_corr = df_new[(df_new.zero_lunar_illum == "1")].groupby("id").mean()[["RadE9_Mult_Nadir_Norm"]]
    # nolunar_avg_before_corr = nolunar_avg_before_corr.rename(columns={"RadE9_Mult_Nadir_Norm":"nolunar_nocorr"})

    # lunar_avg_before_corr = df_new[(df_new.zero_lunar_illum == "0")].groupby("id").mean()[["RadE9_Mult_Nadir_Norm"]]
    # lunar_avg_before_corr = lunar_avg_before_corr.rename(columns={"RadE9_Mult_Nadir_Norm":"lunar_nocorr"})

    # nolunar_avg_after_corr = df_new[(df_new.zero_lunar_illum == "1")].groupby("id").mean()[["rad_corr"]]
    # nolunar_avg_after_corr = nolunar_avg_after_corr.rename(columns={"rad_corr":"nolunar_corr"})

    # lunar_avg_after_corr = df_new[(df_new.zero_lunar_illum == "0")].groupby("id").mean()[["rad_corr"]]
    # lunar_avg_after_corr = lunar_avg_after_corr.rename(columns={"rad_corr":"lunar_corr"})

    # overall_avg_before_corr = df_new.groupby("id").mean()[["RadE9_Mult_Nadir_Norm"]]
    # overall_avg_before_corr = overall_avg_before_corr.rename(columns={"RadE9_Mult_Nadir_Norm":"overall_nocorr"})

    # overall_avg_after_corr = df_new.groupby("id").mean()[["rad_corr"]]
    # overall_avg_after_corr = overall_avg_after_corr.rename(columns={"rad_corr":"overall_corr"})

    # df_summary = pd.concat([nolunar_avg_before_corr, nolunar_avg_after_corr, lunar_avg_before_corr, lunar_avg_after_corr, overall_avg_before_corr, overall_avg_after_corr], axis=1, sort=False)

    # gridcell_id = "sanaa_grid_10_69"
    # title = "{} - All readings [Method={}]".format(gridcell_id, how)
    # STL_decomposition(df_new[(df_new.id==gridcell_id)], "rad_corr", year, title)

    # dk = df_new[(df_new.id == gridcell_id)]
    # dk1 = dk[(dk.zero_lunar_illum == "0")]
    # fig, ax = plt.subplots()
    # plt.scatter(dk.date_c, dk.RadE9_Mult_Nadir_Norm, label="original")
    # plt.scatter(dk.date_c, dk.rad_corr, label="corrected")
    # plt.hlines(dk1.rad_corr.mean(), dk.date_c.min(), dk.date_c.max(), label="mean corr rad")
    # plt.vlines(dk1.date_c, dk.rad_corr.min(), dk.rad_corr.max(), label="LI")
    # # plt.plot(dr.date_c, dr.season, label="seasonal")
    # plt.legend()
    # plt.show()

    # code.interact(local=locals())

    return df_new

def create_basedb_for_correlation_studies(dg, dr):
    # creates a database to be used for correlation studies between original signal's seasonality and new seasonality after decomposition
    # res_comb = []
    # for yr in [2012,2013,2014,2015,2016,2017,2018,2019]:
    #     print("*****************************************")
    #     print("YEAR = {}".format(yr))
    #     do_s = relative_radiance_levels_only_lunar_affected(dg, dr, year=yr, how="STL")
    #     do_r = relative_radiance_levels_only_lunar_affected(dg, dr, year=yr, how="radiance")
    #     da_s = relative_radiance_levels_all_readings(dg, dr, year=yr, how="STL")
    #     da_r = relative_radiance_levels_all_readings(dg, dr, year=yr, how="radiance")
    #     print("Relative radiance extraction DONE!")

    #     for gridcell_id in dg.id.unique():
    #         print("{} for year {}".format(gridcell_id, yr))
    #         do_s_decomp = STL_decomposition(do_s[(do_s.id == gridcell_id)], column="rad_corr", year=yr, title=None)
    #         do_r_decomp = STL_decomposition(do_r[(do_r.id == gridcell_id)], column="rad_corr", year=yr, title=None)
    #         da_s_decomp = STL_decomposition(da_s[(da_s.id == gridcell_id)], column="rad_corr", year=yr, title=None)
    #         da_r_decomp = STL_decomposition(da_r[(da_r.id == gridcell_id)], column="rad_corr", year=yr, title=None)
    #         res_db = pd.concat([do_s_decomp.observed, do_s_decomp.trend, do_s_decomp.seasonal, do_s_decomp.resid, do_r_decomp.observed, do_r_decomp.trend, do_r_decomp.seasonal, do_r_decomp.resid, da_s_decomp.observed, da_s_decomp.trend, da_s_decomp.seasonal, da_s_decomp.resid, da_r_decomp.observed, da_r_decomp.trend, da_r_decomp.seasonal, da_r_decomp.resid], axis=1).reset_index()
    #         res_db["id"] = gridcell_id
    #         res_db["year"] = yr
    #         res_comb.append(res_db)

    # res_comb = pd.concat(res_comb, axis=0)
    # try:
    #     res_comb.columns = ["date_c", "os_rad", "os_trend", "os_season", "os_resid", "or_rad", "or_trend", "or_season", "or_resid", "as_rad", "as_trend", "as_season", "as_resid", "ar_rad", "ar_trend", "ar_season", "ar_resid", "id", "year"]
    #     res_comb.to_hdf("data_for_seasonality_correlation_studies.h5", key="zeal")
    #     print("HDF5 file saved. ")
    #     code.interact(local=locals())
    # except:
    #     print("!-----------------------!")
    #     print("Some error occured")
    #     code.interact(local=locals())
    df = pd.read_hdf("data_for_seasonality_correlation_studies.h5", key="zeal")
    df["year"] = df.date_c.dt.year
    df = pd.merge(dr, df, left_on=["id","year","date_c"], right_on=["id","year","date_c"])
    # code.interact(local=locals())
    return df

def correlation_between_seasonalities(d_corr, id_arr):
    # We are just using low rad points to study correlation among seasonalities
    rad_corr_arr = []
    season_corr_arr = []
    for year in [2012,2013,2014,2015,2016,2017,2018,2019]:
        dcorr = d_corr[(d_corr.year == year)]

        for gridcell_id in id_arr:
            dc = dcorr[(dcorr.id == gridcell_id)]

            rad_corr = dc[["year","date_c","RadE9_Mult_Nadir_Norm","os_rad","or_rad","as_rad","ar_rad"]].corr(method="pearson")
            rad_corr_vals = rad_corr[(rad_corr.index == "RadE9_Mult_Nadir_Norm")]
            rc_db = pd.DataFrame(columns = {"os_rad","or_rad","as_rad","ar_rad"})
            rc_db = rc_db.append({"os_rad":rad_corr_vals["os_rad"][0], "or_rad":rad_corr_vals["or_rad"][0], "as_rad":rad_corr_vals["as_rad"][0], "ar_rad":rad_corr_vals["ar_rad"][0]}, ignore_index=True)
            rc_db["id"] = gridcell_id
            rc_db["year"] = year
            rad_corr_arr.append(rc_db)

            season_corr = dc[["year","date_c","season","os_season","or_season","as_season","ar_season"]].corr(method="pearson")
            season_corr_vals = season_corr[(season_corr.index == "season")]
            sc_db = pd.DataFrame(columns = {"os_season","or_season","as_season","ar_season"})
            sc_db = sc_db.append({"os_season":season_corr_vals["os_season"][0], "or_season":season_corr_vals["or_season"][0], "as_season":season_corr_vals["as_season"][0], "ar_season":season_corr_vals["ar_season"][0]}, ignore_index=True)
            sc_db["id"] = gridcell_id
            sc_db["year"] = year
            season_corr_arr.append(sc_db)

    dr = pd.concat(rad_corr_arr, axis=0).reset_index(drop=True)
    dr[(dr.columns[dr.dtypes != np.object])] = dr[(dr.columns[dr.dtypes != np.object])].abs()
    # code.interact(local=locals())
    dr["best_tech"] = dr[["os_rad","or_rad","as_rad","ar_rad"]].idxmin(axis=1)
    print("******************************")
    print(dr.best_tech.value_counts())
    print(dr.groupby("year").best_tech.value_counts())

    ds = pd.concat(season_corr_arr, axis=0).reset_index(drop=True)
    ds[ds.columns[ds.dtypes != np.object]] = ds[ds.columns[ds.dtypes != np.object]].abs()
    ds["best_tech"] = ds[["or_season","as_season","os_season","ar_season"]].idxmin(axis=1)
    print("******************************")
    print(ds.best_tech.value_counts())
    print(ds.groupby("year").best_tech.value_counts())

    code.interact(local=locals())
    return None

def data_correction(dg, dr):
    do_arr = []
    da_arr = []
    for yr in [2012,2013,2014,2015,2016,2017,2018,2019]:
        print("*****************************************")
        print("YEAR = {}".format(yr))
        do_arr.append(relative_radiance_levels_only_lunar_affected(dg, dr, year=yr, how="radiance"))
        da_arr.append(relative_radiance_levels_all_readings(dg, dr, year=yr, how="STL"))
    do = pd.concat(do_arr, axis=0).reset_index(drop=True)
    da = pd.concat(da_arr, axis=0).reset_index(drop=True)
    return do, da

def meanrad_vs_auc(dm, df, id_arr, year, column, analysis_type, only_lowrad_points=False):
    ###############################SEASONALITY DATA################################
    if only_lowrad_points == True:
        dm = dm[(dm.id.isin(id_arr))]
        df = df[(df.id.isin(id_arr))]

    # df[column] = df[column].abs() #make all seasonality values positive so that we get better idea about AUC

    df = df[(df.year == year)]
    # code.interact(local=locals())
    # df[column] = df.groupby("id")[column].transform(lambda x: (x-min(x))/(max(x)-min(x)))

    df_count = df.groupby("id").count()[["date_c"]].reset_index()
    arr_365 = df_count[(df_count.date_c == 365)].id.values
    arr_364 = df_count[(df_count.date_c == 364)].id.values
    x_365 = np.arange(365)
    x_364 = np.arange(364)
    # code.interact(local=locals())
    if analysis_type == "AUC":
        df_auc_365 = df[(df.id.isin(arr_365))].groupby("id")[column].apply(lambda a: np.trapz(a,x_365)).reset_index()
        df_auc_364 = df[(df.id.isin(arr_364))].groupby("id")[column].apply(lambda a: np.trapz(a,x_364)).reset_index()
    elif analysis_type == "amplitude":
        df_auc_365 = df[(df.id.isin(arr_365))].groupby("id")[column].apply(lambda a: max(a)).reset_index()
        df_auc_364 = df[(df.id.isin(arr_364))].groupby("id")[column].apply(lambda a: max(a)).reset_index()

    df_auc = pd.concat([df_auc_365, df_auc_364], axis=0)
    df_auc.index = df_auc.id

    # db = pd.merge(df_auc, df_mr, left_on=["id"], right_on=["id"], how="left")

    ################################RADIANCE DATA##################################
    dm_nolunar = dm[(dm.date_c.dt.year==year) & (dm.zero_lunar_illum=="1")].groupby("id").mean()[["RadE9_Mult_Nadir_Norm"]]
    dm_nolunar = dm_nolunar.rename(columns={"RadE9_Mult_Nadir_Norm":"nolunar_rad"})

    dm_onlylunar = dm[(dm.date_c.dt.year==year) & (dm.zero_lunar_illum=="0")].groupby("id").mean()[["RadE9_Mult_Nadir_Norm"]]
    dm_onlylunar = dm_onlylunar.rename(columns={"RadE9_Mult_Nadir_Norm":"onlylunar_rad"})

    dm_overalllunar = dm[(dm.date_c.dt.year==year)].groupby("id").mean()[["RadE9_Mult_Nadir_Norm"]]
    dm_overalllunar = dm_overalllunar.rename(columns={"RadE9_Mult_Nadir_Norm":"overall_rad"})

    db = pd.concat([df_auc, dm_nolunar, dm_onlylunar, dm_overalllunar], axis=1, sort=False)
    db = db.reset_index(drop=True)

    # code.interact(local=locals())
    fig, ax = plt.subplots()
    plt.subplot(1,3,1)
    plt.scatter(db.overall_rad, db[column], s=4, alpha=0.6, color="r")
    # plt.title("AUC versus Mean (Overall) radiance")
    plt.title("Bandpass seasonality amplitude versus Mean (Overall) radiance")
    plt.xlabel("Mean radiance")
    plt.ylabel("Wave amplitude")

    plt.subplot(1,3,2)
    plt.scatter(db.nolunar_rad, db[column], s=4, alpha=0.6, color="b")
    # plt.title("AUC versus Mean (No lunar) radiance")
    plt.title("Bandpass seasonality amplitude versus Mean (No lunar) radiance")
    plt.xlabel("Mean radiance")
    plt.ylabel("Wave amplitude")

    plt.subplot(1,3,3)
    plt.scatter(db.onlylunar_rad, db[column], s=4, alpha=0.6, color="g")
    # plt.title("AUC versus Mean (Only lunar) radiance")
    plt.title("Bandpass seasonality amplitude versus Mean (Only lunar) radiance")
    plt.xlabel("Mean radiance")
    plt.ylabel("Wave amplitude")

    # fig.suptitle("Area under seasonality curve versus mean radiance (No Seasonality Scaling)")
    # fig.suptitle("Area under seasonality curve versus mean radiance (MinMax Seasonality Scaling)")
    fig.suptitle("(Year - {}) Bandpass seasonality wave amplitude versus mean radiance".format(year))
    plt.show()
    code.interact(local=locals())
    return None

def meanrad_vs_auc_combined(dm, df, id_arr, year, column, analysis_type, only_lowrad_points=False):
    ###############################SEASONALITY DATA################################
    if only_lowrad_points == True:
        dm = dm[(dm.id.isin(id_arr))]
        df = df[(df.id.isin(id_arr))]

    # df[column] = df[column].abs() #make all seasonality values positive so that we get better idea about AUC

    df = df[(df.year == year)]
    # code.interact(local=locals())
    # df[column] = df.groupby("id")[column].transform(lambda x: (x-min(x))/(max(x)-min(x)))

    df_count = df.groupby("id").count()[["date_c"]].reset_index()
    arr_365 = df_count[(df_count.date_c == 365)].id.values
    arr_364 = df_count[(df_count.date_c == 364)].id.values
    x_365 = np.arange(365)
    x_364 = np.arange(364)
    # code.interact(local=locals())
    if analysis_type == "AUC":
        df_auc_365 = df[(df.id.isin(arr_365))].groupby("id")[column].apply(lambda a: np.trapz(a,x_365)).reset_index()
        df_auc_364 = df[(df.id.isin(arr_364))].groupby("id")[column].apply(lambda a: np.trapz(a,x_364)).reset_index()
    elif analysis_type == "amplitude":
        df_auc_365 = df[(df.id.isin(arr_365))].groupby("id")[column].apply(lambda a: max(a)).reset_index()
        df_auc_364 = df[(df.id.isin(arr_364))].groupby("id")[column].apply(lambda a: max(a)).reset_index()

    df_auc = pd.concat([df_auc_365, df_auc_364], axis=0)
    df_auc.index = df_auc.id

    # db = pd.merge(df_auc, df_mr, left_on=["id"], right_on=["id"], how="left")

    ################################RADIANCE DATA##################################
    dm_nolunar = dm[(dm.date_c.dt.year==year) & (dm.zero_lunar_illum=="1")].groupby("id").mean()[["RadE9_Mult_Nadir_Norm"]]
    dm_nolunar = dm_nolunar.rename(columns={"RadE9_Mult_Nadir_Norm":"nolunar_rad"})

    dm_onlylunar = dm[(dm.date_c.dt.year==year) & (dm.zero_lunar_illum=="0")].groupby("id").mean()[["RadE9_Mult_Nadir_Norm"]]
    dm_onlylunar = dm_onlylunar.rename(columns={"RadE9_Mult_Nadir_Norm":"onlylunar_rad"})

    dm_overalllunar = dm[(dm.date_c.dt.year==year)].groupby("id").mean()[["RadE9_Mult_Nadir_Norm"]]
    dm_overalllunar = dm_overalllunar.rename(columns={"RadE9_Mult_Nadir_Norm":"overall_rad"})

    db = pd.concat([df_auc, dm_nolunar, dm_onlylunar, dm_overalllunar], axis=1, sort=False)
    db = db.reset_index(drop=True)

    # # code.interact(local=locals())
    # fig, ax = plt.subplots()
    # plt.subplot(1,3,1)
    # plt.scatter(db.overall_rad, db[column], s=4, alpha=0.6, color="r")
    # # plt.title("AUC versus Mean (Overall) radiance")
    # plt.title("Bandpass seasonality amplitude versus Mean (Overall) radiance")
    # plt.xlabel("Mean radiance")
    # plt.ylabel("Wave amplitude")

    # plt.subplot(1,3,2)
    # plt.scatter(db.nolunar_rad, db[column], s=4, alpha=0.6, color="b")
    # # plt.title("AUC versus Mean (No lunar) radiance")
    # plt.title("Bandpass seasonality amplitude versus Mean (No lunar) radiance")
    # plt.xlabel("Mean radiance")
    # plt.ylabel("Wave amplitude")

    # plt.subplot(1,3,3)
    # plt.scatter(db.onlylunar_rad, db[column], s=4, alpha=0.6, color="g")
    # # plt.title("AUC versus Mean (Only lunar) radiance")
    # plt.title("Bandpass seasonality amplitude versus Mean (Only lunar) radiance")
    # plt.xlabel("Mean radiance")
    # plt.ylabel("Wave amplitude")

    # # fig.suptitle("Area under seasonality curve versus mean radiance (No Seasonality Scaling)")
    # # fig.suptitle("Area under seasonality curve versus mean radiance (MinMax Seasonality Scaling)")
    # fig.suptitle("(Year - {}) Bandpass seasonality wave amplitude versus mean radiance".format(year))
    # plt.show()
    # code.interact(local=locals())
    return db

def boxplot_seasonality_variation(df, id_arr, year, only_lowrad_points=True):
    df = df[(df.year == year)]
    if only_lowrad_points == True:
        df = df[(df.id.isin(id_arr))]
    # df.season = df.season.abs()

    df.boxplot(column=["season"], by=["date_c"])
    plt.title("Variation in seasonal component over time (year 2013)")
    plt.xlabel("Day")

    # df.boxplot(column=["season"], by=["id"])
    # plt.title("Variation in seasonal component")
    # plt.xlabel("Low radiance grid point ID")
    plt.show()
    return None

def NL_fft(df, id_arr, year, column, only_lowrad_points=True):
    df = df[(df.year == year)]

    if only_lowrad_points == True:
        df = df[(df.id.isin(id_arr))]

    id_needs_correction = []
    for gridcell_id in df.id.unique():
        df1 = df[(df.id == gridcell_id)]
        x = len(df1)
        sp = np.fft.fft(df1[column].values)
        freq = np.fft.fftfreq(x)

        sp_abs = np.abs(sp.real) #deal with only real part of DFT
        sp_top_idx = np.argsort(sp_abs) #get indices in sorted order of values
        freq_maxenergy = [freq[idx] for idx in sp_top_idx[-6:]] #select freq corresponding to top 6 signal energies
        freq_maxenergy = np.unique(np.abs(freq_maxenergy)) #step to remove duplicate freq values (negative counterparts)
        freq_maxenergy = np.asarray([max(x, 0.00000000000000000000001) for x in freq_maxenergy]) #to avoid division by 0

        period_maxenergy = 1/freq_maxenergy
        period_maxenergy = period_maxenergy[(period_maxenergy>=27) & (period_maxenergy<=31)]
        print("{} - Probable Periodicity: {}".format(gridcell_id, period_maxenergy))

        if period_maxenergy.size >= 1:
            id_needs_correction.append(gridcell_id)

    print("Total no. of IDs: {}".format(len(id_needs_correction)))
    print("{}".format(set(id_arr) - set(id_needs_correction)))

    code.interact(local=locals())
    return None

def NL_autocorrelation(df, id_arr, year, column, only_lowrad_points=True):
    df = df[(df.year == year)]
    if only_lowrad_points == True:
        df = df[(df.id.isin(id_arr))]

    id_needs_correction = []
    for gridcell_id in df.id.unique():
        # gridcell_id = "sanaa_grid_0_6"
        df1 = df[(df.id == gridcell_id)]
        s = pd.Series(sm.tsa.acf(df1[column].values, nlags=90))
        s_idx = np.argsort(s.values)[-4:] #s_idx basically represents lag in days for which correlation was high
        s_idx = s_idx[(s_idx>27) & (s_idx<30)]
        if s_idx.size>=1:
            id_needs_correction.append(gridcell_id)

        print("{} - lag = {}".format(gridcell_id, s_idx))

    print("Total no. of IDs: {}".format(len(id_needs_correction)))
    print("{}".format(set(id_arr) - set(id_needs_correction)))
    # code.interact(local=locals())
    return None

def correlation_NL_LI(df, id_arr, year, column, only_lowrad_points=True, use_pickled_file=False):
    if use_pickled_file==False:
        df = df[(df.date_c.dt.year == year)]
        temp_arr = {}
        for gridcell_id in df.id.unique():
            df1 = df[(df.id == gridcell_id)].sort_values(by="date_c")
            df1 = df1[["RadE9_Mult_Nadir_Norm","LI"]].corr()
            temp_arr[gridcell_id] = df1["LI"]["RadE9_Mult_Nadir_Norm"]
            print("{} - {}".format(gridcell_id, temp_arr[gridcell_id]))

        id_corr = []
        for cell_id, cell_corr in temp_arr.items():
            if cell_corr >= 0.8:
                id_corr.append(cell_id)

        print("No. of IDs to be corrected = {}".format(len(id_corr)))
        print("{}".format(set(id_arr) - set(id_corr)))
        # pickle.dump(id_corr, open("./{}_ids_needing_correction.pck".format(year),"wb"))
        # print("Pickling done.")
    else:
        id_corr = pickle.load(open("./{}_ids_needing_correction.pck".format(year),"rb"))
    return id_corr

def correction_of_selected_ids(dg, ds, lowrad_id, id_corr, year, method, metric):
    if method == "relative_radiance":
        # create a dataframe of all IDs that "need to be corrected" in a given year and extract only the values during high lunar days
        dg_lunar = dg[(dg.date_c.dt.year == year) & (dg.id.isin(id_corr) & (dg.zero_lunar_illum == "0"))]
        # select only the low radiance points from the above created DF and then compute a mean radiance timeseries for the whole year
        mean_lunar_ts = dg_lunar[(dg_lunar.id.isin(lowrad_id))].groupby("date_c").mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()
        mean_lunar_ts.columns = ["date_c","correction_radiance"]
        dg_lunar = pd.merge(dg_lunar, mean_lunar_ts, left_on=["date_c"], right_on=["date_c"], how="left")
        # subtract mean timeseries from high lunar radiance of all the IDs that need to be corrected (CORRECTION)
        dg_lunar["corrected_rad"] = dg_lunar["RadE9_Mult_Nadir_Norm"] - dg_lunar["correction_radiance"]
        dg_lunar["corrected_rad"] = dg_lunar["corrected_rad"].clip(lower=0)
        # combine the new DF with original dataframe
        # combined dataframe contains corrected radiance values for IDs that needed correction on high lunar days
        dg_new = pd.merge(dg[(dg.date_c.dt.year==year)], dg_lunar[["id","date_c","corrected_rad"]], on=["id","date_c"], how="left")
        dg_new["corrected_rad"] = np.where(dg_new["zero_lunar_illum"]=="0", dg_new["corrected_rad"], dg_new["RadE9_Mult_Nadir_Norm"])
        # code.interact(local=locals())

    elif method == "STL_only_lunar":
        dg = dg[(dg.date_c.dt.year == year)]
        ds = ds[(ds.date_c.dt.year == year)]
        dc = pd.merge(dg, ds[["id","date_c","trend","season","resid"]], on=["id","date_c"], how="left")
        dc_lunar = dc[(dc.id.isin(id_corr)) & (dc.zero_lunar_illum == "0")]
        if metric == "seasonal":
            dc_lunar["corrected_rad"] = dc_lunar["RadE9_Mult_Nadir_Norm"] - dc["season"]
        elif metric == "absolute_seasonal":
            dc_lunar["corrected_rad"] = dc_lunar["RadE9_Mult_Nadir_Norm"] - dc["season"].abs()
        elif metric == "seasonal_and_residual":
            dc_lunar["corrected_rad"] = dc_lunar["RadE9_Mult_Nadir_Norm"] - dc["season"] - dc["resid"]
        elif metric == "absolute_seasonal_and_residual":
            dc_lunar["corrected_rad"] = dc_lunar["RadE9_Mult_Nadir_Norm"] - dc["season"].abs() - dc["resid"]
        elif metric == "absolute_seasonal_and_absolute_residual":
            dc_lunar["corrected_rad"] = dc_lunar["RadE9_Mult_Nadir_Norm"] - dc["season"].abs() - dc["resid"].abs()

        dc = pd.merge(dc, dc_lunar[["id","date_c","corrected_rad"]], on=["id","date_c"], how="left")
        dc["corrected_rad"] = np.where(dc["zero_lunar_illum"]=="0", dc["corrected_rad"], dc["RadE9_Mult_Nadir_Norm"])
        dg_new = dc.copy()
        # code.interact(local=locals())

    elif method == "STL_allreadings":
        dg = dg[(dg.date_c.dt.year == year)]
        ds = ds[(ds.date_c.dt.year == year)]
        dc = pd.merge(dg[(dg.id.isin(id_corr))], ds[(ds.id.isin(id_corr))][["id","date_c","trend","season","resid"]], on=["id","date_c"], how="left")
        dc["corrected_rad"] = dc["RadE9_Mult_Nadir_Norm"] - dc["season"] - dc["resid"]
        dg_new = pd.merge(dg, dc[["id","date_c","corrected_rad"]], on=["id","date_c"], how="left")
        # code.interact(local=locals())

    # dmean = dg[(dg.date_c.dt.year == year)].groupby("id").mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()
    # dmean = dmean[(dmean.id.isin(id_corr)) & (~dmean.id.isin(lowrad_id))]
    # dmean = dmean.sort_values(by=["RadE9_Mult_Nadir_Norm"])
    # code.interact(local=locals())

    return dg_new

def plots_for_studying_correction(dg, ds, id_arr, id_corr):
    # medrad_ids = ["sanaa_grid_32_12", "sanaa_grid_44_36", "sanaa_grid_16_9"]
    # lowrad_ids = ["sanaa_grid_4_72", "sanaa_grid_9_69", "sanaa_grid_0_63"]
    medrad_ids = ["sanaa_grid_32_12"]
    lowrad_ids = ["sanaa_grid_9_69"]
    metrics = ["seasonal","absolute_seasonal", "seasonal_and_residual", "absolute_seasonal_and_residual"]

    for cell_id in medrad_ids:
        fig, ax = plt.subplots()
        i = 1
        for metric in metrics:
            dg_new = correction_of_selected_ids(dg, ds, id_arr, id_corr, year=2013, method="STL_only_lunar", metric=metric)
            dk1 = dg_new[(dg_new.id == cell_id)]
            plt.subplot(2,3,i)
            plt.scatter(dk1.date_c, dk1.RadE9_Mult_Nadir_Norm, s=4, c="r", label="original")
            plt.scatter(dk1.date_c, dk1.corrected_rad, s=4, c="b", marker="^", label="corrected")
            plt.xlabel("Time (day)")
            plt.ylabel("Radiance")
            plt.legend()
            plt.title("{} (only lunar)".format(metric))
            i = i+1

        dg_new = correction_of_selected_ids(dg, ds, id_arr, id_corr, year=2013, method="STL_allreadings", metric=metric)
        dk1 = dg_new[(dg_new.id == cell_id)]
        plt.subplot(2,3,i)
        plt.scatter(dk1.date_c, dk1.RadE9_Mult_Nadir_Norm, s=4, c="r", label="original")
        plt.scatter(dk1.date_c, dk1.corrected_rad, s=4, c="b", marker="^", label="corrected")
        plt.xlabel("Time (day)")
        plt.ylabel("Radiance")
        plt.legend()
        plt.title("Correcting all readings")

        dg_new = correction_of_selected_ids(dg, ds, id_arr, id_corr, year=2013, method="relative_radiance", metric=metric)
        dk1 = dg_new[(dg_new.id == cell_id)]
        plt.subplot(2,3,i+1)
        plt.scatter(dk1.date_c, dk1.RadE9_Mult_Nadir_Norm, s=4, c="r", label="original")
        plt.scatter(dk1.date_c, dk1.corrected_rad, s=4, c="b", marker="^", label="corrected")
        plt.xlabel("Time (day)")
        plt.ylabel("Radiance")
        plt.legend()
        plt.title("Subtracting mean radiance (only lunar)")

        fig.suptitle("2013 - {} (LOW RADIANCE POINT)".format(cell_id))
        plt.show()

    return None

def boxplots_for_studying_correction(dg, ds, id_arr, id_corr):
    medrad_ids = ["sanaa_grid_32_12", "sanaa_grid_44_36", "sanaa_grid_16_9"]
    # lowrad_ids = ["sanaa_grid_4_72", "sanaa_grid_9_69", "sanaa_grid_0_63"]
    medrad_ids = ["sanaa_grid_32_12"]
    lowrad_ids = ["sanaa_grid_9_69"]
    metrics = ["seasonal","absolute_seasonal", "seasonal_and_residual", "absolute_seasonal_and_residual"]
    # metrics = ["seasonal"]

    for cell_id in medrad_ids:
        fig, ax = plt.subplots()
        i = 1
        for metric in metrics:
            dg_new = correction_of_selected_ids(dg, ds, id_arr, id_corr, year=2013, method="STL_only_lunar", metric=metric)
            dk1 = dg_new[(dg_new.id == cell_id)]
            plt.subplot(2,3,i)
            dk1.boxplot(column=["RadE9_Mult_Nadir_Norm","corrected_rad"])
            plt.ylabel("Radiance")
            plt.legend()
            plt.title("{} (only lunar)".format(metric))
            i = i+1

        dg_new = correction_of_selected_ids(dg, ds, id_arr, id_corr, year=2013, method="STL_allreadings", metric=metric)
        dk1 = dg_new[(dg_new.id == cell_id)]
        plt.subplot(2,3,i)
        dk1.boxplot(column=["RadE9_Mult_Nadir_Norm","corrected_rad"])
        plt.ylabel("Radiance")
        plt.legend()
        plt.title("Correcting all readings")

        dg_new = correction_of_selected_ids(dg, ds, id_arr, id_corr, year=2013, method="relative_radiance", metric=metric)
        dk1 = dg_new[(dg_new.id == cell_id)]
        plt.subplot(2,3,i+1)
        dk1.boxplot(column=["RadE9_Mult_Nadir_Norm","corrected_rad"])
        plt.ylabel("Radiance")
        plt.legend()
        plt.title("Subtracting mean radiance (only lunar)")

        fig.suptitle("2013 - {} (MEDIUM RADIANCE POINT)".format(cell_id))
        plt.show()

    return None

def butter_bandpass(lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def run_bandpassfiltering(dg, lowrad_id, id_corr, use_pickled=True):
    """ Reference: https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    """
    # medrad_ids = ["sanaa_grid_32_12", "sanaa_grid_44_36", "sanaa_grid_16_9"]
    # lowrad_ids = ["sanaa_grid_4_72", "sanaa_grid_9_69", "sanaa_grid_0_63"]

    # fs = 1.0 #sample per day
    # lowcut = 1/29.8
    # highcut = 1/27.0

    # temp_arr = []
    # for cell_id in dg.id.unique():
    #     print("******************************")
    #     print("ID - {}".format(cell_id))
    #     for year in [2012,2013,2014,2015,2016,2017,2018,2019]:
    #         print("year - {}".format(year))
    #         df = dg[(dg.date_c.dt.year == year) & (dg.id == cell_id)]
    #         df = df.sort_values(by="date_c")
    #         original_signal = df.RadE9_Mult_Nadir_Norm.values
    #         seasonality_signal = butter_bandpass_filter(original_signal, lowcut, highcut, fs, order=1)
    #         df["bandpass_output"] = seasonality_signal
    #         temp_arr.append(df[["id","date_c","zero_lunar_illum","LI","bandpass_output"]])
    #         del(df)
    # print("All years, all IDs DONE!")
    # try:
    #     db = pd.concat(temp_arr, axis=0)
    #     db.to_hdf("bandpass_allyears.h5", key="zeal")
    #     code.interact(local=locals())
    # except:
    #     print("something went wrong with concatenating or saving the file!")
    #     code.interact(local=locals())

    # if use_pickled == True:
    #     db = pd.read_hdf("bandpass_allyears.h5", key="zeal")
    # code.interact(local=locals())
    return db

def compare_bandpass_STL(dg, df, id_corr):

    # df1 = df[(df.date_c.dt.year == 2013)]
    # df1.boxplot(column=["season","bandpass_output"])
    # plt.title("Variation in decomposed seasonality signal and bandpass filter output signal")
    # plt.show()
    # df["year"] = df.date_c.dt.year
    column1 = "season"
    column2 = "bandpass_output"
    column3 = "bandpass_output2"

    analysis_type = "AUC"

    ds = meanrad_vs_auc_combined(dg, df, id_arr=None, year=2013, column=column1, analysis_type=analysis_type, only_lowrad_points=False)
    dk = meanrad_vs_auc_combined(dg, df, id_arr=None, year=2013, column=column2, analysis_type=analysis_type, only_lowrad_points=False)
    dk2 = meanrad_vs_auc_combined(dg, df, id_arr=None, year=2013, column=column3, analysis_type=analysis_type, only_lowrad_points=False)
    db = pd.merge(dk, ds[["id",column1]], left_on=["id"], right_on=["id"], how="left")
    db = pd.merge(db, dk2[["id", column3]], left_on=["id"], right_on=["id"], how="left")

    year=2013
    fig, ax = plt.subplots()
    plt.subplot(1,3,1)
    plt.scatter(db.overall_rad, db[column1], s=4, alpha=0.6, color="r", marker="*", label="STL")
    plt.scatter(db.overall_rad, db[column2], s=4, alpha=0.6, color="b", marker="^", label="bandpass (narrow)")
    plt.scatter(db.overall_rad, db[column3], s=4, alpha=0.6, color="g", marker="o", label="bandpass (broad)")
    # plt.title("AUC versus Mean (Overall) radiance")
    plt.legend()
    plt.title("seasonality {} versus Mean (Overall) radiance".format(analysis_type))
    plt.xlabel("Mean radiance")
    plt.ylabel("{}".format(analysis_type))
    plt.ylim([0,200])

    plt.subplot(1,3,2)
    plt.scatter(db.nolunar_rad, db[column1], s=4, alpha=0.6, color="r", marker="*", label="STL")
    plt.scatter(db.nolunar_rad, db[column2], s=4, alpha=0.6, color="b", marker="^", label="bandpass (narrow)")
    plt.scatter(db.nolunar_rad, db[column3], s=4, alpha=0.6, color="g", marker="o", label="bandpass (broad)")
    # plt.title("AUC versus Mean (No lunar) radiance")
    plt.legend()
    plt.title("seasonality {} versus Mean (No lunar) radiance".format(analysis_type))
    plt.xlabel("Mean radiance")
    plt.ylabel("{}".format(analysis_type))
    plt.ylim([0,200])

    plt.subplot(1,3,3)
    plt.scatter(db.onlylunar_rad, db[column1], s=4, alpha=0.6, color="r", marker="*", label="STL")
    plt.scatter(db.onlylunar_rad, db[column2], s=4, alpha=0.6, color="b", marker="^", label="bandpass (narrow)")
    plt.scatter(db.onlylunar_rad, db[column3], s=4, alpha=0.6, color="g", marker="o", label="bandpass (broad)")
    # plt.title("AUC versus Mean (Only lunar) radiance")
    plt.legend()
    plt.title("seasonality {} versus Mean (Only lunar) radiance".format(analysis_type))
    plt.xlabel("Mean radiance")
    plt.ylabel("{}".format(analysis_type))
    plt.ylim([0,200])

    fig.suptitle("(Year - {}) seasonality wave amplitude versus mean radiance".format(year))
    plt.show()

    code.interact(local=locals())
    return None

def basic_plots_bandpass(df, column, id, year):
    df1 = df[(df.id == id) & (df.date_c.dt.year == year)]
    plt.plot(df1.date_c, df1[column], label="raw radiance")
    plt.plot(df1.date_c, df1.season, label="season")
    plt.plot(df1.date_c, df1.bandpass_output, label="bandpass (narrow)")
    plt.plot(df1.date_c, df1.bandpass_output2, label="bandpass (broad)")
    plt.legend()
    plt.show()
    # code.interact(local=locals())
    return None

def correcting_data_using_bandpass(dg, df):
    dg_coords = dg[["id","Latitude","Longitude"]].drop_duplicates()
    df = pd.merge(df, dg_coords, left_on=["id"], right_on=["id"], how="left")
    df = pd.merge(df, dg[["id","date_c","LI","zero_lunar_illum"]], left_on=["id","date_c"], right_on=["id","date_c"], how="left")

    #clip bandpass output to 0 because many values are negative
    df["bandpass_output2"] = df["bandpass_output2"].clip(lower=0)
    df["rad_corr"] = df["RadE9_Mult_Nadir_Norm"] - df["bandpass_output2"]
    df["rad_corr"] = df["rad_corr"].clip(lower=0)

    # # create basic plots of pre and post correction radiance
    df1 = df[(df.id == "sanaa_grid_10_69") & (df.date_c.dt.year == 2013)]
    plt.plot(df1.date_c, df1["RadE9_Mult_Nadir_Norm"], marker="*", label="raw radiance")
    plt.plot(df1.date_c, df1["rad_corr"], marker="^", label="corrected radiance")
    plt.plot(df1.date_c, df1["bandpass_output2"], marker="o", label="bandpass op")
    plt.legend()
    plt.show()
    code.interact(local=locals())
    return None

if __name__=='__main__':
    re = "../sanaa_unfiltered_normalized_csv/"

    #---create & pickle monthly database------
    # dg = create_daily_database(re)
    #--------read & process necessary datasets----------
    dg = pd.read_hdf("sanaa_unfiltered_nadirnormalized.h5", key="zeal")
    # clip all the readings with negative radiance values to 0
    dg.RadE9_Mult_Nadir_Norm = dg.RadE9_Mult_Nadir_Norm.clip(lower=0)
    # combine multiple readings recorded on the same day
    dg = dg.groupby(["id","Latitude","Longitude","date_c","zero_lunar_illum"]).mean()[["RadE9_Mult_Nadir_Norm","LI"]].reset_index()
    # find an array of points with low radiance in 2012-2014 period
    id_arr = find_lowrad_points(dg)
    # code.interact(local=locals())

    # visualize_lunar_effect(dg[(dg.id=="sanaa_grid_9_66")], 2013)
    # seasonal_decomposition(dg[(dg.id=="sanaa_grid_9_66")], 2013, "RadE9_Mult_Nadir_Norm")
    # seasonal_decomposition(dg[(dg.id=="sanaa_grid_6_6")], 2013, "RadE9_Mult_Nadir_Norm")
    # gridcell_id = "sanaa_grid_10_69"
    # STL_decomposition(dg[(dg.id==gridcell_id)],"RadE9_Mult_Nadir_Norm", 2013, "{} original".format(gridcell_id))

    # seasonal_component_for_all(dg, id_arr)
    # dr = pd.read_hdf("STL_decomposition_output_2012_2019.h5", key="zeal")

    # do_s = relative_radiance_levels_only_lunar_affected(dg, dr, year=2013, how="STL")
    # do_r = relative_radiance_levels_only_lunar_affected(dg, dr, year=2013, how="radiance")
    # da_s = relative_radiance_levels_all_readings(dg, dr, year=2013, how="STL")
    # da_r = relative_radiance_levels_all_readings(dg, dr, year=2013, how="radiance")

    #------------------------------------------------------------------
    # create STL decomposition database for all the ids for all years
    # seasonal_component_for_all(dg, id_arr)
    # ds = pd.read_hdf("STL_bandpass_2012_2019_allpoints.h5", key="zeal")

    #------------------------------------------------------------------
    # study correlation between seasonalities
    # corr_db = create_basedb_for_correlation_studies(dg, dr)
    # correlation_between_seasonalities(corr_db, id_arr)

    #----------------FINAL STEP----------------------------------------
    # correct database for lunar effect
    # do, da = data_correction(dg, dr)

    #-----------------Study variation in seasonality-------------------
    # meanrad_vs_auc(dg, ds, id_arr, year=2013, only_lowrad_points=False)
    # boxplot_seasonality_variation(ds, id_arr, year=2013, only_lowrad_points=True)

    #---------finding points that require correction (SELECTION)-------------------
    #----FFT-----------
    # NL_fft(ds, id_arr, year=2013, column="RadE9_Mult_Nadir_Norm", only_lowrad_points=False)

    #----Autocorrelation-------
    # NL_autocorrelation(ds, id_arr, year=2013, column="RadE9_Mult_Nadir_Norm", only_lowrad_points=False)

    #----correlation between radiance and LI--------
    id_corr = correlation_NL_LI(dg, id_arr, year=2013, column="RadE9_Mult_Nadir_Norm", only_lowrad_points=False, use_pickled_file=True)

    #----------------(CORRECTION)--------------------------------------------------
    # use ds if you need to play with radiance as well as seasonality & residual combo
    # correction_of_selected_ids(dg, ds, id_arr, id_corr, year=2013, method="relative_radiance", metric="seasonal")
    # plots_for_studying_correction(dg, ds, id_arr, id_corr)
    # boxplots_for_studying_correction(dg, ds, id_arr, id_corr)

    #----------------BANDPASS FILTERING--------------------------------------------
    # db = run_bandpassfiltering(dg, ds, id_arr, id_corr, use_pickled=True)
    # combine bandpass and STL db (FYI both have different lengths)

    dbs = pd.read_hdf("STL_and_bandpass_yemen_updated.h5", key="zeal")
    # compare_bandpass_STL(dg, dbs, id_corr) #use it to plot scatter between mean radiance and wave amplitude
    # basic_plots_bandpass(dbs, column="RadE9_Mult_Nadir_Norm", id="sanaa_grid_10_69", year=2013)
    correcting_data_using_bandpass(dg, dbs)

    code.interact(local=locals())