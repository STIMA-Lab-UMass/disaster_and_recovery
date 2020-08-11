import pandas as pd
import geopandas as gpd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import code
import datetime
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from scipy.signal import butter, lfilter
from scipy.signal import freqz
import seaborn as sns

def check_which_id_needs_correction(df, output_path, data_path, use_pickled_file=True):
    if use_pickled_file == False:
        dc = pd.DataFrame(columns={"id", "year", "corr_coeff", "mean_rad"})

        for gridcell_id in df.id.unique():
            print("ID = {}".format(gridcell_id), flush=True)
            # [2012,2013,2014,2015,2016,2017,2018,2019]
            for year in [2012,2013,2014,2015,2016,2017,2018,2019]:
                print("year = {}".format(year), flush=True)
                df1 = df[(df.id == gridcell_id) & (df.date_c.dt.year == year)].sort_values(by="date_c")
                mean_rad = df1.mean()[["RadE9_Mult_Nadir_Norm"]].values[0]

                df1 = df1[["RadE9_Mult_Nadir_Norm","LI"]].corr()
                cross_corr_coeff = df1["LI"]["RadE9_Mult_Nadir_Norm"]

                dc = dc.append({"id":gridcell_id, "year":year, "corr_coeff":cross_corr_coeff, "mean_rad":mean_rad}, ignore_index=True)

                print("{} - {}".format(gridcell_id, cross_corr_coeff), flush=True)

            dc.to_hdf(output_path + "yemen_NL_LI_correlation_checkpoint.h5", key="zeal")
            print("Checkpoint saved. Length of DF = {}".format(len(dc)), flush=True)

        dc.to_hdf(output_path + "yemen_NL_LI_correlation_final.h5", key="zeal")
        print("Final correlation output file saved.")

    else:
        dc = pd.read_hdf(data_path + "yemen_NL_LI_correlation_final.h5", key="zeal")
    return dc

def NL_LI_correlation_meanrad_scatterplots(df):
    fig, ax = plt.subplots()
    i = 1
    for year in df.year.unique():
        df1 = df[(df.year == year)]
        plt.subplot(2,4,i)
        plt.scatter(df1.mean_rad, df1.corr_coeff, s=4)
        plt.xlim([0,300])
        plt.title("Year - {}".format(year))
        plt.xlabel("Mean Radiance")
        plt.ylabel("Correlation Coeff")
        i = i+1
    fig.suptitle("Correlation between NL and LI versus mean radiance (Sana'a, Yemen)")
    plt.show()
    return None

def NL_LI_correlation_meanrad_axialhistograms(df, out_path):
    sns.set(style="white", color_codes=True)

    for year in df.year.unique():
        df1 = df[(df.year == year)]
        sns.jointplot(x="mean_rad", y="corr_coeff", data=df1, color="k", height=5, ratio=3)
        plt.title("Year - {}".format(year))
        plt.savefig(out_path + "corr_meanrad_{}.pdf".format(year))
    # fig.suptitle("Correlation between NL and LI versus mean radiance (Sana'a, Yemen)")
    # plt.show()
    return None

def find_lowrad_points(df, ids_needing_correction, month, year):
    # areas with radiance during low moon times < 15 are qualified as low rad points (2012-2014)
    # Input: unfiltered dataframe
    # Output: array of low radiance points, mean radiance timeseries formed by taking mean of low rad timeseries
    dl = df[(df.zero_lunar_illum=="1") & (df.date_c.dt.year==year) & (df.date_c.dt.month.isin(month))].groupby(["id", pd.Grouper(key="date_c", freq="1M")]).mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()
    dl = dl[(dl.RadE9_Mult_Nadir_Norm <= 15)]
    ids_arr = dl.id.unique()
    final_ids_for_correction = list(set(ids_arr) & set(ids_needing_correction))
    # code.interact(local=locals())

    df1 = df[(df.id.isin(final_ids_for_correction)) & (df.date_c.dt.year==year) & (df.date_c.dt.month.isin(month))]
    dmean_rad = df1[((df1.zero_lunar_illum=="0"))].groupby("date_c").median()[["RadE9_Mult_Nadir_Norm"]].reset_index()
    # code.interact(local=locals())
    dmean_rad = dmean_rad.rename(columns={"RadE9_Mult_Nadir_Norm":"correction_val"})
    df_corr = pd.merge(df1, dmean_rad[["date_c","correction_val"]], left_on=["date_c"], right_on=["date_c"], how="left")
    df_corr["rad_corr"] = df_corr["RadE9_Mult_Nadir_Norm"] - df_corr["correction_val"]
    return df_corr

def lunar_corrected_database(df, dc):
    dk = pd.DataFrame()
    for year in [2012,2013,2014,2015,2016,2017,2018,2019]:
        print("Year={}".format(year))
        dc1 = dc[(dc.year == year)]
        ids_w_high_corr = dc1[(dc1.corr_coeff>=0.7)].id.unique()

        for month in [1,2,3,4,5,6,7,8,9,10,11,12]:
            print("Month={}".format(month))
            dcorr = find_lowrad_points(df, ids_w_high_corr, month=[month], year=year)
            dk = dk.append(dcorr)
    code.interact(local=locals())
    return None

def merge_original_and_corrected_databases(df, dc):
    df = pd.merge(df, dc[["id","date_c","correction_val","rad_corr"]], left_on=["id","date_c"], right_on=["id","date_c"], how="left")
    df["rad_corr2"] = np.where(~df["rad_corr"].isnull(), df["rad_corr"], df["RadE9_Mult_Nadir_Norm"])
    df["rad_corr2"] = df["rad_corr2"].clip(lower=0)

    grid_id = "sanaa_grid_10_69"
    year = 2013
    df1 = df[(df.id == grid_id) & (df.date_c.dt.year == year)]
    plt.plot(df1.date_c, df1.RadE9_Mult_Nadir_Norm, marker="*", c="blue", label="Raw radiance")
    plt.plot(df1.date_c, df1.rad_corr2, marker="^", c="red", label="Corrected radiance")
    plt.legend()
    plt.show()
    code.interact(local=locals())
    return df

def STL_based_correction(df, ds, dc):
    # Step 1: Merge original dataframe with seasonality dataframe
    # Step 1.1: Clip lowermost seasonality output to 0
    # NOTE: this merging step will remove all the points that were missing initially due to clouds
    # and were interpolated for seasonality calculations.
    df = pd.merge(df, ds[["id","date_c","season","resid"]], left_on=["id","date_c"], right_on=["id","date_c"], how="left")
    df["year"] = df.date_c.dt.year
    df["season"] = df["season"].clip(lower=0)
    print("Step 1 Done")

    # Step 2: Merge new dataframe with yearly correlation dataframe
    # Step 2.1: Create classes of points based on their mean_rad during a given year
    df = pd.merge(df, dc[["id","year","corr_coeff","mean_rad"]], left_on=["id","year"], right_on=["id","year"], how="left")
    df["point_class"] = df["mean_rad"].apply(lambda x: "lowrad" if x<=15 else "medrad" if 15<x<=50 else "highrad")
    print("Step 2 Done")


    # Step 3: Correct radiance for only the points with correlation value > 0.7 on high lunar days
    # Step 3.1: Clip lowermost radiance values to 0
    df["rad_corr"] = np.where((df["zero_lunar_illum"] == "0") & (df["corr_coeff"]>=0.7), df["RadE9_Mult_Nadir_Norm"] - df["season"], df["RadE9_Mult_Nadir_Norm"])
    print("Step 3 Done")

    # Step 4: Resample the dataframe on daily basis. This will introduce all the timestamps missed due to cloud filtering
    # These timestamps will have rad_corr value of Nan
    # Step 4.1: Replace all the negative rad_corr values with Nan
    df = df.groupby(["id","Latitude","Longitude"]).resample("1D", on="date_c").mean()[["RadE9_Mult_Nadir_Norm","rad_corr","season","LI"]].reset_index()
    df["rad_corr"] = df["rad_corr"].apply(lambda x: np.nan if x<0 else x)
    print("Step 4 Done")

    # Step 5: Interpolate missing values for every cell
    df.index = df.date_c  #because interpolation step needs index to be datetime
    df = df.drop(columns = ["date_c"])
    df = df.groupby(["id","Latitude","Longitude"]).apply(lambda x: x.interpolate(method="time")).reset_index() #takes sometime to get computed
    print("Step 5 Done")

    try:
        df.to_hdf("./yemen_STL_lunar_corrected.h5", key="zeal")
        print("File saved")
        code.interact(local=locals())
    except:
        code.interact(local=locals())

    # Check how many IDs would need correction in a given year and what class do they lie in
    # Observation: All the IDs belong to low radiance category
    # Also, the number of IDs needing correction increase over years and again all IDs belonged to low rad category
    # dc1 = dc[(dc.corr_coeff>=0.7)].groupby("year").point_class.value_counts()

    # grid_id = "sanaa_grid_20_22"
    # year = 2016
    # df1 = df[(df.id == grid_id) & (df.date_c.dt.year == year)]
    # plt.plot(df1.date_c, df1.RadE9_Mult_Nadir_Norm, marker="*", c="blue", label="Raw radiance")
    # plt.plot(df1.date_c, df1.rad_corr, marker="^", c="red", label="Corrected radiance")
    # plt.plot(df1.date_c, df1.LI, marker="h", c="green", label="LI")
    # plt.legend()
    # plt.show()

    # code.interact(local=locals())
    return None

if __name__=='__main__':
    data_path = "./"
    output_path = "./"

    # data_path = "/home/zshah/yemen_files/data/"
    # output_path = "/home/zshah/yemen_files/outputs/"

    #--------read & process necessary datasets----------
    dg = pd.read_hdf("sanaa_unfiltered_nadirnormalized.h5", key="zeal")
    # clip all the readings with negative radiance values to 0
    dg.RadE9_Mult_Nadir_Norm = dg.RadE9_Mult_Nadir_Norm.clip(lower=0)
    # combine multiple readings recorded on the same day
    dg = dg.groupby(["id","Latitude","Longitude","date_c","zero_lunar_illum"]).mean()[["RadE9_Mult_Nadir_Norm","LI"]].reset_index()

    #--------Find which ids need correction for a given year-------------------------------
    dc = check_which_id_needs_correction(dg, output_path=output_path, data_path=data_path)
    # NL_LI_correlation_meanrad_scatterplots(dc)
    # NL_LI_correlation_meanrad_axialhistograms(dc, out_path = output_path+"outputs/")
    # code.interact(local=locals())

    #--------Lunar correction using mean radiance------------------------------------------
    # lunar_corrected_database(dg, dc)
    # dcorr = pd.read_hdf("yemen_lunar_corrected_attempt1.h5", key="zeal") #mean
    # dcorr = pd.read_hdf("yemen_lunar_corrected_attempt2.h5", key="zeal") #median
    # merge_original_and_corrected_databases(dg, dcorr)

    #-----------STL based lunar correction-------------------------------------------------
    # Uncomment following two statements to run the correction process. (It takes a while to complete)
    # ds = pd.read_hdf("STL_decomposition_output_2012_2019_allpoints_updated.h5", key="zeal")
    # STL_based_correction(dg, ds, dc)

    # read the pickled corrected database
    df = pd.read_hdf(data_path + "yemen_STL_lunar_corrected.h5", key="zeal")

    code.interact(local=locals())
