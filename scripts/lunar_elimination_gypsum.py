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
import time

def STL_decomposition(df, column, year):
    df = df[(df.date_c.dt.year == year)]
    df = df.sort_values(by="date_c")
    df = df[["date_c", column]]
    df = df.resample("1D", on="date_c").mean()[[column]]
    df = df.interpolate(method="time")
    series = df[column]

    stl = STL(series, period=29, robust=True)
    res = stl.fit()

    print("Trend mean = {}".format(res.trend.mean()), flush=True)
    return res

def seasonal_component_for_all(df, out_path):
    print("##################################################################", flush=True)
    print("SEASONAL STL DECOMPOSITION", flush=True)
    res_comb = []
    for pt in df.id.unique():
        print("-----------------------", flush=True)
        print("ID - {}".format(pt), flush=True)
        for year in [2012,2013,2014,2015,2016,2017,2018,2019]:
            print("year - {}".format(year), flush=True)
            res = STL_decomposition(df[(df.id==pt)], column="RadE9_Mult_Nadir_Norm", year=year)
            res_db= pd.concat([res.observed,res.trend, res.seasonal, res.resid], axis=1).reset_index()
            res_db["id"] = pt
            res_db["year"] = year
            res_comb.append(res_db)
    res_comb = pd.concat(res_comb, axis=0)
    print("Dataframe has been concatenated", flush=True)
    res_comb.to_hdf(out_path + "STL_decomposition_output_2012_2019_allpoints.h5", key="zeal")
    print("STL HDF file has been saved.", flush=True)
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

def run_bandpassfiltering(dg, column, out_path, out_filename):
    """ Reference: https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    """
    print("##################################################################", flush=True)
    print("BANDPASS FILTERING", flush=True)

    fs = 1.0 #sample per day
    lowcut = 1/29.8
    highcut = 1/27.0

    temp_arr = []
    for cell_id in dg.id.unique():
        print("-----------------------", flush=True)
        print("ID - {}".format(cell_id), flush=True)
        for year in [2012,2013,2014,2015,2016,2017,2018,2019]:
            print("year - {}".format(year), flush=True)
            df = dg[(dg.date_c.dt.year == year) & (dg.id == cell_id)]
            df = df.sort_values(by="date_c")
            df = df.resample("1D", on="date_c").mean()[[column]]
            df = df.interpolate(method="time")
            df = df.reset_index()
            df["id"] = cell_id
            original_signal = df[column].values
            seasonality_signal = butter_bandpass_filter(original_signal, lowcut, highcut, fs, order=1)
            df["bandpass_output"] = seasonality_signal
            temp_arr.append(df[["id","date_c","bandpass_output"]])
            del(df)
    print("All years, all IDs DONE!", flush=True)
    db = pd.concat(temp_arr, axis=0)
    print("Dataframe has been concatenated", flush=True)
    db.to_hdf(out_path + "{}.h5".format(out_filename), key="zeal")
    print("Bandpass HDF file saved.", flush=True)
    return None

if __name__=='__main__':

    print("Start time: {}".format(datetime.datetime.now()), flush=True)

    data_path = "/home/zshah/yemen_files/data/"
    output_path = "/home/zshah/yemen_files/output/"
    # data_path = "./"
    # output_path = "./"

    #--------read & process necessary datasets----------
    dg = pd.read_hdf(data_path + "sanaa_unfiltered_nadirnormalized.h5", key="zeal")
    # # remove all the readings with negative radiance values
    # dg = dg[(dg.RadE9_Mult_Nadir_Norm>=0)]
    # clip negative radiance values to 0
    dg.RadE9_Mult_Nadir_Norm = dg.RadE9_Mult_Nadir_Norm.clip(lower=0)
    # combine multiple readings recorded on the same day
    dg = dg.groupby(["id","Latitude","Longitude","date_c","zero_lunar_illum"]).mean()[["RadE9_Mult_Nadir_Norm","LI"]].reset_index()

    # seasonal_component_for_all(dg, output_path)
    run_bandpassfiltering(dg, column="RadE9_Mult_Nadir_Norm", out_path=output_path, out_filename="bandpass_of_seasonality")

    print("End time: {}".format(datetime.datetime.now()), flush=True)
