"""
This script is used to produce all the plots and analysis for power supply analysis in Yemen.
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

def precrisis_comparisons(df, dgf):
    # Stable: 22_44
    # Unstable: 3_43
    # interesting: 17_49, 15_20
    df1 = df[(df.date_c < "2015-03-20")]
    df1 = df1[(df1.id == "sanaa_grid_3_43")]

    dgf1 = dgf[(dgf.date_c < "2015-03-20")]
    dgf1 = dgf1[(dgf1.id == "sanaa_grid_15_20")]

    code.interact(local=locals())
    return None

def pre_post_comparisons(df, cellid, max_range=55):
    pre = df[(df.date_c<="2015-03-20") & (df.id == cellid)]
    post = df[(df.date_c>="2015-04-15") & (df.date_c<="2018-04-15") & (df.id == cellid)]
    print("ID - {}".format(cellid))
    print("Precrisis - mean={}, kurtosis={} and skew={}".format(pre.rad_corr.mean(), pre.rad_corr.kurtosis(), pre.rad_corr.skew()))
    print("Postcrisis - mean={}, kurtosis={} and skew={}".format(post.rad_corr.mean(), post.rad_corr.kurtosis(), post.rad_corr.skew()))
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    if max_range == "auto":
        bins = None
    else:
        bins = range(0, max_range)
    ax1.hist(pre.rad_corr, bins=bins, label="Pre-crisis")
    ax1.set_ylabel("Count")
    ax1.set_title("Pre-crisis")
    ax2.hist(post.rad_corr, bins=bins, label="Post-crisis")
    ax2.set_ylabel("Count")
    ax2.set_xlabel("Radiance")
    ax2.set_title("Post-crisis")
    fig.suptitle("{}".format(cellid))
    plt.tight_layout()
    plt.show()
    return None

def overall_skew_mean(df):
    pre = df[(df.date_c<="2015-03-20")]
    post = df[(df.date_c>="2015-04-15")]
    # pre = pre.groupby("id").agg({"rad_corr":["mean","skew"]}).reset_index()
    # pre.columns = ["id","rad_mean","rad_skew"]
    pre = pre.groupby("date_c").sum()[["rad_corr"]].reset_index()
    # post=post.groupby("id").agg({"rad_corr":["mean","skew"]}).reset_index()
    # post.columns = ["id","rad_mean","rad_skew"]
    post = post.groupby("date_c").sum()[["rad_corr"]].reset_index()

    # fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    # ax1.scatter(pre.rad_mean, pre.rad_skew, s=4, alpha=0.5)
    # ax1.set_xlim(0,150)
    # ax1.set_ylim(0,15)
    # ax1.set_ylabel("Skew")
    # ax1.set_title("Pre-crisis")
    # ax2.scatter(post.rad_mean, post.rad_skew, s=4, color="red", alpha=0.5)
    # ax2.set_xlim(0,150)
    # ax2.set_ylim(0,15)
    # ax2.set_ylabel("Skew")
    # ax2.set_xlabel("Mean Radiance")
    # ax2.set_title("Post-crisis")
    # plt.show()
    sns.set(style="white", color_codes=True)
    sns.jointplot(x="rad_mean", y="rad_skew", data=pre, color="k", height=5, ratio=3)
    sns.jointplot(x="rad_mean", y="rad_skew", data=post, color="k", height=5, ratio=3)
    return None

def tnl_histograms(df):
    pre = df[(df.date_c<="2015-03-20")]
    post = df[(df.date_c>="2015-04-15")]
    pre = pre.groupby("date_c").sum()[["rad_corr"]].reset_index()
    post = post.groupby("date_c").sum()[["rad_corr"]].reset_index()
    print("Precrisis - mean={}, kurtosis={} and skew={}".format(pre.rad_corr.mean(), pre.rad_corr.kurtosis(), pre.rad_corr.skew()))
    print("Postcrisis - mean={}, kurtosis={} and skew={}".format(post.rad_corr.mean(), post.rad_corr.kurtosis(), post.rad_corr.skew()))
    # fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    # ax1.hist(pre.rad_corr, bins=50)
    # ax1.set_ylabel("Count")
    # ax1.set_title("Pre-crisis")
    # ax2.hist(post.rad_corr, bins=50)
    # ax2.set_ylabel("Count")
    # ax2.set_xlabel("Radiance")
    # ax2.set_title("Post-crisis")
    sns.set(style="white", color_codes=True)
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    ax1.hist(pre.rad_corr, bins=50, label="pre-crisis", alpha=0.8)
    ax1.hist(post.rad_corr, bins=50, label="post-crisis", alpha=0.8)
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Total Night Lights")
    plt.tight_layout()
    plt.legend()
    plt.show()
    return None

def grouped_tnl_histograms(df):
    di = pd.read_hdf("yemen_groups.h5",key="zeal")
    dk = pd.merge(df, di, left_on=["id"], right_on=["id"], how="left")
    pre = dk[(dk.date_c<="2015-03-20")]
    post = dk[(dk.date_c>="2015-04-15")]
    groupby_field = "prerad_group"

    fig, ax = plt.subplots()
    df_pre = pre[(pre[groupby_field] == "High")]
    df_post = post[(post[groupby_field] == "High")]
    ax.hist(df_pre.rad_corr, bins=6, label="pre-crisis", alpha=0.8)
    ax.hist(df_post.rad_corr, bins=6, label="post-crisis", alpha=0.8)

    # sns.set(style="white", color_codes=True)
    # fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2)
    # ax = [ax1,ax2,ax3,ax4]
    # a = 0
    # for grp in pre[groupby_field].unique():
    #     df_pre = pre[(pre[groupby_field] == grp)]
    #     df_post = post[(post[groupby_field] == grp)]
    #     ax[a].hist(df_pre.rad_corr, bins=None, label="pre-crisis", alpha=0.8)
    #     ax[a].hist(df_post.rad_corr, bins=None, label="post-crisis", alpha=0.8)
    #     ax[a].set_title(grp)
    #     ax[a].set_ylabel("Count")
    #     ax[a].set_xlabel("Total Night Lights")
    #     ax[a].legend()
    #     a = a+1
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
    dgf = dgf.rename(columns={"RadE9_Mult_Nadir_Norm":"rad_corr"})

    #---------read lunar corrected dataset-------------- (UNFILTERED)
    data_path = "./"
    dg = pd.read_hdf(data_path + "yemen_STL_lunar_corrected.h5", key="zeal")
    # dg = dg.drop(columns = ["RadE9_Mult_Nadir_Norm"])
    # dg = dg.rename(columns={"rad_corr":"RadE9_Mult_Nadir_Norm"})

    #---------K&S for single points---------------------- (PRECRISIS)
    # precrisis_comparisons(dg.copy(), dgf.copy())

    #---------K&S for single points---------------------- (PRE & POST CRISIS)
    # pre_post_comparisons(df.copy(), cellid=="sanaa_grid_17_49")

    #---------Hist of overall TNL------------------------ (PRE & POST)
    # tnl_histograms(dg.copy())

    #--------Hist of grouped TNL------------------------- (PRE & POST)
    grouped_tnl_histograms(dg.copy())
