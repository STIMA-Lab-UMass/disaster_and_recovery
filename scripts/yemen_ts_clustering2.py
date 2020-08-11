"""
This script computes cross-correlation between radiance timeseries corresponding to each cell in Yemen.
@zeal

# References for Heirarchical clustering of correlation matrix:
https://stackoverflow.com/questions/52787431/create-clusters-using-correlation-matrix-in-python
https://github.com/TheLoneNut/CorrelationMatrixClustering/blob/master/CorrelationMatrixClustering.ipynb
https://stackoverflow.com/questions/37579759/agglomerativeclustering-on-a-correlation-matrix
https://stackoverflow.com/questions/38070478/how-to-do-clustering-using-the-matrix-of-correlation-coefficients
https://stats.stackexchange.com/questions/275720/does-any-other-clustering-algorithms-take-correlation-as-distance-metric-apart
https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.cluster.hierarchy.fcluster.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

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
from yemen_plotting_utils import *
from sklearn.neighbors import DistanceMetric
import scipy.cluster.hierarchy as spc
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import centroid

def correlation_and_clustering_relative_to_baseline(df, use_pickled=True, fixed_distance_clustering=False):
    drad = df[["id","Latitude","Longitude"]].drop_duplicates()

    #---------------get correlation values for pre-crisis levels------------------------
    """
    Pre - period before March 1, 2015
    Post - whole of 2016
    Present - whole of 2018
    """
    states = ["pre","post","present"]
    if use_pickled == False:
        for s in states:
            print("State = {}".format(s))
            if s == "pre":
                df1 = df[(df.date_c <= "2015-03-01")]
            elif s == "post":
                df1 = df[(df.date_c.dt.year == 2016)]
            elif s == "present":
                df1 = df[(df.date_c.dt.year == 2018)]

            df1 = df1.pivot(index="date_c", columns="id", values="RadE9_Mult_Nadir_Norm")
            df1 = df1.corr()
            df1.to_hdf("yemen_ts_correlation_{}.h5".format(s), key="zeal")
            print("Done")
            del(df1)
        print("all correlation files saved")

    #-----------run clustering for every year-------------------------------------------
    for s in states:
        print("State = {}".format(s))
        if s == "post":
            s = 2016
        elif s == "present":
            s = 2018
        df1 = pd.read_hdf("yemen_ts_correlation_{}.h5".format(s), key="zeal")
        corr = df1.values
        corr = 1 - np.abs(corr)
        pdist = spc.distance.pdist(corr) #length = 3285 choose 2 = 5393970
        linkage = spc.linkage(pdist, method='complete')
        if fixed_distance_clustering == True:
            idx = spc.fcluster(linkage, 16.228307171761188, 'distance')
        else:
            idx = spc.fcluster(linkage, 0.5*pdist.max(), 'distance')
        drad["labels_{}".format(s)] = idx
        del(df1)

    drad_gdf = create_geodataframe(drad, radius=462, cap_style=3, buffered=True)

    #---------Visualize for all years---------------------------------------------------
    fig, (ax_pre,ax_post,ax_present) = plt.subplots(nrows=1, ncols=3)
    ax = [ax_pre,ax_post,ax_present]
    i = 0
    for s in states:
        if s == "post":
            s = 2016
        elif s == "present":
            s = 2018
        no_of_clusters = drad_gdf["labels_{}".format(s)].nunique()
        plot_geospatial_heatmap_with_event_locs(geo_df=drad_gdf, col_name="labels_{}".format(s), events_data=None, title="State-{} (clusters={})".format(s,no_of_clusters), cmap=cm.tab10, cmap_type="tab10", needs_colormapping=False, marker_color="black", events_data_type="locations_buffered", add_title=True, event_locs_included=False, include_colorbar=True, with_streetmap=True, ax=ax[i])
        i = i+1
        plt.show()
    code.interact(local=locals())
    return None

def study_changes_in_clusters():
    dc = pd.read_hdf("final_clustering_labels.h5",key="zeal")
    # just for the sake of indexing starting at 0 instead of 1 since all labels start with 1.
    dc["labels_pre"] = dc["labels_pre"] - 1
    dc["labels_2016"] = dc["labels_2016"] - 1
    dc["labels_2018"] = dc["labels_2018"] - 1

    #-------get all ids associated with labels-----------------------------------
    pre = {}
    for label in dc.labels_pre.unique():
        pre[label] = dc[(dc.labels_pre == label)].id.unique()

    post = {}
    for label in dc.labels_2016.unique():
        post[label] = dc[(dc.labels_2016 == label)].id.unique()

    present = {}
    for label in dc.labels_2018.unique():
        present[label] = dc[(dc.labels_2018 == label)].id.unique()

    #--------compare pre and post labels----------------------------------------
    pre_post = {} #dict with tupe of pre and post labels as keys and length of intersecting IDs as value
    c1_abs = np.zeros((len(pre), len(post)))
    c1_perc = np.zeros((len(pre), len(post)))
    for pre_key in pre.keys():
        for post_key in post.keys():
            pre_set = set(pre[pre_key])
            post_set = set(post[post_key])
            c1_abs[pre_key, post_key] = len(pre_set.intersection(post_set))
            c1_perc[pre_key, post_key] = len(pre_set.intersection(post_set))*100.0/len(pre_set)
            c1_abs = np.round(c1_abs, 1)
            c1_perc = np.round(c1_perc, 1)

    #--------compare pre and present labels----------------------------------------
    pre_present = {} #dict with tupe of pre and present labels as keys and length of intersecting IDs as value
    c2_abs = np.zeros((len(pre), len(present)))
    c2_perc = np.zeros((len(pre), len(present)))
    for pre_key in pre.keys():
        for present_key in present.keys():
            pre_set = set(pre[pre_key])
            present_set = set(present[present_key])
            c2_abs[pre_key, present_key] = len(pre_set.intersection(present_set))
            c2_perc[pre_key, present_key] = len(pre_set.intersection(present_set))*100.0/len(present_set)
            c2_abs = np.round(c2_abs, 1)
            c2_perc = np.round(c2_perc, 1)

    #--------compare present and post labels----------------------------------------
    post_present = {} #dict with tupe of pre and present labels as keys and length of intersecting IDs as value
    c3_abs = np.zeros((len(post), len(present)))
    c3_perc = np.zeros((len(post), len(present)))
    for post_key in post.keys():
        for present_key in present.keys():
            post_set = set(post[post_key])
            present_set = set(present[present_key])
            c3_abs[post_key, present_key] = len(post_set.intersection(present_set))
            c3_perc[post_key, present_key] = len(post_set.intersection(present_set))*100.0/len(present_set)
            c3_abs = np.round(c3_abs, 1)
            c3_perc = np.round(c3_perc, 1)

    #-----------assign same labels to similar pre-post groups------------------- (MANUAL)
    dm = dc.copy()
    dm["labels_2016"] = dm["labels_2016"].apply(lambda x: 0 if x==2 else 2 if x==1 else 1)
    dm["labels_2018"] = dm["labels_2018"].apply(lambda x: 2 if x==0 else 0 if x==4 else 1 if x==2 else 4 if x==1 else 3)
    dm = create_geodataframe(dm, radius=462, cap_style=3, buffered=True)

    #--------combining builtup area details-------------------------------------
    extra_data_path = "../extra_datasets/"
    di = pd.read_pickle(extra_data_path + "yemen_infra_pop_data_combined_2.pck")
    dm = pd.merge(dm, di[["id","builtup_area","total_pop"]].drop_duplicates(), left_on=["id"], right_on=["id"], how="left")
    dm_temp = dm.copy()
    #-----------Classifying into 3 groups for simplicity------------------------
    dm["labels_pre"] = dm["labels_pre"].apply(lambda x: "urban" if x==0 else "rural" if x==2 else "periurban")
    dm["labels_2016"] = dm["labels_2016"].apply(lambda x: "urban" if x==0 else "rural" if x==2 else "periurban")
    dm["labels_2018"] = dm["labels_2018"].apply(lambda x: "urban" if x==0 else "rural" if x==2 else "periurban")

    #------------multiple levels of peri-urban (alternate option)---------------
    # da = dm_temp.copy()
    # da["labels_pre"] = da["labels_pre"].apply(lambda x: "urban" if x==0 else "rural" if x==2 else "periurban_2" if x==4 else "periurban_1")
    # da["labels_2016"] = da["labels_2016"].apply(lambda x: "urban" if x==0 else "rural" if x==2 else "periurban_2" if x==4 else "periurban_1")
    # da["labels_2018"] = da["labels_2018"].apply(lambda x: "urban" if x==0 else "rural" if x==2 else "periurban_2" if x==4 else "periurban_1")

    #---------Visualize for all years---------------------------------------------------
    # states = ["pre","post","present"]
    # fig, (ax_pre,ax_post,ax_present) = plt.subplots(nrows=1, ncols=3)
    # ax = [ax_pre,ax_post,ax_present]
    # titles = ["Apr 2012 to Mar 2015", "Year 2016", "Year 2018"]
    # i = 0
    # for s in states:
    #     if s == "post":
    #         s = 2016
    #     elif s == "present":
    #         s = 2018
    #     plot_geospatial_heatmap_with_event_locs(geo_df=dm, col_name="labels_{}".format(s), events_data=None, title=titles[i], cmap=cm.Set1, cmap_type="Set1", max_stretch=4, needs_colormapping=True, marker_color="black", events_data_type="locations_buffered", add_title=True, event_locs_included=False, include_colorbar=True, with_streetmap=True, ax=ax[i])
    #     # testing labels for first plot
    #     # dm1 = dm[(dm.labels_pre == 3)]
    #     # plot_geospatial_heatmap_with_event_locs(geo_df=dm1, col_name="labels_{}".format(s), events_data=None, title="zeal", cmap=cm.Set1, cmap_type="Set1", max_stretch=4, needs_colormapping=False, marker_color="black", events_data_type="locations_buffered", add_title=False, event_locs_included=False, include_colorbar=False, with_streetmap=True, ax=ax[i])
    #     i = i+1
    # plt.show()
    # code.interact(local=locals())
    return dm

def study_changes_in_clusters_simple():
    dc = pd.read_hdf("final_clustering_labels.h5",key="zeal")
    # just for the sake of indexing starting at 0 instead of 1 since all labels start with 1.
    dc["labels_pre"] = dc["labels_pre"] - 1
    dc["labels_2016"] = dc["labels_2016"] - 1
    dc["labels_2018"] = dc["labels_2018"] - 1

    #-------get all ids associated with labels-----------------------------------
    pre = {}
    for label in dc.labels_pre.unique():
        pre[label] = dc[(dc.labels_pre == label)].id.unique()

    post = {}
    for label in dc.labels_2016.unique():
        post[label] = dc[(dc.labels_2016 == label)].id.unique()

    present = {}
    for label in dc.labels_2018.unique():
        present[label] = dc[(dc.labels_2018 == label)].id.unique()

    #--------compare pre and post labels----------------------------------------
    pre_post = {} #dict with tupe of pre and post labels as keys and length of intersecting IDs as value
    c1_abs = np.zeros((len(pre), len(post)))
    c1_perc = np.zeros((len(pre), len(post)))
    for pre_key in pre.keys():
        for post_key in post.keys():
            pre_set = set(pre[pre_key])
            post_set = set(post[post_key])
            c1_abs[pre_key, post_key] = len(pre_set.intersection(post_set))
            c1_perc[pre_key, post_key] = len(pre_set.intersection(post_set))*100.0/len(pre_set)
            c1_abs = np.round(c1_abs, 1)
            c1_perc = np.round(c1_perc, 1)

    #--------compare pre and present labels----------------------------------------
    pre_present = {} #dict with tupe of pre and present labels as keys and length of intersecting IDs as value
    c2_abs = np.zeros((len(pre), len(present)))
    c2_perc = np.zeros((len(pre), len(present)))
    for pre_key in pre.keys():
        for present_key in present.keys():
            pre_set = set(pre[pre_key])
            present_set = set(present[present_key])
            c2_abs[pre_key, present_key] = len(pre_set.intersection(present_set))
            c2_perc[pre_key, present_key] = len(pre_set.intersection(present_set))*100.0/len(present_set)
            c2_abs = np.round(c2_abs, 1)
            c2_perc = np.round(c2_perc, 1)

    #--------compare present and post labels----------------------------------------
    post_present = {} #dict with tupe of pre and present labels as keys and length of intersecting IDs as value
    c3_abs = np.zeros((len(post), len(present)))
    c3_perc = np.zeros((len(post), len(present)))
    for post_key in post.keys():
        for present_key in present.keys():
            post_set = set(post[post_key])
            present_set = set(present[present_key])
            c3_abs[post_key, present_key] = len(post_set.intersection(present_set))
            c3_perc[post_key, present_key] = len(post_set.intersection(present_set))*100.0/len(present_set)
            c3_abs = np.round(c3_abs, 1)
            c3_perc = np.round(c3_perc, 1)

    #-----------assign same labels to similar pre-post groups------------------- (MANUAL)
    dm = dc.copy()
    dm["labels_pre"] = dm["labels_pre"].apply(lambda x: x if (x==0) or (x==2) else 1)
    dm["labels_2016"] = dm["labels_2016"].apply(lambda x: 0 if x==2 else 2 if x==1 else 1)
    dm["labels_2018"] = dm["labels_2018"].apply(lambda x: 2 if x==0 else 0 if x==4 else 1)
    dm = create_geodataframe(dm, radius=462, cap_style=3, buffered=True)

    #--------combining builtup area details-------------------------------------
    extra_data_path = "../extra_datasets/"
    di = pd.read_pickle(extra_data_path + "yemen_infra_pop_data_combined_2.pck")
    dm = pd.merge(dm, di[["id","builtup_area","total_pop"]].drop_duplicates(), left_on=["id"], right_on=["id"], how="left")
    dm_temp = dm.copy()
    #-----------Classifying into 3 groups for simplicity------------------------
    dm["labels_pre"] = dm["labels_pre"].apply(lambda x: "urban" if x==0 else "rural" if x==2 else "periurban")
    dm["labels_2016"] = dm["labels_2016"].apply(lambda x: "urban" if x==0 else "rural" if x==2 else "periurban")
    dm["labels_2018"] = dm["labels_2018"].apply(lambda x: "urban" if x==0 else "rural" if x==2 else "periurban")

    #------------multiple levels of peri-urban (alternate option)---------------
    # da = dm_temp.copy()
    # da["labels_pre"] = da["labels_pre"].apply(lambda x: "urban" if x==0 else "rural" if x==2 else "periurban_2" if x==4 else "periurban_1")
    # da["labels_2016"] = da["labels_2016"].apply(lambda x: "urban" if x==0 else "rural" if x==2 else "periurban_2" if x==4 else "periurban_1")
    # da["labels_2018"] = da["labels_2018"].apply(lambda x: "urban" if x==0 else "rural" if x==2 else "periurban_2" if x==4 else "periurban_1")

    #---------Visualize for all years---------------------------------------------------
    states = ["pre","post","present"]
    fig, (ax_pre,ax_post,ax_present) = plt.subplots(nrows=1, ncols=3)
    ax = [ax_pre,ax_post,ax_present]
    titles = ["Apr 2012 to Mar 2015", "Year 2016", "Year 2018"]
    i = 0
    for s in states:
        if s == "post":
            s = 2016
        elif s == "present":
            s = 2018
        plot_geospatial_heatmap_with_event_locs(geo_df=dm, col_name="labels_{}".format(s), events_data=None, title=titles[i], cmap=cm.Set1, cmap_type="Set1", max_stretch=2, needs_colormapping=False, marker_color="black", events_data_type="locations_buffered", add_title=True, event_locs_included=False, include_colorbar=False, with_streetmap=True, ax=ax[i])
        # testing labels for first plot
        # dm1 = dm[(dm.labels_pre == 3)]
        # plot_geospatial_heatmap_with_event_locs(geo_df=dm1, col_name="labels_{}".format(s), events_data=None, title="zeal", cmap=cm.Set1, cmap_type="Set1", max_stretch=4, needs_colormapping=False, marker_color="black", events_data_type="locations_buffered", add_title=False, event_locs_included=False, include_colorbar=False, with_streetmap=True, ax=ax[i])
        i = i+1
    plt.show()
    code.interact(local=locals())
    return dm



def calculate_TNL_changes_in_areas(dg, dm):
    df = pd.merge(dg, dm[["id","labels_pre"]].drop_duplicates(), left_on=["id"], right_on=["id"], how="left")

    pre = df[(df.date_c<"2015-03-01")].groupby(["id","labels_pre"]).mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()
    pre = pre.groupby(["labels_pre"]).sum()[["RadE9_Mult_Nadir_Norm"]].reset_index()

    post = df[(df.date_c.dt.year==2016)].groupby(["id","labels_pre"]).mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()
    post = post.groupby(["labels_pre"]).sum()[["RadE9_Mult_Nadir_Norm"]].reset_index()

    present = df[(df.date_c.dt.year==2018)].groupby(["id","labels_pre"]).mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()
    present = present.groupby(["labels_pre"]).sum()[["RadE9_Mult_Nadir_Norm"]].reset_index()

    code.interact(local=locals())
    return None


if __name__=='__main__':
    data_path = "./"
    output_path = "./"
    #---------read lunar corrected dataset--------------
    data_path = "./"
    dg = pd.read_hdf(data_path + "yemen_STL_lunar_corrected.h5", key="zeal")
    dg = dg.drop(columns = ["RadE9_Mult_Nadir_Norm"])
    dg = dg.rename(columns={"rad_corr":"RadE9_Mult_Nadir_Norm"})

    #---------calculate correlation for every year-------
    # correlation_and_clustering_relative_to_baseline(dg.copy(), use_pickled=True, fixed_distance_clustering=False)

    #--------comparing clusters--------------------------
    # dc = study_changes_in_clusters()
    dc = study_changes_in_clusters_simple()

    #--------calculating proportions---------------------
    # calculate_TNL_changes_in_areas(dg.copy(), dc.copy())