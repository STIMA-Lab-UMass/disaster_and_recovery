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

def precrisis_correlation(df, use_pickled=True, visualize_output=False):
    #---------------Classification of cells based on pre-crisis radiance levels---------
    drad = df[(df.date_c < "2015-03-01")].groupby(["id","Latitude","Longitude"]).mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()
    drad["rad_group"] = drad["RadE9_Mult_Nadir_Norm"].apply(lambda x: "low" if x<=15 else "medium" if 15<x<=50 else "high")
    drad = drad[["id","Latitude","Longitude","rad_group"]].drop_duplicates()

    #---------------get correlation values for pre-crisis levels------------------------
    if use_pickled == False:
        df = df[(df.date_c.dt.year<=2014)]
        df1 = df.pivot(index="date_c", columns="id", values="RadE9_Mult_Nadir_Norm")
        df1 = df1.corr()
    else:
        df1 = pd.read_hdf("precrisis_ts_correlation.h5",key="zeal")

    # get 10 IDs with highest correlation values to a given ID
    # (Reference: https://stackoverflow.com/questions/35871907/pandas-idxmax-best-n-results)
    dm = df1.apply(lambda s: s.abs().nlargest(100).index.tolist(), axis=1)

    #-----get distance matrix-------------------------------------
    dd = create_distance_matrix(df)

    #----------DBSCAN for clustering------------------------------
    # X = df1.values
    # X = 1 - np.abs(X)
    # clustering = DBSCAN(eps=0.2, min_samples=6, metric="precomputed").fit_predict(X)
    # drad["labels"] = clustering

    #----------Agglomerative Clustering---------------------------
    # X = df1.values
    # X = 1 - np.abs(X)
    # clustering = AgglomerativeClustering(n_clusters=5, affinity="precomputed", linkage="complete").fit(X)
    # drad["labels"] = clustering.labels_

    #------Correlation based heirarchical clustering--------------
    corr = df1.values
    corr = 1-np.abs(corr)
    pdist = spc.distance.pdist(corr) #length = 3285 choose 2 = 5393970
    linkage = spc.linkage(pdist, method='complete')
    idx = spc.fcluster(linkage, 0.5*pdist.max(), 'distance') #0.5*pdist.max()
    drad["labels"] = idx #should be fine since order of IDs in drad and df1 is exactly the same.

    #--------Visualize spatially----------------------------------
    drad_gdf = create_geodataframe(drad, radius=462, cap_style=3, buffered=True)
    # code.interact(local=locals())
    if visualize_output:
        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2)
        plot_geospatial_heatmap_with_event_locs(geo_df=drad_gdf, col_name="labels", events_data=None, title=None, cmap=cm.tab10, cmap_type="tab10", needs_colormapping=False, marker_color="black", events_data_type="locations_buffered", add_title=False, event_locs_included=False, include_colorbar=True, ax=ax1)
        plot_geospatial_heatmap_with_event_locs(geo_df=drad_gdf, col_name="rad_group", events_data=None, title=None, cmap=cm.Set1, cmap_type="Set1", needs_colormapping=False, marker_color="black", events_data_type="locations_buffered", add_title=False, event_locs_included=False, include_colorbar=True, ax=ax2)
        plt.show()

    #------Visualize for both - distance and correlation ---------
    # cell_id = "sanaa_grid_20_26"
    # df_dist = dd[cell_id]
    # df_corr = df1[cell_id]
    # dc = pd.concat([df_dist, df_corr], axis=1)
    # dc = dc.reset_index()
    # dc.columns = ["id","distance","corr_coeff"]
    # dc = pd.merge(dc, drad, left_on=["id"], right_on=["id"], how="left")
    # dc["rad_group"] = dc["rad_group"].apply(lambda x: "red" if x=="low" else "blue" if x=="medium" else "green")
    # # dc.distance = round(dc.distance, 1)
    # # dc = dc.groupby("distance").mean()[["corr_coeff"]].reset_index()
    # dc = dc.sort_values(by=["distance"])
    # plt.scatter(dc.distance, dc.corr_coeff, s=4, c=dc.rad_group)
    # plt.xlabel("Distance from the point")
    # plt.ylabel("Correlation Coefficient")
    # plt.title("Cell ID - {}".format(cell_id))
    # plt.show()

    # code.interact(local=locals())
    return drad_gdf

def postcrisis_correlation(df, use_pickled=True, visualize_output=False):
    #---------------Classification of cells based on pre-crisis radiance levels---------
    drad = df[(df.date_c < "2015-03-01")].groupby(["id","Latitude","Longitude"]).mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()
    drad["rad_group"] = drad["RadE9_Mult_Nadir_Norm"].apply(lambda x: "low" if x<=15 else "medium" if 15<x<=50 else "high")
    drad = drad[["id","Latitude","Longitude","rad_group"]].drop_duplicates()

    #---------------get correlation values for post-crisis levels------------------------
    if use_pickled == False:
        df = df[(df.date_c >= "2015-03-01" )]
        df1 = df.pivot(index="date_c", columns="id", values="RadE9_Mult_Nadir_Norm")
        df1 = df1.corr()
        df1.to_hdf("postcrisis_ts_correlation.h5", key="zeal")
        print("HDF for postcrisis correlation saved.")
    else:
        df1 = pd.read_hdf("postcrisis_ts_correlation.h5",key="zeal")

    # get 10 IDs with highest correlation values to a given ID
    # (Reference: https://stackoverflow.com/questions/35871907/pandas-idxmax-best-n-results)
    dm = df1.apply(lambda s: s.abs().nlargest(100).index.tolist(), axis=1)

    #-----get distance matrix-------------------------------------
    # dd = create_distance_matrix(df)

    #----------DBSCAN for clustering------------------------------
    # X = df1.values
    # X = 1 - np.abs(X)
    # clustering = DBSCAN(eps=0.2, min_samples=6, metric="precomputed").fit_predict(X)
    # drad["labels"] = clustering
    #----------Agglomerative Clustering---------------------------
    # X = df1.values
    # X = 1 - np.abs(X)
    # clustering = AgglomerativeClustering(n_clusters=5, affinity="precomputed", linkage="complete").fit(X)
    # drad["labels"] = clustering.labels_

    #------Correlation based heirarchical clustering--------------
    corr = df1.values
    corr = 1-np.abs(corr)
    pdist = spc.distance.pdist(corr) #length = 3285 choose 2 = 5393970
    linkage = spc.linkage(pdist, method='complete')
    idx = spc.fcluster(linkage, 0.5*pdist.max(), 'distance') #0.5*pdist.max()
    drad["labels"] = idx

    #--------Visualize spatially----------------------------------
    drad_gdf = create_geodataframe(drad, radius=462, cap_style=3, buffered=True)

    if visualize_output:
        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2)
        plot_geospatial_heatmap_with_event_locs(geo_df=drad_gdf, col_name="labels", events_data=None, title=None, cmap=cm.tab10, cmap_type="tab10", needs_colormapping=False, marker_color="black", events_data_type="locations_buffered", add_title=False, event_locs_included=False, include_colorbar=True, ax=ax1)
        plot_geospatial_heatmap_with_event_locs(geo_df=drad_gdf, col_name="rad_group", events_data=None, title=None, cmap=cm.Set1, cmap_type="Set1", needs_colormapping=False, marker_color="black", events_data_type="locations_buffered", add_title=False, event_locs_included=False, include_colorbar=True, ax=ax2)
        plt.show()

    #------Visualize for both - distance and correlation ---------
    # cell_id = "sanaa_grid_20_26"
    # df_dist = dd[cell_id]
    # df_corr = df1[cell_id]
    # dc = pd.concat([df_dist, df_corr], axis=1)
    # dc = dc.reset_index()
    # dc.columns = ["id","distance","corr_coeff"]
    # dc = pd.merge(dc, drad, left_on=["id"], right_on=["id"], how="left")
    # dc["rad_group"] = dc["rad_group"].apply(lambda x: "red" if x=="low" else "blue" if x=="medium" else "green")
    # # dc.distance = round(dc.distance, 1)
    # # dc = dc.groupby("distance").mean()[["corr_coeff"]].reset_index()
    # dc = dc.sort_values(by=["distance"])
    # plt.scatter(dc.distance, dc.corr_coeff, s=4, c=dc.rad_group)
    # plt.xlabel("Distance from the point")
    # plt.ylabel("Correlation Coefficient")
    # plt.title("Cell ID - {}".format(cell_id))
    # plt.show()

    # code.interact(local=locals())
    return drad_gdf

def correlation_and_clustering_per_year(df, use_pickled=True, fixed_distance_clustering=False):
    #---------------Classification of cells based on pre-crisis radiance levels---------
    drad = df[(df.date_c < "2015-03-01")].groupby(["id","Latitude","Longitude"]).mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()
    drad["rad_group"] = drad["RadE9_Mult_Nadir_Norm"].apply(lambda x: "low" if x<=15 else "medium" if 15<x<=50 else "high")
    drad = drad[["id","Latitude","Longitude","rad_group"]].drop_duplicates()

    #---------------get correlation values for pre-crisis levels------------------------
    if use_pickled == False:
        for year in [2012,2013,2014,2015,2016,2017,2018,2019]:
            print("Year = {}".format(year))
            df1 = df[(df.date_c.dt.year == year)]
            df1 = df1.pivot(index="date_c", columns="id", values="RadE9_Mult_Nadir_Norm")
            df1 = df1.corr()
            df1.to_hdf("yemen_ts_correlation_{}.h5".format(year), key="zeal")
            print("Done")
        print("all correlation files saved")

    #-----------run clustering for every year-------------------------------------------
    for year in [2012,2013,2014,2015,2016,2017,2018,2019]:
        print("Year = {}".format(year))
        df1 = pd.read_hdf("yemen_ts_correlation_{}.h5".format(year), key="zeal")
        corr = df1.values
        corr = 1 - np.abs(corr)
        pdist = spc.distance.pdist(corr) #length = 3285 choose 2 = 5393970
        linkage = spc.linkage(pdist, method='complete')
        if fixed_distance_clustering == True:
            idx = spc.fcluster(linkage, 16.228307171761188, 'distance')
        else:
            idx = spc.fcluster(linkage, 0.5*pdist.max(), 'distance')
        drad["labels_{}".format(year)] = idx
    drad_gdf = create_geodataframe(drad, radius=462, cap_style=3, buffered=True)

    #-----------run agglomerative clustering for every year------------------------------
    # for year in [2012,2013,2014,2015,2016,2017,2018,2019]:
    #     print("Year = {}".format(year))
    #     df1 = pd.read_hdf("yemen_ts_correlation_{}.h5".format(year), key="zeal")
    #     X = df1.values
    #     X = 1 - np.abs(X)
    #     clustering = AgglomerativeClustering(n_clusters=5, affinity="precomputed", linkage="complete").fit(X)
    #     drad["labels_{}".format(year)] = clustering.labels_

    #---------Visualize for all years---------------------------------------------------
    fig, (ax2012,ax2013,ax2014,ax2015,ax2016,ax2017,ax2018,ax2019) = plt.subplots(nrows=1, ncols=8)
    ax = [ax2012,ax2013,ax2014,ax2015,ax2016,ax2017,ax2018,ax2019]
    i = 0
    for year in [2012,2013,2014,2015,2016,2017,2018,2019]:
        no_of_clusters = drad_gdf["labels_{}".format(year)].nunique()
        plot_geospatial_heatmap_with_event_locs(geo_df=drad_gdf, col_name="labels_{}".format(year), events_data=None, title="Year-{} (clusters={})".format(year,no_of_clusters), cmap=cm.tab10, cmap_type="tab10", needs_colormapping=False, marker_color="black", events_data_type="locations_buffered", add_title=True, event_locs_included=False, include_colorbar=False, ax=ax[i])
        i = i+1
    plt.show()
    code.interact(local=locals())
    return drad_gdf

def visualize_pre_post_clustering(df):
    dpre = precrisis_correlation(df.copy(), use_pickled=True)
    dpost = postcrisis_correlation(df.copy(), use_pickled=True)

    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2)
    plot_geospatial_heatmap_with_event_locs(geo_df=dpre, col_name="labels", events_data=None, title="Pre-crisis ({} clusters)".format(dpre.labels.nunique()), cmap=cm.tab10, cmap_type="tab10", max_stretch=5, needs_colormapping=False, marker_color="black", events_data_type="locations_buffered", add_title=True, event_locs_included=False, include_colorbar=True, ax=ax1)
    plot_geospatial_heatmap_with_event_locs(geo_df=dpost, col_name="labels", events_data=None, title="Post-crisis ({} clusters)".format(dpost.labels.nunique()), cmap=cm.tab10, cmap_type="tab10", max_stretch=5, needs_colormapping=False, marker_color="black", events_data_type="locations_buffered", add_title=True, event_locs_included=False, include_colorbar=True, ax=ax2)
    plt.show()
    code.interact(local=locals())
    return None

def create_distance_matrix(df):
    #Reference: https://kanoki.org/2019/12/27/how-to-calculate-distance-in-python-and-pandas-using-scipy-spatial-and-distance-functions/
    dd = df[["id","Latitude","Longitude"]].drop_duplicates()
    dd["Latitude"] = np.radians(dd["Latitude"])
    dd["Longitude"] = np.radians(dd["Longitude"])
    dist = DistanceMetric.get_metric("haversine") #can use haversine
    dm = pd.DataFrame(dist.pairwise(dd[["Latitude","Longitude"]].to_numpy())*6373, columns=dd.id.unique(), index=dd.id.unique()) #for distance in km
    return dm

def correlation_and_clustering_relative_to_baseline(df, use_pickled=True, fixed_distance_clustering=False):
    #---------------Classification of cells based on pre-crisis radiance levels---------
    drad = df[(df.date_c < "2015-03-01")].groupby(["id","Latitude","Longitude"]).mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()
    drad["rad_group"] = drad["RadE9_Mult_Nadir_Norm"].apply(lambda x: "low" if x<=15 else "medium" if 15<x<=50 else "high")
    drad = drad[["id","Latitude","Longitude","rad_group"]].drop_duplicates()

    #---------------get correlation values for pre-crisis levels------------------------
    if use_pickled == False:
        for year in [2012,2013,2014,2015,2016,2017,2018,2019]:
            print("Year = {}".format(year))
            df1 = df[(df.date_c.dt.year == year)]
            df1 = df1.pivot(index="date_c", columns="id", values="RadE9_Mult_Nadir_Norm")
            df1 = df1.corr()
            df1.to_hdf("yemen_ts_correlation_{}.h5".format(year), key="zeal")
            print("Done")
        print("all correlation files saved")

    #-----------run clustering for every year-------------------------------------------
    pre_crisis = precrisis_correlation(df, use_pickled=True, visualize_output=False)
    pre_crisis = pre_crisis.rename(columns={"labels":"labels_precrisis"})

    for year in [2015,2016,2017,2018,2019]:
        print("Year = {}".format(year))
        df1 = pd.read_hdf("yemen_ts_correlation_{}.h5".format(year), key="zeal")
        corr = df1.values
        corr = 1 - np.abs(corr)
        pdist = spc.distance.pdist(corr) #length = 3285 choose 2 = 5393970
        linkage = spc.linkage(pdist, method='complete')
        if fixed_distance_clustering == True:
            idx = spc.fcluster(linkage, 16.228307171761188, 'distance')
        else:
            idx = spc.fcluster(linkage, 0.5*pdist.max(), 'distance')
        drad["labels_{}".format(year)] = idx
        del(df1)

    drad = pd.merge(drad, pre_crisis[["id","labels_precrisis"]], left_on=["id"], right_on=["id"], how="left")
    drad_gdf = create_geodataframe(drad, radius=462, cap_style=3, buffered=True)

    #---------Visualize for all years---------------------------------------------------
    fig, (ax_precrisis,ax2015,ax2016,ax2017,ax2018,ax2019) = plt.subplots(nrows=1, ncols=6)
    ax = [ax_precrisis,ax2015,ax2016,ax2017,ax2018,ax2019]
    i = 0
    for year in ["precrisis",2015,2016,2017,2018,2019]:
        no_of_clusters = drad_gdf["labels_{}".format(year)].nunique()
        plot_geospatial_heatmap_with_event_locs(geo_df=drad_gdf, col_name="labels_{}".format(year), events_data=None, title="Year-{} (clusters={})".format(year,no_of_clusters), cmap=cm.tab10, cmap_type="tab10", needs_colormapping=False, marker_color="black", events_data_type="locations_buffered", add_title=True, event_locs_included=False, include_colorbar=False, ax=ax[i])
        i = i+1
    plt.show()
    code.interact(local=locals())
    return None

def study_changes_in_clusters():
    dc = pd.read_hdf("precrisis_postyearly_labels.h5", key="zeal")
    code.interact(local=locals())
    return None

if __name__=='__main__':
    #--------read & process necessary datasets----------
    # dg = pd.read_hdf("filtered_data.h5", key="zeal")
    # #clip negative radiance values to 0
    # dg["RadE9_Mult_Nadir_Norm"] = dg["RadE9_Mult_Nadir_Norm"].clip(lower=0)
    # # combine multiple readings for same day and same id
    # dg = dg.groupby(["id","Latitude","Longitude","date_c"]).mean()[["RadE9_Mult_Nadir_Norm","LI"]].reset_index()

    #---------read lunar corrected dataset--------------
    data_path = "./"
    dg = pd.read_hdf(data_path + "yemen_STL_lunar_corrected.h5", key="zeal")
    dg = dg.drop(columns = ["RadE9_Mult_Nadir_Norm"])
    dg = dg.rename(columns={"rad_corr":"RadE9_Mult_Nadir_Norm"})

    #---------create pre-crisis corr matrix-------------
    # precrisis_correlation(dg.copy(), use_pickled=True, visualize_output=False)

    #---------create post-crisis corr matrix-------------
    # postcrisis_correlation(dg.copy(), use_pickled=True)

    # visualize_pre_post_clustering(dg.copy())

    #---------calculate correlation for every year-------
    # correlation_and_clustering_per_year(dg.copy(), use_pickled=True, fixed_distance_clustering=False)
    # correlation_and_clustering_relative_to_baseline(dg.copy(), use_pickled=True, fixed_distance_clustering=False)

    #----Analyze changes in clusters over time-----------
    study_changes_in_clusters()