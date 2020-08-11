from shapely.geometry import Point, Polygon
import pandas as pd
import geopandas as gpd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import code
import datetime

# read Yemen's shapefile and filter out shapes corresponding to regions of interest
street_map = gpd.read_file("../yemen_shp_files/yem_admbnda_adm2_govyem_mola_20181102.shp")
street_map = street_map[(street_map.ADM1_EN.isin(["Amanat Al Asimah","Sana'a"]))]

def basic_preprocessing(db):
    db["dow"] = db.date_c.apply(lambda x: x.dayofweek)
    db["year"] = db.date_c.dt.year
    # db = db.groupby(["id", "Latitude", "Longitude", "year", "dow"]).mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()
    db["dayname"] = db.dow.apply(lambda x: "Mon" if x==0 else "Tue" if x==1 else "Wed" if x==2 else "Thu" if x==3 else "Fri" if x==4 else "Sat" if x==5 else "Sun")
    db["weekname"] = db.dow.apply(lambda x: "Weekend" if x>=5 else "Weekday")
    # code.interact(local=locals())
    return db

def lineplot_average_dow_rad_by_year(db):
    # take mean of radiance when there are multiple readings for the same date
    db = db.groupby(["id","date_c"]).mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()

    # compute sum value of night lights (TNL) in the region on a daily bases
    db = db.groupby("date_c").sum()[["RadE9_Mult_Nadir_Norm"]].reset_index()

    db["dow"] = db.date_c.apply(lambda x: x.dayofweek)
    db["year"] = db.date_c.dt.year
    db["dayname"] = db.dow.apply(lambda x: "Mon" if x==0 else "Tue" if x==1 else "Wed" if x==2 else "Thu" if x==3 else "Fri" if x==4 else "Sat" if x==5 else "Sun")
    db["weekname"] = db.dow.apply(lambda x: "Weekend" if x>=5 else "Weekday")

    db = db.groupby(["year","dayname","dow"]).mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()
    db = db.sort_values(by=["year","dow"])

    fig, ax = plt.subplots()
    i=0
    mark = ["*","^","s","v","o","P","d"]
    labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    for l in labels:
        group = db[(db.dayname==l)]
        group.plot(x="year", y="RadE9_Mult_Nadir_Norm", ax=ax, marker = mark[i], label=l, linewidth=1.5)
        i=i+1
    # for dn, group in db.groupby("dayname"):
    #     group.plot(x="year", y="RadE9_Mult_Nadir_Norm", ax=ax, marker = mark[i], label=dn, linewidth=1.5)
    #     i = i+1
    plt.xlabel("Year")
    plt.ylabel("Radiance")
    plt.title("Average of total night lights per year by day of the week")
    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    plt.show()
    code.interact(local=locals())
    return None

def boxplot_weekday_weekend_rad(db):
    # take mean of radiance when there are multiple readings for the same date
    db = db.groupby(["id","date_c"]).mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()

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
    plt.ylabel("Radiance")
    plt.title("Variation in total night lights during weekdays and weekends")
    plt.show()
    code.interact(local=locals())
    return None

def lineplot_rad_by_dow(db):
    # take mean of radiance when there are multiple readings for the same date
    db = db.groupby(["id","date_c"]).mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()

    # compute sum value of night lights (TNL) in the region on a daily bases
    db = db.groupby("date_c").sum()[["RadE9_Mult_Nadir_Norm"]].reset_index()

    db["dow"] = db.date_c.apply(lambda x: x.dayofweek)
    db["year"] = db.date_c.dt.year
    db["dayname"] = db.dow.apply(lambda x: "Mon" if x==0 else "Tue" if x==1 else "Wed" if x==2 else "Thu" if x==3 else "Fri" if x==4 else "Sat" if x==5 else "Sun")
    db["weekname"] = db.dow.apply(lambda x: "Weekend" if x>=5 else "Weekday")

    db = db.groupby(["year","dayname","dow"]).mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()
    db = db.sort_values(by=["year","dow"])

    fig, ax = plt.subplots()
    i=0
    mark = ["*","^","s","v","o","P","d","h"]
    for yr, group in db.groupby("year"):
        group.plot(x="dayname", y="RadE9_Mult_Nadir_Norm", ax=ax, marker = mark[i], label=yr, linewidth=2)
        i = i+1
    # db.groupby("year").plot(x="dayname", y="RadE9_Mult_Nadir_Norm", ax=ax, label=db.year)
    # sns.lineplot(x="dayname", y="RadE9_Mult_Nadir_Norm", hue="year", style="year", data=db, markers=True, dashes=False, palette="Dark2", estimator=None)
    plt.xlabel("Day of the week")
    plt.ylabel("Radiance")
    plt.title("Mean TNL for the region by day of week")
    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    plt.show()
    code.interact(local=locals())
    return None

def spatial_overall_baselines(df):
    # take mean of radiance when there are multiple readings for the same date
    df = df.groupby(["id","Latitude","Longitude","date_c"]).mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()

    df["year"] = df.date_c.dt.year
    # df = df.groupby(["id","Latitude","Longitude","year"]).agg(["mean", "var", "std"])[["RadE9_Mult_Nadir_Norm"]].reset_index()
    # df.columns = ["id","Latitude","Longitude","year","mean_rad","var_rad","std_rad"]

    ##########################Comment out the above two lines to produce graphs given below#################
    # Use the code line below instead
    df = df.groupby(["id","Latitude","Longitude","year"]).mean()[["RadE9_Mult_Nadir_Norm"]].reset_index()
    # dg = df.pivot_table(index="id", columns="year", values="RadE9_Mult_Nadir_Norm").reset_index()

    ## create CDFs of points and their mean yearly radiance values subplots
    fig, ax = plt.subplots(2,4)
    i=0
    j=0
    for y in df.year.unique():
        df1 = df[(df.year==y)]
        sorted_data = np.sort(df1["RadE9_Mult_Nadir_Norm"])
        yvals = np.arange(len(sorted_data))*100.0/float(len(sorted_data) - 1)
        ax[i,j].plot(sorted_data, yvals, color="k")
        ax[i,j].set_title(y)
        j = j+1
        if j == 4:
            j=0
            i=1
    fig.text(0.5, 0.04, "Radiance Levels", ha='center')
    fig.text(0.04, 0.5, "Proportion of grid cells (%)", va='center', rotation='vertical')
    fig.suptitle("CDF of grid cells by radiance")
    fig.tight_layout()
    plt.show()

    ## create CDFs of points and their mean yearly radiance values in single plot
    # fig, ax = plt.subplots()
    # for y in df.year.unique():
    #     df1 = df[(df.year==y)]
    #     sorted_data = np.sort(df1["RadE9_Mult_Nadir_Norm"])
    #     yvals = np.arange(len(sorted_data))*100.0/float(len(sorted_data) - 1)
    #     ax.plot(sorted_data, yvals, label=y)
    # plt.legend()
    # plt.xlabel("Radiance Levels")
    # plt.ylabel("Proportion of grid cells (%)")
    # plt.title("CDF of grid cells by radiance")
    # plt.show()

    code.interact(local=locals())
    return None

def spatial_weekly_baselines(db):

    # compute mean, var and std of night lights for every day of week per grid cell per year
    db["dow"] = db.date_c.apply(lambda x: x.dayofweek)
    db["year"] = db.date_c.dt.year
    db["dayname"] = db.dow.apply(lambda x: "Mon" if x==0 else "Tue" if x==1 else "Wed" if x==2 else "Thu" if x==3 else "Fri" if x==4 else "Sat" if x==5 else "Sun")
    # db["weekname"] = db.dow.apply(lambda x: "Weekend" if x>=5 else "Weekday")

    db = db.groupby(["id","Latitude","Longitude","year","dayname","dow"]).agg(["mean","var","std"])[["RadE9_Mult_Nadir_Norm"]].reset_index()
    db.columns = ["id","Latitude","Longitude","year","dayname","dow","mean_rad","var_rad","std_rad"]
    code.interact(local=locals())
    return None


def plot_heatmap(df, col_name, title, cmap, city, airport, with_sites=False):
    crs = {'init': 'epsg:4326'}
    geometry = [Point(xy) for xy in zip(df["Longitude"], df["Latitude"])]
    geo_df = gpd.GeoDataFrame(df, crs = crs, geometry = geometry)

    street_map['coords'] = street_map['geometry'].apply(lambda x: x.representative_point().coords[:])
    street_map['coords'] = [coords[0] for coords in street_map['coords']]

    fig, ax = plt.subplots(figsize = (10, 6))
    xlim = ([geo_df.total_bounds[0],  geo_df.total_bounds[2]])
    ylim = ([geo_df.total_bounds[1],  geo_df.total_bounds[3]])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    street_map.plot(ax = ax, color = 'white', edgecolor = 'black')
    geo_df.plot(ax = ax, column=col_name, markersize = 20, cmap=cmap, legend = True, alpha = 0.8)

    if with_sites:
        city.plot(ax=ax, color="black", markersize=10, marker="x", alpha=0.6)
        airport.plot(ax=ax, color="black", markersize=10, marker="x", alpha=0.6)

    ax.set_axis_off()
    ax.set_title(title)
    plt.show()
#     plt.savefig(re+'/updated_recovery.pdf')
    return geo_df

if __name__=='__main__':
    re = "../sanaa_csv/"

    #---read pickled dataset------------------
    # read yemen NL data
    df = pd.read_hdf("../yemen_filtered_db.h5", key="zeal")
    df["RadE9_Mult_Nadir_Norm"] = df["RadE9_Mult_Nadir_Norm"].clip(lower=0) #clip negative readings to 0
    code.interact(local=locals())
    # code.interact(local=locals())
    # lineplot_average_dow_rad_by_year(df)
    # boxplot_weekday_weekend_rad(df)
    # lineplot_rad_by_dow(df)
    # spatial_overall_baselines(df)
    # spatial_weekly_baselines(df)
    # yearly_monthly_daily_boxplot(df)

    # gdf = plot_heatmap(dz_crisis, "z_score", "Disaster Map (Z-scores)", "seismic", sites_city, sites_airport, with_sites=False)

    # read Yemen's shapefile and filter out shapes corresponding to regions of interest
    # street_map = gpd.read_file("./yemen_shp_files/yem_admbnda_adm2_govyem_mola_20181102.shp")
    # street_map = street_map[(street_map.ADM1_EN.isin(["Amanat Al Asimah","Sana'a"]))]

    # # read disaster data
    # de = pd.read_pickle("./extra_datasets/conflict_data_filtered.pck")
    # de = de[(de.fatalities != 0)]

    # # read damaged structures dataset released by UNITAR
    # sites_city = gpd.read_file("./extra_datasets/unitar_unisat_data/Damage_Sites_Sanaa_City_20150910.shp")
    # sites_airport = gpd.read_file("./extra_datasets/unitar_unisat_data/Damage_Sites_Sanaa_Airport_20150515.shp")