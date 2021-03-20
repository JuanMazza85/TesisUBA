# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:48:17 2019

@author: Juani
@Desc: Plots de GC_Metadata.xlsx
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
from geovoronoi import voronoi_regions_from_coords
from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area
import geopandas as gpd
from shapely.geometry import Point


RootFolder = 'C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\'
ExcelFile  = 'Datos Lluvias\\Gran Chaco\\GC_METADATA.xlsx'


#Excel Cols
col_idOMM	 = 0
col_NomEstacion = 1	
col_Longitud	 = 2
col_Latitud	= 3
col_Elevacion = 4


#Read File
File = pd.read_excel(RootFolder + ExcelFile)
 
#Plot on Map
crs = {'init': 'epsg:4326'}

MapaArgentina = gpd.read_file(RootFolder + '\\Shape Files\\boundArgentina72.shp')
MapaArgentina = MapaArgentina.to_crs(crs)

GranChaco = gpd.read_file(RootFolder + '\\Shape Files\\GranChacoCompleto.shp')
GranChaco = GranChaco.to_crs(crs)

geometry = [Point(xy) for xy in zip(File.Longitud, File.Latitud)]
geo_df = gpd.GeoDataFrame(File, crs = crs, geometry = geometry)

PoligonoArgentina = MapaArgentina.iloc[0].geometry
PointsXY = []
[PointsXY.append(xy) for xy in zip(File.Longitud, File.Latitud)]
voronoiShapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(PointsXY, PoligonoArgentina)
voronoiDF = gpd.GeoDataFrame(None, crs = crs, columns=["polygon"], geometry = voronoiShapes )


%matplotlib qt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.xaxis.set_label_text("Longitud")
ax.yaxis.set_label_text("Latitud")
plt.xlim([-70, -55.5])
plt.ylim([-34, -21.8])
MapaArgentina.plot(ax = ax, alpha = 1, color = 'grey')
voronoiDF.plot(ax = ax, alpha = 1, cmap='OrRd')

GranChaco.plot(ax = ax, alpha = 0.5, color = 'yellow')
geo_df.plot(ax = ax, markersize = 50, color = 'blue', marker = 'o')


for x, y, label in zip(geo_df.geometry.x, geo_df.geometry.y, geo_df.NomEstacion):
    if label in ['Chilecito Aero', 'Resistencia Aero','Ceres Aero', 'Cordoba Aero']:
        y = y + .3
    ax.annotate(label, xy=(x, y), xytext=(-15, -10), textcoords="offset points")

