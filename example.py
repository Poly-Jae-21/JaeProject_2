import numpy as np


import geopandas as gpd
from pyogrio import read_dataframe
import gemgis as gg

import matplotlib.pyplot as plt
import numpy as np

import geopandas as gpd
from env.utils.data_conversion import Polygon_to_matrix


read_community_boundary_data = read_dataframe("env/MultiCity/Chicago/data/community_boundary_map/geo_export_b5a56d3a9_Project.shp")
numpy_array, gdf, x_min, y_min, x_max, y_max = Polygon_to_matrix().transform_data_community_boundary(read_community_boundary_data)
print(numpy_array.shape, gdf.shape, x_min, y_min, x_max, y_max)

kkkk = read_dataframe("env/MultiCity/Chicago/data/landuse_map/Landuse2018_Dissolve_Pr_Clip.shp")
numpy_array, xxxx, x_min_, y_min_, x_max_, y_max_ = Polygon_to_matrix().transform_data_landuse(kkkk)
print(numpy_array.shape, xxxx.shape, x_min_, y_min_, x_max_, y_max_)

existing_charging_infra = read_dataframe("env/MultiCity/Chicago/data/existing_infrastructure_map/Alternative_Fuel_Loc_Pr_Clip.shp")
existing_charging_infra = gg.vector.extract_xy(existing_charging_infra)
existing_charging_infra.X, existing_charging_infra.Y = np.trunc(existing_charging_infra.X / 10), np.trunc(existing_charging_infra.Y / 10)
print(np.min(existing_charging_infra.X), np.min(existing_charging_infra.Y), np.max(existing_charging_infra.X), np.max(existing_charging_infra.Y))