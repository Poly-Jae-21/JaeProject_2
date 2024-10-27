import gemgis as gg
from pyogrio import read_dataframe



read_landuse_data =  read_dataframe("../World_Environment/envs/data/community_boundary_map/geo_export_b5a56d3a9_Project.shp")
read_landuse_data_no_multipolygon = read_landuse_data[read_landuse_data['geometry'].geom_type != 'MultiPolygon']

multipolygon_rows = read_landuse_data_no_multipolygon[read_landuse_data_no_multipolygon['geometry'].geom_type == 'MultiPolygon']

gdf2 = gg.vector.extract_xy(read_landuse_data_no_multipolygon)
