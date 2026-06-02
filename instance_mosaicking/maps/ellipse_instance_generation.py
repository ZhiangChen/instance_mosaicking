import fiona
from fiona.crs import CRS
from pyproj import CRS
import os, sys
from matplotlib.patches import Ellipse
from shapely.geometry import Polygon, mapping
import numpy as np
from tqdm import tqdm
from osgeo import gdal
from osgeo import osr

def generate_ellipse(center_x, center_y, width, height, angle):
    """generate an ellipse

    Args:
        center_x (float): ellipse center coordinate x
        center_y (float): ellipse center coordinate y
        width (float): ellipse major axis length
        height (float): ellipse minor axis length
        angle (float): ellipse orientation

    Returns:
        matplotlib.patches.Ellipse: ellipse object
    """
    ellipse = Ellipse((center_x, center_y), width, height, angle) 
    vertices = ellipse.get_verts()     # get the vertices from the ellipse object
    ellipse = Polygon(vertices)        # Turn it into a polygon
    return ellipse



def spawn_ellipses(tile_length, ellipse_length_range, density, tile_number, overlap, shp_path, generate_tiff, xres = 0.01, yres = 0.01):
    """spawn a list of random ellipses and save to a shapefile

    Args:
        tile_length (float): tile_length/tile_width in meters
        ellipse_length_range (tuple): (ellipse_min, ellipse_max)
        density (int): average number of ellipses per tile
        tile_number (int): number of tiles in horizontal or vertical direction
        overlap (float): overlap between adjacent tiles
        shp_path (string): shape file directory 
        generate_tiff (bool): generate fake tiff file for the boundary of the study area if True. otherwise, not. 
        xres (float): raster resolution in meters. xres and yres are used only when generate_tiff is True
        yres (float): raster resolution in meters
    """
    assert os.path.isdir(shp_path)
    # create random ellipses
    ellipses = []
    N = tile_number**2 * density
    X = (tile_length - overlap)* tile_number + overlap
    Y = (tile_length - overlap)* tile_number + overlap
    center_x_list = np.random.rand(N) * X
    center_y_list = np.random.rand(N) * Y
    angle_list = np.random.rand(N) * 180
    width_list = np.random.uniform(ellipse_length_range[0], ellipse_length_range[1], N)
    height_list = np.random.uniform(ellipse_length_range[0], ellipse_length_range[1], N)
    print('generate ellipse instances: ')
    for i in tqdm(range(N)):
        ellipse = generate_ellipse(center_x_list[i], center_y_list[i], width_list[i], height_list[i], angle_list[i])
        ellipses.append(ellipse)
    
    # write ellipses to a shapefile
    # create a schema
    epsg = 32611  # WGS 84 / UTM zone 11N
    crs = CRS.from_epsg(epsg)
    schema = {
    'geometry':'Polygon',
    'properties':[('id','str')]
    }

    # open a write file
    file_name = "ellipses_{density}_{tile_length}_{tile_number}_{overlap}_{pixel_size}_{ellipse_max}_{ellipse_min}.shp".format(
        density=density, tile_length=int(tile_length), tile_number=tile_number,  overlap=int(overlap*1000), pixel_size=int(xres*1000), ellipse_max=int(ellipse_length_range[0]*1000), ellipse_min=int(ellipse_length_range[1]*1000))
    file_path = os.path.join(shp_path, file_name)
    polyShp = fiona.open(file_path, mode='w', driver='ESRI Shapefile', schema = schema, crs = crs)

    # add polygons to file
    for id, polygon in enumerate(ellipses):
        rowDict = {'geometry' : mapping(polygon), 'properties': {'id': str(id)}}
        polyShp.write(rowDict)

    # write file
    polyShp.close()
    
    if generate_tiff:

        #  Choose some Geographic Transform (Around Lake Tahoe)
        lat = [0, (tile_length - overlap)*tile_number + overlap + ellipse_length_range[1]/2]
        lon = [0, (tile_length - overlap)*tile_number + overlap + ellipse_length_range[1]/2]

        #  Calculate the Image Size
        image_size = (int(lat[1]/xres), int(lon[1]/yres))

        #  Create Each Channel
        r_pixels = np.ones((image_size), dtype=np.uint8)*0

        # set geotransform
        nx = image_size[0]
        ny = image_size[1]
        xmin, ymin, xmax, ymax = [min(lon), min(lat), max(lon), max(lat)]
        geotransform = (xmin, xres, 0, ymax, 0, -yres)

        # create the 3-band raster file
        file_name = "ellipses_{density}_{tile_length}_{tile_number}_{overlap}_{pixel_size}_{ellipse_max}_{ellipse_min}.tif".format(
        density=density, tile_length=int(tile_length), tile_number=tile_number, overlap=int(overlap*1000), pixel_size=int(xres*1000), ellipse_max=int(ellipse_length_range[0]*1000), ellipse_min=int(ellipse_length_range[1]*1000))
        file_path = os.path.join(shp_path, file_name)
        
        dst_ds = gdal.GetDriverByName('GTiff').Create(file_path, ny, nx, 1, gdal.GDT_Byte)

        dst_ds.SetGeoTransform(geotransform)    # specify coords
        srs = osr.SpatialReference()            # establish encoding
        srs.ImportFromEPSG(epsg)                # WGS84 lat/long
        dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
        dst_ds.GetRasterBand(1).WriteArray(r_pixels)   # write r-band to the raster
        dst_ds.FlushCache()                     # write to disk
        dst_ds = None
    