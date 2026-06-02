import os
from instance_mosaicking.maps import ellipse_instance_generation as generation
from instance_mosaicking.maps.annotation_map_split import Dataset, Tile_Splitter
from fiona.crs import CRS
from instance_mosaicking.maps.evaluation import evaluate
from instance_mosaicking.maps.instance_registration import Instance_Registration
import shutil

class Workflow(object):
    def __init__(self, config):
        tile_length = config['tile_length']
        ellipse_max = config['ellipse_max']
        ellipse_min = config['ellipse_min']
        density = config['density']
        tile_number = config['tile_number']
        save_dir = config['save_dir'] # e.g., '/root/instance_mosaicking/data/random_generation'
        overlap = config['overlap']
        pixel_size = config['pixel_size']
        iou_threshold = config['iou_threshold']
        zip = config['zip']
        
        file_prefix = "ellipses_{density}_{tile_length}_{tile_number}_{overlap}_{pixel_size}_{ellipse_max}_{ellipse_min}".format(
        density=density, tile_length=int(tile_length), tile_number=tile_number, overlap=int(overlap*1000), pixel_size=int(pixel_size*1000), ellipse_max=int(ellipse_min*1000), ellipse_min=int(ellipse_max*1000))
        save_sub_dir = os.path.join(save_dir, file_prefix) # e.g., '/root/instance_mosaicking/data/random_generation/ellipses_400_10_3_300_2000'
        

        if not os.path.exists(save_sub_dir):
            os.makedirs(save_sub_dir)
            
        generate_shp_path = os.path.join(save_sub_dir, file_prefix+'.shp')
        generate_shp_tif = os.path.join(save_sub_dir, file_prefix+'.tif')
        split_shp_tif_dir = os.path.join(save_sub_dir, "split_shp_tif")
        merge_shp_path = os.path.join(save_sub_dir, file_prefix+'_merge_'+str(int(iou_threshold*100))+'.shp')
        
        epsg = 32611  # WGS 84 / UTM zone 11N
        crs = CRS.from_epsg(epsg)
        
        
        generation.spawn_ellipses(tile_length=tile_length, ellipse_length_range=(ellipse_min, ellipse_max), 
                                  density=density, tile_number=tile_number, overlap=overlap,
                                  shp_path=save_sub_dir, generate_tiff=True, xres = pixel_size, yres = pixel_size)
        
        
        split_config = {'shapefile_path': generate_shp_path, 
                    'save_dir': split_shp_tif_dir, 
                    'crs':crs, 
                    'area_x1':0, 
                    'area_y1':0, 
                    'area_x2': (tile_length - overlap) * tile_number + overlap, 
                    'area_y2': (tile_length - overlap) * tile_number + overlap, 
                    'tile_size': tile_length, 
                    'overlap': overlap,
                    'tif_path': generate_shp_tif, 
                    'keep_instance_tif': True}

        ts = Tile_Splitter(**split_config)
        ts.split()
        
        
        # crop the study area (remove edges)
        split_config = {'shapefile_path': generate_shp_path, 
                    'save_dir': save_sub_dir, # this should be sub dir
                    'crs':crs, 
                    'area_x1':0, 
                    'area_y1':0, 
                    'area_x2':(tile_length - overlap) * tile_number + overlap, 
                    'area_y2':(tile_length - overlap) * tile_number + overlap, 
                    'tile_size':(tile_length - overlap) * tile_number + overlap + ellipse_max/2., # just change this to the size of study area. here because the ellipses go slightly than the study area, if we set tile_size of 30, the exceeding part will be split into other tiles. thus we have a large tile_size here.
                    'overlap':0,
                    'tif_path': None, 
                    'keep_instance_tif': True}

        ts = Tile_Splitter(**split_config)
        ts.split()
        
        # generate pickle files
        ellipse_data = Dataset(pixel_size=int(tile_length/pixel_size), split_path=split_shp_tif_dir, input_channel=(0,))
        ellipse_data.save_pickles(zip=zip)
        
        
        # instance registration
        ir = Instance_Registration(split_shp_tif_dir, 
                                   merge_shp_path, 
                                   tif_width_pixel=int(tile_length/pixel_size),
                                   tif_height_pixel=int(tile_length/pixel_size),
                                   tif_width_res=pixel_size,
                                   tif_height_res=-pixel_size,
                                   tile_overlap_ratio=overlap/tile_length,
                                   iou_threshold=iou_threshold, 
                                   disable_merge=False, 
                                   unzip=zip)

        updated_tile_files, timestamps = ir.start_registration()
        ir.combine_shapefiles()
        
        
        print(merge_shp_path)
        evaluate(os.path.join(save_sub_dir, '0_0.shp'), os.path.join(save_sub_dir, merge_shp_path))

        
        # clear files
        """
        npy_path = os.path.join(split_shp_tif_dir, file_prefix+'_time.npy')
        parent_dir = os.path.abspath(os.path.join(npy_path, os.pardir, os.pardir))
        new_npy_path = os.path.join(parent_dir, file_prefix+'_time.npy')
        if os.path.exists(new_npy_path):
            os.remove(new_npy_path)
        shutil.move(npy_path, parent_dir)
        shutil.rmtree(split_shp_tif_dir)
        """
        
# main
if __name__ == "__main__":
    """
    config = {'density':30, 
              'tile_length':10, 
              'tile_number':5, 
              'overlap':1, 
              'pixel_size': 0.01, # pixel size shouldn't be greater than ellipse_min
              'ellipse_min':0.5, 
              'ellipse_max':2, 
              'iou_threshold': 0.75,
              'zip':True,
              'save_dir':"/root/instance_mosaicking/data/random_generation"}
    
    """

    
    config = {'density':30, 
              'tile_length':2000, 
              'tile_number':5, 
              'overlap':100, 
              'pixel_size': 1, # pixel size shouldn't be greater than ellipse_min
              'ellipse_min':50, 
              'ellipse_max':200, 
              'iou_threshold': 0.75,
              'zip':True,
              'save_dir':"/root/instance_mosaicking/data/random_generation_8"}
    #"""

    w = Workflow(config)
