

from instance_mosaicking.maps.instance_registration import Instance_Registration


tile_length = 500
pixel_size = 1
iou_threshold = 0.75
overlap = 100

ir = Instance_Registration("/root/instance_mosaicking/data/random_generation/ellipses_30_500_5_100000_1000_50000_200000/split_shp_tif", 
                           "/root/instance_mosaicking/data/random_generation/ellipses_30_500_5_100000_1000_50000_200000/ellipses_30_500_5_100000_1000_50000_200000_merge.shp", 
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