import pickle
import os
import numpy as np
import rioxarray
import cv2
import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely.geometry import Polygon
from tqdm import tqdm
import time
from random import shuffle
import matplotlib.pyplot as plt
from PIL import Image
import zipfile
import pandas as pd
from shapely.ops import unary_union
import tracemalloc

class Instance_Registration(object):
    def __init__(self, instance_dir, 
                 save_shapefile,
                 tif_height_pixel = 1000,
                 tif_width_pixel = 1000,
                 tif_height_res = -0.01,
                 tif_width_res = 0.01,
                 tif_map_file = None,
                 tile_overlap_ratio=0.1, 
                 detection_threshold=0.75, 
                 segmentation_threshold=0.5, 
                 iou_threshold=0.5, 
                 disable_merge=False,
                 test=True,
                 unzip=False):
        """merge and register instances using split pickle files and tiff tiles. The pickle file name is consistant with the names of its corresponding shapefile and tif file. Each pickle file includes a dictionary of instance predictions, e.g., {'image'=, 'bb'=, 'labels'=, 'scores'=, 'masks'=, 'image_name'=, 'ids'=}.
        'image_name' has the absolute path of the image. 'masks' is masks of prediction instances with segmentation scores. 'ids' is designed to evaluate instance registration for fake ellipse instance data. It indicates the ids of the original instances which the split instances come from. For actual instance segmentation practice, 'ids' can be random integers.
        
        Args:
            instance_dir (string): directory of split tiles. it should include split pickle files and tiff tiles.
            save_shapefile (string): file to save the merged shapefile. 
            tif_height_pixel (int, optional): tif file height in pixel. Defaults to 1000.
            tif_width_pixel (int, optional): tif file width in pixel. Defaults to 1000.
            tif_height_res (float, optional): tif file resolution in height, meters per pixel. Origin of a tif file is at top left, thus the pixel size in height is negative. This argument can be found using `dalinfo x.tif`. Pixel Size = (0.010000000000000,-0.010000000000000). Defaults to -0.01.
            tif_width_res (float, optional): tif file resolution in width, meters per pixel. Defaults to 0.01.
            tif_map_file (string, optional): the path of the tif map.
            tile_overlap_ratio (float, optional): tile overlap ratio scale. For example, if overlap is 1 m and tile height is 10 m, then the tile_overlap_ratio is 0.1. Defaults to 0.1.
            detection_threshold (float, optional): only instances with detection scores greater than the detection_threshold will be considered for instance registration. Defaults to 0.75.
            segmentation_threshold (float, optional): only pixels with segmentation scores greater than the segmentation threshold will be considered for instance registration. Defaults to 0.5.
            iou_threshold (float, optional): if polygons from two instances at their overlap region exceed the iou_threshold, the two instances will be merged. Defaults to 0.5.
            disable_merge (bool, optional): skip instance registration if True. Defaults to False.
            test (bool, optional): set True if masks in pickle files has three dimensions; set False for four dimensions. This difference is caused by the fact that real instance segmentation and fake ellipse generation have different dimensions for masks. Defaults to True.
        """
        
        
        assert os.path.exists(instance_dir)
        self.unzip = unzip
        if self.unzip:
            self.tile_files = [os.path.join(instance_dir, f) for f in os.listdir(instance_dir) if f.endswith('.zip')]
        else:
            self.tile_files = [os.path.join(instance_dir, f) for f in os.listdir(instance_dir) if f.endswith('.pickle')]
        
        self.instance_dir = instance_dir
        
        coords = [self._get_coords_from_file_name(f) for f in self.tile_files]
        coords.sort()
        coords = np.array(coords)
        self.X = np.max(coords[:, 0])
        self.Y = np.max(coords[:, 1])
        
        self.tile_files = []
        for coord in coords:
            if self.unzip:
                f = str(coord[0]) + '_' + str(coord[1]) + '.zip'
                self.tile_files.append(os.path.join(instance_dir, f))
            else:
                f = str(coord[0]) + '_' + str(coord[1]) + '.pickle'
                self.tile_files.append(os.path.join(instance_dir, f))
        
        self.tiles = {}
        self.instances = []  # 
        tile_data = self._get_instance(self.tile_files[1])
        self.test = test
        if self.test == True:
            _, self.h, self.w = tile_data['masks'].shape
        else:
            _, _, self.h, self.w = tile_data['masks'].shape
            
        #print(self.h, self.w)
        
        tif_name = tile_data['image_name']
        tif = rioxarray.open_rasterio(tif_name)
        epsg = tif.rio.crs.to_epsg()
        self.crs = CRS(epsg)
        #_, self.tiff_h, self.tiff_w = tif.shape
        #self.tif_h_size, self.tif_v_size = tif.rio.resolution()
        
        self.tiff_h, self.tiff_w = tif_height_pixel, tif_width_pixel
        self.tif_h_size, self.tif_v_size = tif_width_res, tif_height_res
        
        #print(self.tiff_h, self.tiff_w)

        self.mask_h_size = self.tif_h_size * self.tiff_h / self.h
        self.mask_v_size = self.tif_v_size * self.tiff_w / self.w
        self.overlap = int(self.h * tile_overlap_ratio)
        self.overlap_float = self.h * tile_overlap_ratio
        self.iou_threshold = iou_threshold
        
        #print(self.mask_h_size, self.mask_v_size)
        
        #print(self.mask_h_size, self.mask_v_size)
        
        if tif_map_file is None:
            self.h_start, self.v_start = 0, 0
        else:
            #dir_path = os.path.dirname(os.path.realpath(tif_name))
            #tif_name = os.path.join(dir_path, '0_0.tif')
            assert os.path.isfile(tif_map_file)
            tif = rioxarray.open_rasterio(tif_map_file)
            self.h_start, self.v_start, _, _= tif.rio.bounds()
            
        self.save_shapefile = save_shapefile
        self.detection_threshold = detection_threshold
        self.segmentation_threshold = segmentation_threshold
        self.disable_merge = disable_merge
        self.twins = []
        
        
    def start_registration(self, continue_regitration=False):
        updated_tile_files = []
        timestamps = []
        memories = []
        self.instances = []
        self.tiles = {}
        print("Instance registration: ")
        if continue_regitration:
            twin_file = self.save_shapefile[:-4] + "_twins.npy"
            if os.path.isfile(twin_file):
                with open(twin_file, 'rb') as f:
                    self.twins = pickle.load(f)
                
        start_time = time.time()
        tracemalloc.start()
        for tile_file in tqdm(self.tile_files):
            if continue_regitration:
                if os.path.isfile(tile_file.split('.')[0]+"_merge.shp"):
                    continue            
            tile_data = self._get_instance(tile_file)
            masks = tile_data['masks']
            instance_N = masks.shape[0]
            if instance_N == 0:
                continue
            updated_tile_files.append(tile_file)
            if not self.test:
                masks = np.squeeze(masks, axis=1)
            tif_name = tile_data['image_name']
            #print(tif_name) # debug
            tile_indices = tuple([int(i) for i in tif_name.split('/')[-1].split('.')[0].split('_')])
            #print("processing tile: ", tile_indices)
            # post processing: detection confidence filter
            detect_scores = tile_data['scores']
            id_strs = tile_data['ids'] 
            # prune data by detection_threshold
            masks = masks[detect_scores>self.detection_threshold]
            id_strs = id_strs[detect_scores>self.detection_threshold] 
            detect_scores = detect_scores[detect_scores>self.detection_threshold]
            tif = rioxarray.open_rasterio(tif_name) 
            for idx, mask in enumerate(masks):
                # post processing: segmentation confidence filter
                mask = mask > self.segmentation_threshold
                # post processing: contour analysis
                contours, _ = cv2.findContours(mask.astype(np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                if len(contours) > 1:
                    areas = [cv2.contourArea(cnt) for cnt in contours]
                    i = np.argmax(areas)
                    contour = contours[i]
                    mask = np.zeros_like(mask).astype(np.uint8)
                    cv2.fillPoly(mask, pts =[contour], color=(255))
                    local_mask = mask > 0
                else:
                    contour = contours[0]
                    local_mask = mask
                # get location using pixel coords: top, bottom, left, right, middle, top-left, top-right, bottom-left, bottom-right
                contour = np.squeeze(contour, axis=1)
                locations = self._get_locations(contour, local_mask, tile_indices)
                # convert masks to global mask
                global_mask = self._convert_global_mask(local_mask, tile_indices)
                if len(global_mask)<4:
                    continue
                
                # get global bbox
                global_bbox = self._get_global_bbox(global_mask)
                # instance registration
                instance = {'locations': locations, 
                            'score': detect_scores[idx], 
                            'global_bbox': global_bbox, 
                            'global_mask': global_mask,
                            'id_str': id_strs[idx],}

                if self.disable_merge:
                    self.instances.append(instance)
                else:
                    self._instance_registration(instance)
            timestamps.append(time.time() - start_time)
            
            self.register_tiles(tile_indices)
            current, peak = tracemalloc.get_traced_memory() 
            memories.append(current / 10**6)
            
            
        tracemalloc.stop()    
        self._save_remaining_tiles()
        np.save(self.save_shapefile[:-4] + "_time_space.npy", np.asarray((timestamps, memories)))
        return updated_tile_files, timestamps
    
    
    def _save_remaining_tiles(self):
        dataframesList = []
        for id, instance in enumerate(self.instances):
            if instance is None:
                continue
            geodataframe = gpd.GeoDataFrame(pd.DataFrame({'instance': id, 'score': [instance['score']], 'id': instance['id_str']}), 
                                           crs=self.crs, 
                                           geometry=[self._convert_mask_to_poly(instance)])
            dataframesList.append(geodataframe)
        
        if len(dataframesList) > 0:
            rdf = gpd.GeoDataFrame(pd.concat(dataframesList, ignore_index=True))
            rdf.to_file(self.save_shapefile[:-4]+"_remaining.shp")
    
                
    def combine_shapefiles(self):
        shp_files = [os.path.join(self.instance_dir, f) for f in os.listdir(self.instance_dir) if f.endswith("merge.shp")]
        if os.path.isfile(self.save_shapefile[:-4]+"_remaining.shp"):
            pd_files = [gpd.read_file(self.save_shapefile[:-4]+"_remaining.shp")]
        else:
            pd_files = []
        for shp_file in shp_files:
            pd_files.append(gpd.read_file(shp_file))
            
        self.gdf = pd.concat(pd_files, ignore_index=True)
        
        self.clean_twin_instances()
        # write the merged GeoDataFrame to a new shapefile
        #self.gdf.to_file(self.save_shapefile[:-4]+"_"+str(int(self.iou_threshold*100)) +".shp")
        self.gdf.to_file(self.save_shapefile)
        
        
    def register_tiles(self, tile_indices):
        #print("processing tile: ", tile_indices)
        for adjacent_indices in self._get_adjacent_tiles(tile_indices):
            if adjacent_indices not in self.tiles.keys():
                continue
            register_flag = True
            for i in self._get_adjacent_tiles(adjacent_indices):
                if i not in self.tiles.keys():
                    register_flag = False
            if register_flag:
                self._save_tile(adjacent_indices)
    
    def _get_adjacent_tiles(self, tile_indices):
        adjacent_tile_indices = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if (i !=0) | (j != 0):
                    a = tile_indices[0]+i
                    b = tile_indices[1]+j
                    if (0<= a <= self.X) & (0 <= b <= self.Y):
                        adjacent_tile_indices.append((a, b ))
        return adjacent_tile_indices
        
        
    def _save_tile(self, tile_indices):
        dataframesList = []
        for k,v in self.tiles[tile_indices].items():
            for id in v:
                instance = self.instances[id]
                if instance is None:
                    continue
                
                geodataframe = gpd.GeoDataFrame(pd.DataFrame({'instance': id, 'score': [instance['score']], 'id': instance['id_str']}), 
                                                crs=self.crs, 
                                                geometry=[self._convert_mask_to_poly(instance)])
                dataframesList.append(geodataframe)
                self.instances[id] = None
        
        if len(dataframesList) > 0:
            rdf = gpd.GeoDataFrame(pd.concat(dataframesList, ignore_index=True))
            save_shapefile = os.path.join(self.instance_dir, str(tile_indices[0]) + "_" + str(tile_indices[1]) + "_merge.shp")
            rdf.to_file(save_shapefile)
            #print(save_shapefile)

                
    def _instance_registration(self, instance):

        (x,y) = list(instance['locations'].keys())[0]
        if 'middle' in instance['locations'][(x, y)].keys():
            self._add_instance(instance, middle=True)
            return None
        
        merged = []
        for location in instance['locations'][(x, y)]:
            if location == 'left':
                adjacent_indices = (x-1, y)
                adjacent_location = 'right'
                merged.append(self._merge_instance(instance, location, adjacent_indices, adjacent_location))
            if location == 'right':
                adjacent_indices = (x+1, y)
                adjacent_location = 'left'
                merged.append(self._merge_instance(instance, location, adjacent_indices, adjacent_location))
            if location == 'top':
                adjacent_indices = (x, y+1)
                adjacent_location = 'bottom'
                merged.append(self._merge_instance(instance, location, adjacent_indices, adjacent_location))
            if location == 'bottom':
                adjacent_indices = (x, y-1)
                adjacent_location = 'top'
                merged.append(self._merge_instance(instance, location, adjacent_indices, adjacent_location))
            if location == 'top-left':
                adjacent_indices = (x-1, y+1)
                adjacent_location = 'bottom-right'
                merged.append(self._merge_instance(instance, location, adjacent_indices, adjacent_location))
            if location == 'top-right':
                adjacent_indices = (x+1, y+1)
                adjacent_location = 'bottom-left'
                merged.append(self._merge_instance(instance, location, adjacent_indices, adjacent_location))
            if location == 'bottom-left':
                adjacent_indices = (x-1, y-1)
                adjacent_location = 'top-right'
                merged.append(self._merge_instance(instance, location, adjacent_indices, adjacent_location))
            if location == 'bottom-right':
                adjacent_indices = (x+1, y-1)
                adjacent_location = 'top-left'
                merged.append(self._merge_instance(instance, location, adjacent_indices, adjacent_location))
                
        merged = np.array(merged)
        merged_num = np.unique(merged[merged>=0]).shape[0]
        if merged_num == 0:
            self._add_instance(instance)
        elif merged_num > 1:
            self.twins.append(np.unique(merged[merged>=0]))
            f_path = self.save_shapefile[:-4] + "_twins.pickle"
            with open(f_path, 'wb') as file:
                # write the data to the file
                pickle.dump(self.twins, file)  
            
                
    def _add_instance(self, instance, middle=False):
        if middle == True:
            simple_instance = instance.copy()
            del simple_instance['locations']
            self.instances.append(simple_instance)
        else:
            self.instances.append(instance)
        instance_id = len(self.instances) - 1
        # update tile table
        tile_indices = list(instance['locations'].keys())[0]
        if not self.tiles.get(tile_indices, False):
            # initialize tile
            empty_locations = {'left':[], 'right':[], 'top':[], 'bottom':[], 'middle':[], 'top-left':[], 'top-right':[], 'bottom-left':[], 'bottom-right':[]}
            self.tiles[tile_indices] = empty_locations
        # update tile
        tile_indices = list(instance['locations'].keys())[0]
        for location in instance['locations'][tile_indices].keys():
            self.tiles[tile_indices][location].append(instance_id)
                        
    def _merge_instance(self, instance, location, adjacent_indices, adjacent_location):
        if not self.tiles.get(adjacent_indices, False): 
            return -1
        else:
            for adjacent_id in self.tiles[adjacent_indices][adjacent_location]:
                adjacent_instance = self.instances[adjacent_id]
                if adjacent_instance is None:
                    print(adjacent_indices, adjacent_id)
                    
                if self._bbox_intersection(instance['global_bbox'], adjacent_instance['global_bbox']):
                    tile_indices = list(instance['locations'].keys())[0]
                    mask1 = instance['locations'][tile_indices][location]
                    mask2 = adjacent_instance['locations'][adjacent_indices].get(adjacent_location, None)
                    iou = self._mask_IoU(mask1, mask2)
                    #print(iou) # debug
                    if iou > self.iou_threshold:
                        merged_mask = np.concatenate((instance['global_mask'], adjacent_instance['global_mask']))
                        merged_mask = np.unique(merged_mask, axis=0)
                        merged_mask = tuple(map(tuple, merged_mask))

                        self.instances[adjacent_id]['global_mask'] = merged_mask
                        self.instances[adjacent_id]['global_bbox'] = self._get_global_bbox(merged_mask)
                        self.instances[adjacent_id]['score'] = np.max((self.instances[adjacent_id]['score'], instance['score']))
                        self.instances[adjacent_id]['id_str'] += ','+ instance['id_str'] ########################################################
                        merged_mask_partial = np.concatenate((mask1, mask2))
                        merged_mask_partial = np.unique(merged_mask_partial, axis=0)
                        merged_mask_partial = tuple(map(tuple, merged_mask_partial))

                        # after merging, instance belongs to multiple tiles. instance's new tile should be updated
                        if self.instances[adjacent_id]['locations'].get(tile_indices, False):
                            if self.instances[adjacent_id]['locations'][tile_indices].get(location, False):
                                existing_mask = self.instances[adjacent_id]['locations'][tile_indices][location]
                                temp_mask = np.concatenate((existing_mask, merged_mask_partial))
                                temp_mask = np.unique(temp_mask, axis=0)
                                temp_mask = tuple(map(tuple, temp_mask))
                                self.instances[adjacent_id]['locations'][tile_indices][location] = temp_mask
                            else:
                                self.instances[adjacent_id]['locations'][tile_indices] = {location: merged_mask_partial}
                        else:
                            self.instances[adjacent_id]['locations'][tile_indices] = {}
                            for loc, mask_partial in instance['locations'][tile_indices].items():
                                if loc == location:
                                    self.instances[adjacent_id]['locations'][tile_indices][location] = merged_mask_partial
                                else:
                                    self.instances[adjacent_id]['locations'][tile_indices][loc] = mask_partial

                        # tile is also expanded as instance merge
                        if not self.tiles.get(tile_indices, False):
                            empty_locations = {'left':[], 'right':[], 'top':[], 'bottom':[], 'middle':[], 'top-left':[], 'top-right':[], 'bottom-left':[], 'bottom-right':[]}
                            self.tiles[tile_indices] = empty_locations
                        for location in instance['locations'][tile_indices].keys():
                            self.tiles[tile_indices][location].append(adjacent_id)
                            
                        return adjacent_id   
        
            return -1

    def clean_twin_instances(self):
        
        for twins in self.twins:
            twins.sort()
            for i in twins[1:]:
                self.merge_twins(twins[0], i)
        twins = np.unique(np.array([item for twin in self.twins for item in twin.flatten()]))
        
        twins_sorted = np.sort(twins, kind='quicksort')[::-1]
        for id in twins_sorted:
            rows_to_delete = self.gdf[self.gdf['instance'] == id].index
            self.gdf.drop(rows_to_delete, inplace=True)
            
        
            
    
    def merge_twins(self, id1, id2):
        idx1 = self.gdf.index[self.gdf['instance'] == id1]
        idx2 = self.gdf.index[self.gdf['instance'] == id2]
        merge_poly = unary_union([self.gdf.loc[idx1, 'geometry'], self.gdf.loc[idx2, 'geometry']])
        id1_list = self.gdf.loc[idx1, 'id'].values[0]
        id2_list = self.gdf.loc[idx2, 'id'].values[0]
        merge_id = id1_list + ',' + id2_list
        score1 = self.gdf.loc[idx1, 'score'].values[0]
        score2 = self.gdf.loc[idx2, 'score'].values[0]
        merge_score = (score1 + score2)/2.
        
        new_gdf = gpd.GeoDataFrame(pd.DataFrame({'instance': len(self.gdf), 'score': [merge_score], 'id': merge_id}), 
                                    crs=self.crs, 
                                    geometry=[merge_poly])
        
        self.gdf = pd.concat([self.gdf, new_gdf], ignore_index=True)
        
        

    def _mask_IoU(self, mask1, mask2):
        if mask2 == None:
            return -1
        mask_hash1 = [hash(mask_indices) for mask_indices in mask1]
        mask_hash2 = [hash(mask_indices) for mask_indices in mask2]
        intersection = np.count_nonzero(np.in1d(mask_hash1, mask_hash2, assume_unique=True))
        union = np.unique(mask_hash1 + mask_hash2).shape[0]
        if union == 0:
            return 0
        else:
            return intersection / union

    def _mask_intersection_hash_method(self, mask1, mask2):
        mask_hash1 = [hash(mask_indices) for mask_indices in mask1]
        mask_hash2 = [hash(mask_indices) for mask_indices in mask2]
        return -np.count_nonzero(np.in1d(mask_hash1, mask_hash2, assume_unique=True)) * self.mask_h_size * self.mask_v_size
    
    def _mask_intersection_brute_force_method(self, mask1, mask2):
        overlap = len([px for px in mask1 if px in mask2])
        return -overlap * self.mask_h_size * self.mask_v_size
    
        
    def _convert_global_mask(self, mask, tile_indices):
        x = int(tile_indices[0] * (self.w - self.overlap_float))
        y = int(tile_indices[1] * (self.h - self.overlap_float) + self.h)
        return tuple([(x+i, y-j) for j,i in np.asarray(np.nonzero(mask)).transpose()])
    
    def _get_global_bbox(self, mask):
        xmin, ymin = np.asarray(mask).min(axis=0)
        xmax, ymax = np.asarray(mask).max(axis=0)
        return [xmin, ymin, xmax, ymax]
    
    def _bbox_intersection(self, bbox1, bbox2):
        (xmin_a, ymin_a, xmax_a, ymax_a) = bbox1
        (xmin_b, ymin_b, xmax_b, ymax_b) = bbox2
        if xmin_a <= xmax_b <= xmax_a and (ymin_a <= ymax_b <= ymax_a or ymin_a <= ymin_b <= ymax_a):
            return True
        elif xmin_a <= xmin_b <= xmax_a and (ymin_a <= ymax_b <= ymax_a or ymin_a <= ymin_b <= ymax_a):
            return True
        elif xmin_b <= xmax_a <= xmax_b and (ymin_b <= ymax_a <= ymax_b or ymin_b <= ymin_a <= ymax_b):
            return True
        elif xmin_b <= xmin_a <= xmax_b and (ymin_b <= ymax_a <= ymax_b or ymin_b <= ymin_a <= ymax_b):
            return True
        else:
            return False
        
    def _get_instance(self, tile_file):
        if self.unzip:
            with zipfile.ZipFile(tile_file, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(tile_file))
            
            tile_file = tile_file.split('.')[0] + ".pickle"
        
        with open(tile_file, 'rb') as handle:
            tile_data = pickle.load(handle)
        
        if self.unzip:   
            os.remove(tile_file)
        return tile_data
    
    def _get_locations(self, contour, local_mask, tile_indices):
        x1, y1 = np.min(contour, axis=0)
        x2, y2 = np.max(contour, axis=0)
        locations = {}
        if x1 < self.overlap:
            mask = local_mask.copy()
            mask[:, self.overlap:] = 0
            mask_global = self._convert_global_mask(mask, tile_indices)
            locations['left'] = mask_global
        if x2 > self.h - self.overlap:
            mask = local_mask.copy()
            mask[:, :-self.overlap] = 0
            mask_global = self._convert_global_mask(mask, tile_indices)
            locations['right'] = mask_global
        if y1 < self.overlap:
            mask = local_mask.copy()
            mask[self.overlap:,:] = 0
            mask_global = self._convert_global_mask(mask, tile_indices)
            locations['top'] = mask_global
        if y2 > self.w - self.overlap:
            mask = local_mask.copy()
            mask[:-self.overlap, :] = 0
            mask_global = self._convert_global_mask(mask, tile_indices)
            locations['bottom'] = mask_global
        if len(locations) == 0:
            mask = local_mask.copy()
            mask_global = self._convert_global_mask(mask, tile_indices)
            locations['middle'] = mask_global
        
        location_list = locations.keys()
        if ('left' in location_list) & ('top' in location_list):
            mask = local_mask.copy()
            mask[:, self.overlap:] = 0
            mask[self.overlap:,:] = 0
            mask_global = self._convert_global_mask(mask, tile_indices)
            locations['top-left'] = mask_global

        if ('left' in location_list) & ('bottom' in location_list):
            mask = local_mask.copy()
            mask[:, self.overlap:] = 0
            mask[:-self.overlap, :] = 0
            mask_global = self._convert_global_mask(mask, tile_indices)
            locations['bottom-left'] = mask_global
            
        if ('right' in location_list) & ('top' in location_list):
            mask = local_mask.copy()
            mask[:, :-self.overlap] = 0
            mask[self.overlap:,:] = 0
            mask_global = self._convert_global_mask(mask, tile_indices)
            locations['top-right'] = mask_global

        if ('right' in location_list) & ('bottom' in location_list):
            mask = local_mask.copy()
            mask[:, :-self.overlap] = 0
            mask[:-self.overlap, :] = 0
            mask_global = self._convert_global_mask(mask, tile_indices)
            locations['bottom-right'] = mask_global

        return {tile_indices: locations}
        
        
    def _convert_mask_to_poly(self, instance):
        mask = np.asarray(instance['global_mask'])
        bbox = instance['global_bbox']
        bottom_left = np.asarray(mask).min(axis=0)
        local_mask = mask - bottom_left
        #print(local_mask.min(axis=0))
        #print(local_mask.max(axis=0))
        mask_shape = (int(bbox[2] - bbox[0]) + 1, int(bbox[3] - bbox[1]) + 1)  
        local_mask = self._create_bool_mask(local_mask, mask_shape)
        contours, _ = cv2.findContours(local_mask.astype(np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if len(contours) == 1:
            contour = np.asarray(contours[0]) + bottom_left
            contour = contour.reshape(-1, 2).tolist()
            coords = [(self.h_start + pixel[0]*self.mask_h_size, 
                    self.v_start - pixel[1]*self.mask_v_size) 
                    for pixel in contour]
            coords = np.asarray(coords)
            poly = Polygon(zip(coords[:, 0].tolist(), coords[:, 1].tolist()))
            return poly
        else:
            max_area = 0
            max_contour = None

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    max_contour = contour
            contour = np.asarray(max_contour) + bottom_left
            contour = contour.reshape(-1, 2).tolist()
            coords = [(self.h_start + pixel[0]*self.mask_h_size, 
                    self.v_start - pixel[1]*self.mask_v_size) 
                    for pixel in contour]
            coords = np.asarray(coords)
            poly = Polygon(zip(coords[:, 0].tolist(), coords[:, 1].tolist()))
            return poly

        
    def _create_bool_mask(self, mask, size):
        """
        :param mask: mask by index
        :param size: size of image
        :return: bool mask
        """
        mask_ = np.zeros(size)
        mask = mask.tolist()
        for x, y in mask:
            #if (x < size[0]) & (y < size[1]):
            mask_[int(x), int(y)] = 1
        return mask_.transpose() 
    
    
    def _get_coords_from_file_name(self, f):
        coords = f.split('.')[0].split('/')[-1].split('_')
        return (int(coords[0]), int(coords[1]))
    
    
if __name__ == "__main__":
    instance_dir = "/root/instance_mosaicking/instance_mosaicking/mask_rcnn/data/rock"
    save_shp_file = "/root/instance_mosaicking/instance_mosaicking/mask_rcnn/data/rocks_merge.shp"
    tif_map = "/root/instance_mosaicking/instance_mosaicking/mask_rcnn/data/0_0.tif"
    iou_threshold = 0.5

    tile_length = 10
    overlap = 1
    ellipse_max = 1.0
    pixel_size = 0.00728542 
    detection_threshold = 0.55
    segmentation_threshold = 0.5
    
    ir = Instance_Registration(instance_dir, 
                            save_shp_file,
                            tif_width_pixel=tile_length/pixel_size,
                            tif_height_pixel=tile_length/pixel_size,
                            tif_width_res=pixel_size,
                            tif_height_res=-pixel_size,
                            tif_map_file=tif_map,
                            tile_overlap_ratio=overlap/tile_length,
                            detection_threshold=detection_threshold, 
                            segmentation_threshold=segmentation_threshold, 
                            iou_threshold=iou_threshold, 
                            disable_merge=False, 
                            test=False,
                            unzip=True)
    
    updated_tile_files, timestamps = ir.start_registration()
    ir.combine_shapefiles()