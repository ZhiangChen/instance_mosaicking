import rasterio
import geopandas as gpd
import rioxarray
from rasterio.plot import show
from pyproj import CRS
import os
import numpy as np
import math
import fiona
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import shapely
from fiona.crs import CRS
from rasterio.features import rasterize
import cv2
from PIL import Image
import pickle
import random
import colorsys
import time
import zipfile
from tqdm import tqdm
import tracemalloc

class Tile_Splitter(object):
    def __init__(self, shapefile_path, save_dir, crs, area_x1=0, area_y1=0, area_x2=30, area_y2=30, tile_size=10, overlap=1, tif_path=None, keep_instance_tif=True):
        """Split an annotation map to annotation tiles (shapefiles). It provides an option to split the corresponding tif map. 

        Args:
            shapefile_path (string): path of the shapefile to be split
            save_dir (string): path of the split shapefiles to be saved
            crs (_type_): coordinate reference system (e.g., from fiona.crs import from_epsg)
            area_x1 (float): study area coordinate x1 in meters. Not required if tif_path is not None. (x1, y1) are small coordinates; (x2, y2) are large coordinates. 
            area_y1 (float): study area coordinate y1 in meters.
            area_x2 (float): study area coordinate x2 in meters. (x2, y2) are used to intially decide the study area. The actual study are will also consider the tile_size in the last tiles.
            area_y2 (float): study area coordinate y2 in meters.
            tile_size (float): in meters
            overlap (float): overlap between tiles in meters.
            tif_path (string or None): path of the tif file to be split. None if only shapefile is to be split.
            keep_instance_tif (bool): it will only save tif files where there are instances if True. Otherwise, it saves all tif tiles. It works only when tif_path is not None. 
        """
        
        if tif_path is not None:
          assert os.path.isfile(tif_path)
        assert os.path.isfile(shapefile_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.tile_dir = save_dir
        self.tile_size = tile_size
        if tif_path is not None:
          self.tif = rioxarray.open_rasterio(tif_path)
          self.x1, self.y1, self.x2, self.y2 = self.tif.rio.bounds()
          self.crs = self.tif.rio.crs
        else:
          self.tif = None
          self.x1, self.y1, self.x2, self.y2 = area_x1, area_y1, area_x2, area_y2
          self.crs = crs
        self.instances_shp = gpd.read_file(shapefile_path).explode(index_parts=True)
        self.tiles = {}
        self.overlap = overlap
        self.I_max = int((self.x2 - self.x1) / (self.tile_size - self.overlap))
        self.J_max = int((self.y2 - self.y1) / (self.tile_size - self.overlap))
        self.keep_instance_tif = keep_instance_tif
        self.time_file = shapefile_path.split('/')[-1][:-4]+"_time_space.npy"

    def split(self):
        times = []
        memories = []
        start_t = time.time()
        tracemalloc.start()
        print("annotation splitting ... ")
        for id, instance in enumerate(self.instances_shp['geometry']):
            bbox = instance.bounds
            tile_indices = self._calculate_tile_indices(bbox)
            # instance is within one tile
            if tile_indices.shape[0] == 1:
                tile_index = tile_indices[0]
                if ((tile_index[0] == 0) | (tile_index[0] == self.I_max)) | ((tile_index[1] == 0) | (tile_index[1] == self.J_max)):
                    tile_poly = self._calculate_tile_polygon(tile_index)
                    instance_poly = self._extract_instance_polygon(instance)
                    poly = tile_poly.intersection(instance_poly)
                    self._add_instance(poly, tile_index, str(id))   
                else:
                    instance_poly = self._extract_instance_polygon(instance)
                    self._add_instance(instance_poly, tile_indices[0], str(id))
                times.append(time.time() - start_t)
                current, peak = tracemalloc.get_traced_memory() 
                memories.append(current / 10**6)
                continue
            # instance is on multiple tiles
            for tile_index in tile_indices:
                tile_poly = self._calculate_tile_polygon(tile_index)
                instance_poly = self._extract_instance_polygon(instance)
                if not tile_poly.intersects(instance_poly):
                    times.append(time.time() - start_t)
                    current, peak = tracemalloc.get_traced_memory() 
                    memories.append(current / 10**6)
                    continue
                poly = tile_poly.intersection(instance_poly)
                self._add_instance(poly, tile_index, str(id))
            times.append(time.time() - start_t)
            current, peak = tracemalloc.get_traced_memory() 
            memories.append(current / 10**6)
            
        print("annotation splitting done ")
        tracemalloc.stop()

        self._create_tile_shapefiles()
        if self.tif is not None:
            self._clip_tiff_tiles()
        
        if len(times) > 1:
            np.save(os.path.join(self.tile_dir, self.time_file), np.asarray((times, memories)))
    
    def _clip_tiff_tiles(self):
        print("save tiff tiles: ")
        if self.keep_instance_tif:
            for tile_index in tqdm(self.tiles):
                i, j = tile_index
                x1 = self.x1 + i*(self.tile_size - self.overlap)
                x2 = self.x1 + i*(self.tile_size - self.overlap) + self.tile_size
                y1 = self.y1 + j*(self.tile_size - self.overlap)
                y2 = self.y1 + j*(self.tile_size - self.overlap) + self.tile_size
                clipped_tif = self.tif.rio.clip_box(
                    minx=x1,
                    miny=y1,
                    maxx=x2,
                    maxy=y2,
                )
                f = os.path.join(self.tile_dir, "{x}_{y}.tif".format(x=tile_index[0], y=tile_index[1]))
                clipped_tif.rio.to_raster(f)
                #print(f)
        else:
            coords = np.array(list(self.tiles.keys()))
          
            X = np.max(coords[:, 0])
            Y = np.max(coords[:, 1])
            for i in tqdm(range(X)):
                for j in range(Y):
                    x1 = self.x1 + i*(self.tile_size - self.overlap)
                    x2 = self.x1 + i*(self.tile_size - self.overlap) + self.tile_size
                    y1 = self.y1 + j*(self.tile_size - self.overlap)
                    y2 = self.y1 + j*(self.tile_size - self.overlap) + self.tile_size
                    clipped_tif = self.tif.rio.clip_box(
                        minx=x1,
                        miny=y1,
                        maxx=x2,
                        maxy=y2,
                    )
                    f = os.path.join(self.tile_dir, "{x}_{y}.tif".format(x=i, y=j))
                    clipped_tif.rio.to_raster(f)
                    #print(f)

    def _create_tile_shapefiles(self):
        print("save tile shapefiles: ")
        for tile_index in tqdm(self.tiles):
            instance_polys = self.tiles[tile_index]
            schema = {'geometry':'Polygon', 'properties':[('id','str')]}
            f = os.path.join(self.tile_dir, "{x}_{y}.shp".format(x=tile_index[0], y=tile_index[1]))
            polyShp = fiona.open(f, mode='w', driver='ESRI Shapefile', schema = schema, crs = self.crs)
            for instance_poly, id in instance_polys:
                if isinstance(instance_poly, shapely.geometry.multipolygon.MultiPolygon):
                    instance_poly = list(instance_poly.geoms)
                    for ins_poly in instance_poly:
                        x,y = ins_poly.exterior.coords.xy
                        xy = np.asarray((x,y)).transpose()
                        rowDict = {'geometry' : {'type':'Polygon', 'coordinates': [xy]}, 'properties': {'id': id}}
                        polyShp.write(rowDict)
                else:
                    x,y = instance_poly.exterior.coords.xy
                    xy = np.asarray((x,y)).transpose()
                    rowDict = {'geometry' : {'type':'Polygon', 'coordinates': [xy]}, 'properties': {'id': id}}
                    polyShp.write(rowDict)

            polyShp.close()
            
    def _add_instance(self, instance_poly, tile_index, id):
        tile_index = tuple(tile_index)
        result = self.tiles.get(tile_index, False)
        if not result:
            self.tiles[tile_index] = [(instance_poly, id)]
        else:
            self.tiles[tile_index].append((instance_poly, id))

    def _calculate_tile_indices(self, bbox):
        x1, y1, x2, y2 = bbox
        pt0 = (x1, y1)
        pt1 = (x1, y2)
        pt2 = (x2, y1)
        pt3 = (x2, y2)
        pts = (pt0, pt1, pt2, pt3)
        tile_index = []
        # find the ranges of i and j
        i_min = 1e100
        i_max = -1
        j_min = 1e100
        j_max = -1
        for pt in pts:
            i = int((pt[0] - self.x1)/(self.tile_size - self.overlap))
            if i > i_max:
                i_max = i
            if ((pt[0] - self.x1) % (self.tile_size - self.overlap)) < self.overlap:
              i = i - 1
            if i < i_min:
                i_min = i

            j = int((pt[1] - self.y1)/(self.tile_size - self.overlap))
            if j > j_max:
                j_max = j
            if ((pt[1] - self.y1) % (self.tile_size - self.overlap)) < self.overlap:
              j = j - 1
            if j < j_min:
                j_min = j

        if i_min < 0: i_min = 0
        if i_max > self.I_max: i_max = self.I_max
        if j_min < 0: j_min = 0
        if j_max > self.J_max: j_max = self.J_max

        for i in range(i_min, i_max+1):
            for j in range(j_min, j_max+1):
                tile_index.append((i, j))

        tile_index = np.asarray(tile_index)
        return np.unique(tile_index, axis=0)

    def _calculate_tile_polygon(self, tile_index):
        i, j = tile_index
        pt0 = (self.x1 + i*(self.tile_size-self.overlap), self.y1 + j*(self.tile_size-self.overlap))
        pt1 = (self.x1 + i*(self.tile_size-self.overlap) + self.tile_size, self.y1 + j*(self.tile_size-self.overlap))
        pt2 = (self.x1 + i*(self.tile_size-self.overlap) + self.tile_size, self.y1 + j*(self.tile_size-self.overlap) + self.tile_size)
        pt3 = (self.x1 + i*(self.tile_size-self.overlap), self.y1 + j*(self.tile_size-self.overlap) + self.tile_size)
        return Polygon((pt0, pt1, pt2, pt3))

    def _extract_instance_polygon(self, instance):
        XYZ = []
        for xyz in instance.boundary.coords:
            XYZ.append(xyz)

        XYZ = np.asarray(XYZ)
        XY = XYZ[:, :2]
        return Polygon(XY)

    def _initiate_tile_shapefiles(self):
        x_N = math.ceil((self.x2 - self.x1)/(self.tile_size - self.overlap))
        y_N = math.ceil((self.y2 - self.y1)/(self.tile_size - self.overlap))
        for i in range(x_N):
            for j in range(y_N):
                schema = {'geometry':'Polygon', 'properties':[('name','str')]}
                f = os.path.join(self.tile_dir, "{x}_{y}.shp".format(x=i, y=j))
                polyShp = fiona.open(f, mode='w', driver='ESRI Shapefile', schema = schema, crs = self.crs)
                
                pt0 = (self.x1 + i*(self.tile_size-self.overlap), self.y1 + j*(self.tile_size-self.overlap))
                pt1 = (self.x1 + i*(self.tile_size-self.overlap) + self.tile_size, self.y1 + j*(self.tile_size-self.overlap))
                pt2 = (self.x1 + i*(self.tile_size-self.overlap) + self.tile_size, self.y1 + j*(self.tile_size-self.overlap) + self.tile_size)
                pt3 = (self.x1 + i*(self.tile_size-self.overlap), self.y1 + j*(self.tile_size-self.overlap) + self.tile_size)
                tile = [[pt0, pt1, pt2, pt3]]
                rowDict = {'geometry' : {'type':'Polygon', 'coordinates': tile}, 'properties': {'name': 'tile'}}
                polyShp.write(rowDict)

                polyShp.close()
                
                
class Dataset(object):
    def __init__(self, pixel_size, split_path, input_channel=(0,1,2), max_tile_number=100000):
        """create training dataset (pickle files) from split tiles (split shapefiles and tiff files). 

        Args:
            pixel_size (float): pixel size 
            split_path (string): directory of split tiles. it must include tif tiles and shapefile tiles.
            input_channel (tuple, optional): support multispectral selection. Defaults to (0,1,2). If the tif tiles are fake, it should be (0,). Don't forget the comma.
        """
        data_path = os.path.realpath(split_path)
        self.data_path = data_path
        self.data_files = [os.path.join(data_path, data_file) for data_file in os.listdir(split_path) if data_file.endswith('.tif')]
        coords = [self._get_coords_from_file_name(f) for f in self.data_files]
        coords = np.array(coords)
        self.X = np.max(coords[:, 0])
        self.Y = np.max(coords[:, 1])
        
        for k in range(max_tile_number):
            x_edge_file = os.path.join(data_path, '{i}_{k}.tif'.format(i=self.X, k=k))
            if os.path.exists(x_edge_file):
                break
        
        for k in range(max_tile_number):
            y_edge_file = os.path.join(data_path, '{k}_{i}.tif'.format(k=k, i=self.Y))
            if os.path.exists(y_edge_file):
                break
            
        x_edge_tif = rasterio.open(x_edge_file)
        y_edge_tif = rasterio.open(y_edge_file)
        h, w = x_edge_tif.shape
        self.x_edge_scaling = pixel_size/h
        h, w = y_edge_tif.shape
        self.y_edge_scaling = pixel_size/w

        self.pixel_size = pixel_size
        self.input_channel = input_channel

    def __getitem__(self, idx):
        data_path = self.data_files[idx]
        coords = self._get_coords_from_file_name(data_path)
        
        image = np.asarray(Image.open(data_path).resize((self.pixel_size, self.pixel_size)))

        if len(self.input_channel) != 1:
            image = image[:,:,self.input_channel]
        
        shp_path = data_path[:-3]+'shp'
        if not os.path.isfile(shp_path):
            return image, None
        
        shp = gpd.read_file(shp_path)
        masks = []
        boxes = []
        id_strs = []
        N, _ = shp.shape
        for i in range(N):
            poly = shp['geometry'][i]
            id_str = shp['id'][i]
            if poly is None:
                continue
            tif = rasterio.open(data_path)
            p = self._poly_from_utm(poly, tif.meta['transform'])
            mask = np.zeros((self.pixel_size, self.pixel_size))
            raster_tif = rasterize([p], (tif.height, tif.width))
            if (abs(tif.width - self.pixel_size) < 2) & (abs(tif.height - self.pixel_size) < 2):
                mask = cv2.resize(raster_tif, dsize=(self.pixel_size, self.pixel_size), interpolation=cv2.INTER_LINEAR)
            else:
              if (coords[0] == self.X) & (coords[1] == self.Y):
                  tmp = cv2.resize(raster_tif, (int(tif.width*self.x_edge_scaling), int(tif.height*self.y_edge_scaling)))
                  mask[-int(tif.height*self.y_edge_scaling):, : int(tif.width*self.x_edge_scaling)] = tmp
              elif coords[0] == self.X:
                  tmp = cv2.resize(raster_tif, (int(tif.width*self.x_edge_scaling), self.pixel_size))
                  mask[:self.pixel_size, : int(tif.width*self.x_edge_scaling)] = tmp
              elif coords[1] == self.Y:
                  tmp = cv2.resize(raster_tif, (self.pixel_size, int(tif.height*self.y_edge_scaling)))
                  mask[-int(tif.height*self.y_edge_scaling):, :self.pixel_size] = tmp
              else:
                  mask = cv2.resize(raster_tif, dsize=(self.pixel_size, self.pixel_size), interpolation=cv2.INTER_LINEAR)
            mask = mask > 0
            if np.count_nonzero(mask) > 0:
                pos = np.where(mask)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if (xmax - xmin > 0) & (ymax - ymin > 0):
                    masks.append(mask)
                    boxes.append([xmin, ymin, xmax, ymax])
                    id_strs.append(id_str)
        
        num_objs = len(masks)
        if num_objs > 0:
            masks = np.stack(masks)
        else:
            return image, None
   
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        image = image/255.
        image = np.moveaxis(image, 2, 0)
        boxes = np.asarray(boxes)
        obj_ids = np.ones(num_objs)
        labels = np.asarray(obj_ids, dtype=np.int64)
        masks = np.asarray(masks, dtype=np.uint8)
        image_id = np.asarray([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = np.zeros((num_objs,), dtype=np.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_name"] = data_path
        target["scores"] = np.ones((num_objs,), dtype=np.int64)
        target["ids"] = id_strs

        return image, target

    def __len__(self):
        return len(self.data_files)
    
    def _poly_from_utm(self, poly, transform):
        poly_pts = []
        for i in np.array(poly.exterior.coords):

            # Convert polygons to the image CRS
            poly_pts.append(~transform * tuple(i))

        # Generate a polygon object
        new_poly = Polygon(poly_pts)
        return new_poly
    
    def _get_coords_from_file_name(self, f):
        coords = f.split('.')[0].split('/')[-1].split('_')
        return (int(coords[0]), int(coords[1]))

    def show(self, idx):
        image, target = self.__getitem__(idx)
        if target is None:
            print("No objects in the tif")
            return

        rgb = (np.moveaxis(image, 0, -1)*255).astype(np.uint8)
        if rgb.shape[2] == 1:
            rgb = np.squeeze(rgb, axis=2)
        rgb = Image.fromarray(rgb)
        rgb.save(os.path.join(self.data_path, 'rgb.png'))
        #display(rgb)  # jupyter notebook built-in function to display PIL image
        #rgb.show()
        masks = target["masks"]
        masks = masks.max(axis=0) * 255
        masks = Image.fromarray(masks)
        masks.save(os.path.join(self.data_path, "masks.png"))
        print("image saved to " + os.path.join(self.data_path, "masks.png"))
        #display(masks) 
        #masks.show()

    def show_overlay(self, idx, random_color=False, bbox=False):
        """This currently supports RGB tif tiles.

        Args:
            idx (_type_): _description_
            random_color (bool, optional): _description_. Defaults to False.
            bbox (bool, optional): _description_. Defaults to False.
        """
        image, target = self.__getitem__(idx)
        if target is None:
            print("No objects in the tif")
            return
        image = (np.moveaxis(image, 0, -1)*255).astype(np.uint8)
        
        image1 = np.zeros_like(image)
        
        image = image.copy()
        image1 = image1.copy()
        
        boxes = target['boxes']
        masks = target["masks"]
        masks = masks>0.5
        _, _, N = masks.shape
        colors  = self.random_colors(N)
        if not random_color:
            for mask in masks:
                image = self.apply_mask(image, mask, (1,0,1))
        else:
            for i, mask in enumerate(masks):
                image = self.apply_mask(image, mask, colors[i])
        if bbox:
            for i, box in enumerate(boxes):
                if not random_color:
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), 2)
                else:
                    color = tuple([int(c*255) for c in colors[i]])
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.imwrite(os.path.join(self.data_path, 'overlap.png'), image)
        print("image saved to " + os.path.join(self.data_path, 'overlap.png'))
        if not random_color:
            for mask in masks:
                image1 = self.apply_mask(image1, mask, (1,1,1), alpha=1)
        else:
            for i, mask in enumerate(masks):
                image1 = self.apply_mask(image1, mask, colors[i])
        if bbox:
            for i, box in enumerate(boxes):
                if not random_color:
                    cv2.rectangle(image1, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 2)
                else:
                    color = tuple([int(c*255) for c in colors[i]])
                    cv2.rectangle(image1, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.imwrite(os.path.join(self.data_path, 'overlap_background.png'), image1)
        print("image saved to " + os.path.join(self.data_path, 'overlap_background.png'))
        

    def show_bbox(self, idx):
        """This currently supports RGB tif tiles.

        Args:
            idx (_type_): _description_
        """
        image, target = self.__getitem__(idx)
        if target is None:
            print("No objects in the tif")
            return
        masks = target["masks"]
        boxes = target['boxes']
        image = (np.moveaxis(image, 0, -1)*255).astype(np.uint8)
        image1 = np.zeros_like(image)
        _, _, N = masks.shape
        colors  = self.random_colors(N)
        for i, box in enumerate(boxes):
            color = tuple([int(c*255) for c in colors[i]])
            image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.imwrite(os.path.join(self.data_path, 'bbox.png'), image)
        for i, box in enumerate(boxes):
            color = tuple([int(c*255) for c in colors[i]])
            image1 = cv2.rectangle(image1, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.imwrite(os.path.join(self.data_path, 'bbox_background.png'), image1)


    def apply_mask(self, image, mask, color, alpha=0.35):
        """Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image
  
    def random_colors(self, N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors
  
    def save_pickles(self, zip=False):
        print("save pickles: ")
        for i in tqdm(range(len(self.data_files))):
            data = self.__getitem__(i)
            image, target = data
            result = {}
            result['image'] = image
            result['bb'] = target['boxes']
            result['labels'] = target['labels']
            result['scores'] = target['scores']
            result['masks'] = target['masks']
            result['image_name'] = target['image_name']
            result['ids'] = np.asarray(target['ids'])
            image_name = target['image_name']
            file_name = image_name.split('/')[-1].split('.')[0] +'.pickle'
            f = os.path.join(self.data_path, file_name)
            with open(f, 'wb') as handle:
                pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            if zip:
                zip_name = image_name.split('/')[-1].split('.')[0] +'.zip'
                zip_file = os.path.join(self.data_path, zip_name)
                with zipfile.ZipFile(zip_file, 'w', compression=zipfile.ZIP_DEFLATED) as zip_obj:
                    zip_obj.write(f, arcname=file_name)
            
                os.remove(f)
