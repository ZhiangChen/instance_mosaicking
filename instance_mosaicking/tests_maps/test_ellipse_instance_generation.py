

import instance_mosaicking.maps.ellipse_instance_generation as test

test.spawn_ellipses(tile_length=10, ellipse_length_range=(0.3, 1.0), density=100, tile_number=3, overlap=1, shp_path='/root/instance_mosaicking/data/random_generation', generate_tiff=True)

