

from instance_mosaicking.maps.evaluation import evaluate

# note that the ground truth should be cropped shapefile instead of the original generated shapefile
evaluate('/root/instance_mosaicking/data/random_generation/ellipses_30_500_5_100000_1000_50000_200000/0_0.shp', 
         '/root/instance_mosaicking/data/random_generation/ellipses_30_500_5_100000_1000_50000_200000/ellipses_30_500_5_100000_1000_50000_200000_merge.shp')