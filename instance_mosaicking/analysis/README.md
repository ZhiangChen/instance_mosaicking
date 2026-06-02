# Performance evaluation and time complexity analysis 
## Performance evalution
Metrics: accuracy, precision, and recall.  
Experiments: (1) Granite Dells and (2) Ellipes. 

**Ellipse experiment** examines the effects of ellipse generation parameters, such as ellipse_max, ellipse_min, density, tile_number, and iou_threshold. 


## Time complexity analysis
Time complexity of the annotation map split algorithm is O(N), where N is the number of instances in an annotation map. Time complexity of the instance registration algorithm is O(N M^2 P log(2P)), where N is the number of prediction tilee, M is the number of instances in a prediction tile, and P is the number of pixels in an instance.  


Experiment uses the synthetic ellipse data. 


## Workflow test
Apply Mask RCNN on the Granite Dells data to demonstrate the workflow of maps.