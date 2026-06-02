import numpy as np
import os

from instance_mosaicking.maps.workflow import Workflow



def create_config(param, save_dir):
    config = {'density':int(param[0]), 
              'tile_length':param[1], 
              'tile_number':int(param[2]), 
              'overlap':param[3], 
              'pixel_size': param[4], # pixel size shouldn't be greater than ellipse_min
              'ellipse_min':param[5], 
              'ellipse_max':param[6], 
              'iou_threshold': param[7],
              'zip':True,
              'save_dir':save_dir}
    return config   


def format_params(params, save_dir="/root/instance_mosaicking/data/random_generation"):
    config_params = [params[:, 0]]
    for i in range(7):
        for j in range(1, 3):
            param = params[:, 0].copy()
            param[i] = params[i, j]
            config_params.append(param)
            
    config_list = [create_config(param, save_dir) for param in config_params]
    return config_list

def automate_performance_evaluation(save_dir):
    density = [30, 100, 200]
    tile_length = [1000, 500, 2000]
    tile_number = [5, 20, 40]
    overlap = [100, 50, 200]
    pixel_size = [1, 0.5, 2]
    ellipse_min = [50, 20, 100]
    ellipse_max = [200, 400, 800]
    iou_threshold = [0.75, 0.55, 0.65]

    params = np.asarray((density, tile_length, tile_number, overlap, pixel_size, ellipse_min, ellipse_max, iou_threshold))

    config_list = format_params(params, save_dir)
    for config in config_list:
        print(config)
        #Workflow(config)
        #"""
        try:
            w = Workflow(config)
            del w
        except:
            print("Error")
        #"""
        
def automate_time_analysis_M(save_dir):
    param = [20, 1000, 100, 100, 1, 50, 200, 0.75]    
    config = create_config(param, save_dir)
    try:
        w = Workflow(config)
        del w
    except:
        print("Error")

def automate_time_analysis_K(k, save_dir):
    param = [20, 1000, 5, 100, 1, 50, 200, 0.75]
    param[0] = k    
    config = create_config(param, save_dir)
    print(config)
    try:
        w = Workflow(config)
        del w
    except:
        print("Error")
        
def automate_time_analysis_P(p, save_dir):
    param = [10, 1000, 5, 100, 1, 50, 200, 0.75]
    param[6] = p    
    config = create_config(param, save_dir)
    print(config)
    try:
        w = Workflow(config)
        del w
    except:
        print("Error-"*10)
        
    

if __name__ == "__main__":
    for i in range(10):
        save_dir = "/root/instance_mosaicking/data/time_complexity_K_" + str(i)
        for k in [10, 50, 100, 150, 200, 250, 300]:
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            automate_time_analysis_K(k, save_dir)
    
    for i in range(10):
        save_dir = "/root/instance_mosaicking/data/time_complexity_P_" + str(i)
        for p in range(100,1100,100):
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            automate_time_analysis_P(p, save_dir)
    
    for i in range(10):
        save_dir = "/root/instance_mosaicking/data/time_complexity_M_" + str(i)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        automate_time_analysis_M(save_dir)
    
    
    
    