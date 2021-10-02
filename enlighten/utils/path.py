import yaml
import os
import logging
import numpy as np

#wd = os.getcwd()
cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.join(cur_path[:cur_path.find("/enlighten")], "enlighten") 
package_path = os.path.join(root_path, "enlighten")
config_path = os.path.join(root_path, "configs")
video_path = os.path.join(root_path, "video")	

home_path = os.path.expanduser('~')
data_path = os.path.join(home_path, "habitat-sim", "data")

    	

if __name__ == "__main__": 
    print(cur_path)
    print(root_path)
    print(config_path)
    print(home_path)
    print(data_path)