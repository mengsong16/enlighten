import numpy as np
from enlighten.utils.image_utils import try_cv2_import
cv2 = try_cv2_import()
from habitat.utils.visualizations import maps
import math
import magnum as mn
import habitat_sim

# Ref: gibson2/render/viewer.py
class MyViewer:
    def __init__(self, sim):
        self.isopen = False

        # auto window size
        cv2.namedWindow('RobotView')

        self.sim = sim
        if self.sim.pathfinder.is_loaded:
            cv2.namedWindow('MapView')

    def imshow(self, image, map):
        cv2.imshow('RobotView', image)
        self.isopen = True

        # To get keyboard response, must after imshow in the first window, and click robotview window to make it active
        q = cv2.waitKey(1)

        if q == 27:
            exit()

        if map is not None:    
            cv2.imshow('MapView', map)
            #print("*******************")
            #print("map shape: " + str(map.shape))
            #print("*******************")

        
    def close(self):
        if self.isopen:
            cv2.destroyAllWindows()
            self.isopen = False
            