import numpy as np
import cv2

# Ref: gibson2/render/viewer.py
class MyViewer:
    def __init__(self, depth=False):
        self.depth = depth
        self.isopen = False

        # auto window size
        cv2.namedWindow('RobotView')


    def imshow(self, image):
        '''
        if self.depth:
            image = np.dstack([image.astype(np.uint8)] * 3)
        '''
        cv2.imshow('RobotView', image)
        self.isopen = True
        q = cv2.waitKey(1)

        if q == 27:
            exit()
        
    def close(self):
        if self.isopen:
            cv2.destroyAllWindows()
            self.isopen = False
            