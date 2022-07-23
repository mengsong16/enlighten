import numpy as np
from enlighten.utils.image_utils import try_cv2_import
cv2 = try_cv2_import()
from habitat.utils.visualizations import maps
import math
import magnum as mn
import habitat_sim
from PIL import Image

# Ref: gibson2/render/viewer.py
class MyViewer:
    def __init__(self, sim, show_attention=False):
        self.isopen = False

        # auto window size
        cv2.namedWindow('RobotView')

        self.sim = sim
        if self.sim.pathfinder.is_loaded:
            cv2.namedWindow('MapView')

        if show_attention:
            cv2.namedWindow('AttentionView')    

    def imshow(self, image, map, attention_image=None):
        cv2.imshow('RobotView', image)
        #image.show()
        self.isopen = True

        # To get keyboard response, must after imshow in the first window, and click robotview window to make it active
        # wait for 1ms
        q = cv2.waitKey(1)

        #if ESC is pressed, close window
        if q == 27:
            exit()

        if map is not None:    
            cv2.imshow('MapView', map)
            #print("*******************")
            #print("map shape: " + str(map.shape))
            #print("*******************")
        # [H,W,C] numpy
        if attention_image is not None:
            # blend attention_image and image
            #print(attention_image.shape)
            #print(image.shape)
            #print(type(attention_image))
            #print(type(image))

            # print("=================")
            # print(np.amax(attention_image))
            # print(np.amin(attention_image))
            # print("=================")

            # [H,W,C]
            org_image = Image.fromarray(np.uint8(image), mode="RGB")
            # print("=================")
            # print(np.amax(org_image))
            # print(np.amin(org_image))
            # print("=================")

            # 0 represents black, 1 or 255 represents white
            alpha = Image.fromarray(np.squeeze(np.uint8(attention_image*255), axis=2), mode="L")
            # just repeat the gray channel for three times
            alpha = alpha.convert("RGB")
            #alpha_array = np.asarray(alpha)
            #assert np.array_equal(alpha_array[:,:,0], alpha_array[:,:,1])
            #print(org_image.size)
            #print(alpha.size)

            # PIL.Image.blend(im1, im2, alpha)
            # out = image1 * (1.0 - alpha) + image2 * alpha
            # If alpha is 0.0, a copy of the first image is returned. 
            alpha_blend_image = Image.blend(org_image, alpha, alpha=0.8)
            alpha_blend_image = np.asarray(alpha_blend_image)
            #print(np.asarray(alpha).shape)
            #print(alpha_blend_image.shape)
            cv2.imshow('AttentionView', alpha_blend_image)  
            #cv2.imshow('AttentionView', np.asarray(alpha))   
            
            return alpha_blend_image
        else:
            return None    

        
    def close(self):
        if self.isopen:
            cv2.destroyAllWindows()
            self.isopen = False
            