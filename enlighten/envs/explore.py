import attr
import numpy as np
from gym import spaces
import math

class State_Visitation:
    def __init__(self, position_resolution, rotation_resolution):
        # meter
        self.position_resolution = position_resolution
        # pi
        self.rotation_resolution = rotation_resolution * math.pi / 180.0

        #print(self.position_resolution)
        #print(self.rotation_resolution)
        # print(math.pi / self.rotation_resolution)
        # print(-math.pi / self.rotation_resolution)
        # print((-math.pi) % (2*math.pi) / self.rotation_resolution)
        # print(0 / self.rotation_resolution)
        # print(2*math.pi / self.rotation_resolution)
        # print((2*math.pi) % (2*math.pi) / self.rotation_resolution)
        # print(0.14 % (2*math.pi))
        #exit()

        # reset dictionary
        self.reset()
    
    # state: [pos, angle] numpy array: (6,)
    # rotation: radians, already normalized to [-pi, pi], note that -pi and pi are different
    def state_to_key(self, state):
        #s = state.view(-1)
        #state_key = tuple(state.view(-1).tolist())
        downsample_position = np.rint(state[:3] / float(self.position_resolution)).astype(int)
        #normalized_angle = (state[3:] % (2*math.pi)) - math.pi
        # print('~~~~~~~~~~~~~~~~~~~~~~~~~')
        # print(state[3:])
        # print(normalized_angle)
        # print('~~~~~~~~~~~~~~~~~~~~~~~~~')
        downsample_rotation = np.rint(state[3:] / float(self.rotation_resolution)).astype(int)
        state_array = np.concatenate((downsample_position, downsample_rotation), axis=0)

        state_key = tuple((state_array.tolist()))
        return state_key

    def reset(self):
        self.state_count_dict = dict()  

    # state: [pos, angle] numpy array: (6,)
    def add(self, state):
        state_key = self.state_to_key(state)
        if state_key in self.state_count_dict:
            self.state_count_dict[state_key] += 1
        else:
            self.state_count_dict.update({state_key: 1})
    
    def get(self, state):
        state_key = self.state_to_key(state)
        if state_key in self.state_count_dict.keys():
            return self.state_count_dict.get(state_key)
        else:
            return 0    

    def print(self):
        print("Total: %d"%len(self.state_count_dict.keys()))
        # for key, value in self.state_count_dict.items():
        #     print(str(key)+": "+str(value))

        