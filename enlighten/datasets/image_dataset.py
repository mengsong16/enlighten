import random
import numpy as np
import copy
import torch
from torch.utils.data import Dataset as TorchDataset
from enlighten.agents.common.other import get_obs_channel_num
import pickle
from enlighten.agents.common.other import get_device
from enlighten.utils.config_utils import parse_config
from enlighten.utils.path import *
from enlighten.agents.common.seed import set_seed_except_env_seed

class ImageDataset:
    """ Sample images for domain adaptation 
    """
    def __init__(self, config, device=None):
        self.config = config  # config is a dictionary
        if device is None:
            self.device = get_device(self.config)
        else:    
            self.device = device
        
        self.load_images()
        

    def load_images(self):
        # load all images from the training dataset
        dataset_path = self.config.get("image_dataset_path")
        dataset_path = os.path.join(dataset_path, "train_data.pickle")
        print("Loading images from %s"%(dataset_path))
        with open(dataset_path, 'rb') as f:
            self.images = pickle.load(f)

        self.num_images = 0
        for image_set in self.images.values():
            self.num_images += len(image_set)

        print("Loaded %d training images"%(self.num_images))

    # sample a batch of images (T,C,H,W), where T is the total number of steps in the batch
    # T=B*K when trajectories have the same regular shape
    def get_image_batch(self, batch_shape):
        traj_num = batch_shape.shape[0]
        # all scenes
        scenes = list(self.images.keys())
        # sample traj_num scenes without replacement
        selected_scenes = random.sample(scenes, k=traj_num)
       
        # for each scene, sample a trajectory
        image_batch = []
        for i, scene_id in enumerate(selected_scenes):
            # all images from current scene
            scene_images = self.images[scene_id]
            # sample batch_shape[i] images 
            # selected_images is a list of (C,H,W) images 
            if batch_shape[i] <= len(scene_images):
                # without replacement
                selected_images = random.sample(scene_images, k=batch_shape[i])
            else:
                # with replacement
                selected_images = random.choices(scene_images, k=batch_shape[i])

            image_batch.extend(selected_images)
            
        # Concate elements in the list and convert numpy to torch tensor
        # [(C,H,W)] --> (L,C,H,W)
        #obs_seg = np.expand_dims(np.stack(selected_images), axis=0)
        o = torch.from_numpy(np.stack(image_batch, axis=0)).to(dtype=torch.float32, device=self.device)
        
        return o 

    
if __name__ == "__main__":
    set_seed_except_env_seed(seed=1)
    config_file = os.path.join(config_path, "imitation_learning_rnn.yaml")
    config = parse_config(config_file)
    dataset = ImageDataset(config)
    for i in range(10):
        dataset.get_image_batch(batch_shape=np.array([100,20,21]))

        print("Batch %d Done"%(i+1))
        break