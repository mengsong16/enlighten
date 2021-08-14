# [setup]
import math
import os

import magnum as mn
import numpy as np
from matplotlib import pyplot as plt

import habitat_sim
from habitat_sim.gfx import LightInfo, LightPositionModel, DEFAULT_LIGHTING_KEY, NO_LIGHT_KEY

from habitat_sim.utils.common import quat_from_angle_axis

import sys
import random


data_path = "/home/meng/habitat-sim/data"
dir_path = os.path.dirname(os.path.realpath(__file__))
output_path = os.path.join(dir_path, "van-gogh-room-default-lighting/")
save_index = 0


def show_img(data):
    plt.figure(figsize=(12, 12))
    plt.imshow(data, interpolation="nearest")
    plt.axis("off")

    # the plot only lasts for 1 second
    plt.show(block=False)
    #plt.show(block=True)   
    plt.pause(1)

def save_img(data):
    plt.figure(figsize=(12, 12))
    plt.imshow(data, interpolation="nearest")
    plt.axis("off")

    global save_index
    plt.savefig(
        output_path + str(save_index) + ".jpg",
        bbox_inches="tight",
        pad_inches=0,
        quality=50,
    )
    save_index += 1

# get observation image
def get_obs(sim, show, save):
    obs = sim.get_sensor_observations()["rgba_camera"]
    if show:
        show_img(obs)
    if save:
        save_img(obs) 

    return obs


def place_agent(sim):
    # place our agent in the scene
    agent_state = habitat_sim.AgentState()

    agent_state.position = [5.0, 0.0, 1.0]
    agent_state.rotation = quat_from_angle_axis(
        math.radians(70), np.array([0, 1.0, 0])
    ) * quat_from_angle_axis(math.radians(-20), np.array([1.0, 0, 0]))

    agent = sim.initialize_agent(0, agent_state)
    return agent.scene_node.transformation_matrix()

def get_state(agent):
    agent_state = agent.get_state()
    print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)    

# /home/meng/habitat-sim/data/scene_datasets/habitat-test-scenes/van-gogh-room.glb
# /home/meng/habitat-sim/data/scene_datasets/habitat-test-scenes/apartment_1.glb
# /home/meng/habitat-sim/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb
# /dataset/mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb
def make_configuration(scene_path="/home/meng/habitat-sim/data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"):
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    # scene path
    backend_cfg.scene_id = scene_path
    # enable physics
    backend_cfg.enable_physics = True
    # enable scene lighting change
    backend_cfg.override_scene_light_defaults = True

    # agent configuration
    sensor_cfg = habitat_sim.CameraSensorSpec()
    sensor_cfg.resolution = [1080, 960]
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_cfg]

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def random_agent_with_light(show=True, save=True):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # create config and set lighting in config
    cfg = make_configuration()
    cfg.sim_cfg.scene_light_setup = "current_scene_lighting"
    #cfg.sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY

    # create simulator
    sim = habitat_sim.Simulator(cfg)    

    # register lighting in simulator
    sim.set_light_setup([], "current_scene_lighting")

    # the randomness is needed when choosing the actions
    random.seed(42)
    sim.seed(42)

    # initialize agent
    agent = sim.initialize_agent(agent_id=0)
    set_robot_pose(sim, position=[2.0, 0.0, 1.0], rotation=[0,0,0])
    #set_robot_pose(sim, position=[0.0, 0.0, 0.0], rotation=[0,0,0])

    total_frames = 0
    #action_names = list(cfg.agents[0].action_space.keys())
    action_names = ['turn_right', 'turn_right']
    max_frames = 20

    while total_frames < max_frames:
        action = random.choice(action_names)
        print("step: %d, action: %s"%(total_frames, action))
        observations = sim.step(action)

        sim.set_light_setup([
            LightInfo(vector=[0.0, 0.0, -2.0, 1.0], model=LightPositionModel.Camera)
        ], "current_scene_lighting")

        #sim.set_light_setup([
        #    LightInfo(vector=[2.0, 1.0, 1.0, 1.0], model=LightPositionModel.Global)
        #], "current_scene_lighting")


        #print(observations.keys())
        rgb = observations["rgba_camera"]
        
        if show:
            show_img(rgb)
        if save:
            save_img(rgb) 

        total_frames += 1

def get_action_names(cfg):
    action_names = list(cfg.agents[0].action_space.keys())
    print(action_names)

def get_rotation_quat(rotation):
    x,y,z = rotation
    # z * y * x
    quat = quat_from_angle_axis(math.radians(z), np.array([0, 0, 1.0])
        ) * quat_from_angle_axis(math.radians(y), np.array([0, 1.0, 0])
        ) * quat_from_angle_axis(math.radians(x), np.array([1.0, 0, 0]))

    return quat

def set_robot_pose(sim, position, rotation):
    # place our agent in the scene
    agent_state = habitat_sim.AgentState()
    agent_state.position = position
    agent_state.rotation = get_rotation_quat(rotation)
    agent = sim.initialize_agent(0, agent_state)
    return agent.scene_node.transformation_matrix()

def create_new_lightsetup(sim):
    # create and register new light setup:
    my_scene_lighting_setup = [
        LightInfo(vector=[0.0, 2.0, 0.6, 0.0], model=LightPositionModel.Global)
    ]
    sim.set_light_setup(my_scene_lighting_setup, "my_scene_lighting")

def set_new_lightsetup(sim, cfg, key):
    cfg.sim_cfg.scene_light_setup = key
    sim.reconfigure(cfg)

def spin_robot():
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # create config and set lighting in config
    cfg = make_configuration()
    #cfg.sim_cfg.scene_light_setup = "current_scene_lighting"
    #cfg.sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY
    cfg.sim_cfg.scene_light_setup = habitat_sim.gfx.NO_LIGHT_KEY

    # create simulator
    sim = habitat_sim.Simulator(cfg)

    # register lighting in simulator
    sim.set_light_setup([], "current_scene_lighting")
    
    # changing lighting according to robot's pose
    rotation_list = [[0, 0, 0], [0, 90, 0], [0, 180, 0], [0, 270, 0]]
    light_vector = [[0.0, 2.0, 0.1, 0.0], [0.0, 2.0, 0.2, 0.0], [0.0, 2.0, 0.3, 0.0], [0.0, 2.0, 0.4, 0.0]]
    for i, r in enumerate(rotation_list):
        set_robot_pose(sim, position=[2.0, 0.0, 1.0], rotation=r)
        
        #sim.set_light_setup([
        #    LightInfo(vector=light_vector[i], model=LightPositionModel.Global)
        #], "current_scene_lighting")

        sim.set_light_setup([
            LightInfo(vector=[0.0, 0.0, -2.0, 1.0], model=LightPositionModel.Camera)
        ], "current_scene_lighting")

        get_obs(sim, show=True, save=True)
        print_scene_light_vector(sim, "current_scene_lighting")
        print_current_scene_light_vector(sim)
    
    # close simulator
    sim.close()

    

def print_current_scene_light_vector(sim):
    print("******************************************************************")
    if not sim.get_current_light_setup():
        print("Current scene light setup: No Light")
    else:    
        print("Current scene light setup: vector=%s"%sim.get_current_light_setup()[0].vector)
    print("******************************************************************")

def print_scene_light_vector(sim, key):
    print("******************************************************************")
    if not sim.get_light_setup(key):
        print("Scene light setup: No Light, key=%s"%(key))
    else:    
        print("Scene light setup: key=%s, vector=%s"%(key, sim.get_light_setup(key)[0].vector))
    print("******************************************************************")    

def lighting_tutorial(show_imgs=True, save_imgs=False):
    if save_imgs and not os.path.exists(output_path):
        os.mkdir(output_path)


    # case 0: default scene lighting: no light
    # create the simulator and render flat shaded scene
    cfg = make_configuration()
    sim = habitat_sim.Simulator(cfg)
    agent_transform = place_agent(sim)
    get_obs(sim, show_imgs, save_imgs)
    print_current_scene_light_vector(sim)

    # case 1: [scene swap shader]
    # close the simulator and re-initialize with DEFAULT_LIGHTING_KEY:
    sim.close()
    cfg = make_configuration()
    cfg.sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY
    sim = habitat_sim.Simulator(cfg)
    agent_transform = place_agent(sim)
    get_obs(sim, show_imgs, save_imgs)
    print_current_scene_light_vector(sim)
    
    
    # case 2: create and register new light setup [option 1]
    '''
    my_scene_lighting_setup = [
        LightInfo(vector=[0.0, 2.0, 0.6, 0.0], model=LightPositionModel.Global)
    ]
    sim.set_light_setup(my_scene_lighting_setup, "my_scene_lighting")

    # reconfigure with custom key:
    new_cfg = make_configuration()
    new_cfg.sim_cfg.scene_light_setup = "my_scene_lighting"
    sim.reconfigure(new_cfg)
    agent_transform = place_agent(sim)
    get_obs(sim, show_imgs, save_imgs)
    print_current_scene_light_vector(sim)
    '''
    # [option 2]
    sim.close()
    my_scene_lighting_setup = [
        #LightInfo(vector=[0.0, 2.0, 0.6, 0.0], model=LightPositionModel.Global)
        LightInfo(vector=[0.0, 2.0, 0.6, 1.0], model=LightPositionModel.Global)
    ]
    new_cfg = make_configuration()
    new_cfg.sim_cfg.scene_light_setup = "my_scene_lighting"
    sim = habitat_sim.Simulator(new_cfg)
    sim.set_light_setup(my_scene_lighting_setup, "my_scene_lighting")
    agent_transform = place_agent(sim)
    get_obs(sim, show_imgs, save_imgs)
    print_current_scene_light_vector(sim)
    
    
    # reset to default scene shading: no light
    sim.close()
    cfg = make_configuration()
    sim = habitat_sim.Simulator(cfg)
    agent_transform = place_agent(sim)  # noqa: F841
    #get_obs(sim, show_imgs, save_imgs)
    #print_current_scene_light_vector(sim)

    # case 3: 
    # get the rigid object attributes manager, which manages
    # templates used to create objects
    obj_template_mgr = sim.get_object_template_manager()
    # get the rigid object manager, which provides direct access to objects
    rigid_obj_mgr = sim.get_rigid_object_manager()

    # load ball and chair templates from assets
    sphere_template_id = obj_template_mgr.load_configs(
        str(os.path.join(data_path, "test_assets/objects/sphere"))
    )[0]
    chair_template_id = obj_template_mgr.load_configs(
        str(os.path.join(data_path, "test_assets/objects/chair"))
    )[0]

    # create a sphere and place it at a desired location
    obj_1 = rigid_obj_mgr.add_object_by_template_id(sphere_template_id)
    obj_1.translation = [3.2, 0.23, 0.03]
    get_obs(sim, show_imgs, save_imgs)

    # case 4
    # create a custom light setup
    my_default_lighting = [
        LightInfo(vector=[2.0, 2.0, 1.0, 1.0], model=LightPositionModel.Camera)
    ]
    # overwrite the default DEFAULT_LIGHTING_KEY light setup
    sim.set_light_setup(my_default_lighting)
    get_obs(sim, show_imgs, save_imgs)

    # case 5
    # create a chair and place it at a location with a specified orientation
    obj_2 = rigid_obj_mgr.add_object_by_template_id(chair_template_id)
    obj_2.rotation = mn.Quaternion.rotation(mn.Deg(-115), mn.Vector3.y_axis())
    obj_2.translation = [3.06, 0.47, 1.15]

    get_obs(sim, show_imgs, save_imgs)

    # case 6
    light_setup_2 = [
        LightInfo(
            vector=[2.0, 1.5, 5.0, 1.0],
            color=[0.0, 100.0, 100.0],
            model=LightPositionModel.Global,
        )
    ]
    sim.set_light_setup(light_setup_2, "my_custom_lighting")

    rigid_obj_mgr.remove_all_objects()

    # create and place 2 chairs with custom light setups
    chair_1 = rigid_obj_mgr.add_object_by_template_id(
        chair_template_id, light_setup_key="my_custom_lighting"
    )
    chair_1.rotation = mn.Quaternion.rotation(mn.Deg(-115), mn.Vector3.y_axis())
    chair_1.translation = [3.06, 0.47, 1.15]

    chair_2 = rigid_obj_mgr.add_object_by_template_id(
        chair_template_id, light_setup_key="my_custom_lighting"
    )
    chair_2.rotation = mn.Quaternion.rotation(mn.Deg(50), mn.Vector3.y_axis())
    chair_2.translation = [3.45927, 0.47, -0.624958]

    get_obs(sim, show_imgs, save_imgs)

    # case 7
    existing_light_setup = sim.get_light_setup("my_custom_lighting")

    # create a new setup with an additional light
    new_light_setup = existing_light_setup + [
        LightInfo(
            vector=[0.0, 0.0, 1.0, 1.0],
            color=[1.6, 1.6, 1.4],
            model=LightPositionModel.Camera,
        )
    ]

    # register the old setup under a new name
    sim.set_light_setup(existing_light_setup, "my_old_custom_lighting")

    # register the new setup overwriting the old one
    sim.set_light_setup(new_light_setup, "my_custom_lighting")
    get_obs(sim, show_imgs, save_imgs)


    # case 8
    chair_1.set_light_setup(habitat_sim.gfx.DEFAULT_LIGHTING_KEY)
    get_obs(sim, show_imgs, save_imgs)
    

if __name__ == "__main__":
    '''
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show-images", dest="show_images", action="store_false")
    parser.add_argument("--no-save-images", dest="save_images", action="store_false")
    parser.set_defaults(show_images=True, save_images=True)
    args = parser.parse_args()
    '''
    #lighting_tutorial(show_imgs=True, save_imgs=True)
    spin_robot()
    #random_agent_with_light()
