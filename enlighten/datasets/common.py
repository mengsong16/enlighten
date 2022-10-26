# behavior_dataset_path: "/dataset/behavior_dataset_gibson"
import habitat_sim
import os
import numpy as np

import pickle

def load_behavior_dataset_meta(behavior_dataset_path, split_name):

    behavior_dataset_meta_data_path = os.path.join(behavior_dataset_path, "meta_data")
    behavior_dataset_meta_split_path = os.path.join(behavior_dataset_meta_data_path, '%s.pickle'%(split_name))

    if not os.path.exists(behavior_dataset_meta_split_path):
        print("Error: path does not exist: %s"%(behavior_dataset_meta_split_path))
        exit()
    
    episode_list = pickle.load(open(behavior_dataset_meta_split_path, "rb" ))

    print("Behavior data split: %s"%split_name)
    print("Loaded %d episodes"%(len(episode_list)))
    
    return episode_list


# plan the optimal action sequence path from the current state
def get_optimal_path(env):
    try:
        # No need to reset or recreate the path follower before path planning
        # once create the path follower attaching to an agent 
        # it will always update itself to the current state when find_path is called
        #env.create_shortest_path_follower()
        #env.follower.reset()
        optimal_action_seq = env.follower.find_path(goal_pos=env.goal_position)
        
        assert len(optimal_action_seq) > 0, "Error: optimal action sequence must have at least one element"
        # append STOP if not appended
        if optimal_action_seq[-1] != env.action_name_to_index("stop"):
            print("Error: the last action in the optimal action sequence must be STOP, but %d now, appending STOP."%(optimal_action_seq[-1]))
            optimal_action_seq.append(env.action_name_to_index("stop"))
       
    except habitat_sim.errors.GreedyFollowerError as e:
        print("Error: optimal path NOT found! set optimal action sequence to []")
        optimal_action_seq = []

    return optimal_action_seq