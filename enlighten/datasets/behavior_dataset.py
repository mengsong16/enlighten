import random
import numpy as np
import copy
import torch
import math
from torch.utils.data import Dataset as TorchDataset
from enlighten.agents.common.other import get_obs_channel_num
import pickle
from enlighten.agents.common.other import get_device
from enlighten.utils.config_utils import parse_config
from enlighten.utils.path import *
from enlighten.agents.common.seed import set_seed_except_env_seed

class BehaviorDataset:
    """ Sample trajectory segments for supervised learning 
    """
    def __init__(self, config, device=None):
        self.config = config  # config is a dictionary
        if device is None:
            self.device = get_device(self.config)
        else:    
            self.device = device
        
         
        self.goal_dim = int(self.config.get("goal_dimension")) 
        self.obs_channel = get_obs_channel_num(self.config)
        if self.obs_channel == 0:
            print("Error: channel of observation input to the encoder is 0")
            exit()
        self.obs_width = int(self.config.get("image_width")) 
        self.obs_height = int(self.config.get("image_height"))
        self.goal_form = self.config.get("goal_form")
        print("goal form =====> %s"%(self.goal_form))
        self.batch_mode = self.config.get("batch_mode")
        print("batch mode =====> %s"%(self.batch_mode))
        self.state_form = self.config.get("state_form", "observation")
        print("state form =====> %s"%(self.state_form))
        # reward type
        self.reward_type = self.config.get("reward_type", "original")
        print("reward type =====> %s"%(self.reward_type))
        # reward scale [used in reward type "minus_one_zero"]
        self.reward_scale = float(self.config.get("reward_scale", 1.0))
        print("reward scale =====> %f"%(self.reward_scale))
        
        # augment transition dataset with relabeled actions
        self.relabel_actions = False
        algorithm_name = self.config.get("algorithm_name")
        if "dqn" in algorithm_name:
            if self.config.get("q_learning_type") == "ours":
                self.relabel_actions = True
        print("relabel_actions: %s"%(self.relabel_actions))

        if self.batch_mode == "random_segment":
            self.max_ep_len = int(self.config.get("dt_max_ep_len"))

        if self.config.get("algorithm_name") == "dt":
            self.pad_mode = self.config.get("pad_mode")
            self.context_length = int(self.config.get("K"))
        
        # load trajectories in the dataset
        self.load_trajectories()
        # load augment trajectories if necessary
        if self.config.get("use_augment_train_data"):
            self.load_augment_trajectories()
        
        # create trajectory indices from loaded trajectories
        self.generate_trajectory_index()

        # create transition indices from loaded trajectories
        self.generate_transition_index()


    def load_trajectories(self):
        # load all trajectories from the training dataset
        dataset_path = self.config.get("behavior_dataset_path")
        self.trajectories = []
        for file in os.listdir(dataset_path):
            if file.endswith(".pickle") and file.startswith("train_data"):
                current_train_dataset_path = os.path.join(dataset_path, file)
                print("Loading trajectories from %s"%(current_train_dataset_path))
                with open(current_train_dataset_path, 'rb') as f:
                    trajectories_current_file = pickle.load(f)
                    self.trajectories.extend(trajectories_current_file)
        

        self.num_trajectories = len(self.trajectories)

        print("Loaded %d training trajectories"%(self.num_trajectories))



    def load_augment_trajectories(self):
        # load augment training trajectories and use some or all of them
        dataset_path = self.config.get("behavior_dataset_path")
        dataset_path = os.path.join(dataset_path, "train_aug_data.pickle")
        print("Loading trajectories from %s"%(dataset_path))
        with open(dataset_path, 'rb') as f:
            augment_trajectories = pickle.load(f)
            self.trajectories.extend(augment_trajectories)

        augment_traj_num = len(augment_trajectories)
        self.num_trajectories += augment_traj_num

        print("Loaded %d augment training trajectories"%(augment_traj_num))
        print("Use %d training trajectories in total"%(self.num_trajectories))

    def generate_original_transition_index(self):
        self.transition_index_list = []
        self.num_steps = 0
        for traj_index, traj in enumerate(self.trajectories):
            trans_num = len(traj['observations']) - 1
            traj_index_list = [traj_index] * (trans_num)
            trans_index_list = list(range(trans_num))
            self.transition_index_list.extend(list(zip(traj_index_list, trans_index_list)))
            self.num_steps += len(traj['observations'])
        
        assert len(self.transition_index_list) == self.num_steps - len(self.trajectories), "Error: the number of transitions and steps do not match"
    
    def generate_relabel_action_transition_index(self):
        self.action_num = int(self.config.get("action_number"))

        self.transition_index_list = []
        self.num_steps = 0
        for traj_index, traj in enumerate(self.trajectories):
            # current trajectory
            trans_num = len(traj['observations']) - 1
            traj_index_list = [traj_index] * (trans_num * self.action_num)
            trans_index_list = []
            relabel_action_list = []
            optimal_action_list = []
            for trans_index in list(range(trans_num)):
                # current transition
                for action_index in list(range(self.action_num)):
                    trans_index_list.append(trans_index)
                    relabel_action_list.append(action_index)
                    # optimal action
                    if traj['actions'][trans_index] == action_index:
                        optimal_action_list.append(True)
                    # not optimal action
                    else:
                        optimal_action_list.append(False)
            self.transition_index_list.extend(list(zip(traj_index_list, trans_index_list, relabel_action_list, optimal_action_list)))
            self.num_steps += len(traj['observations'])

        assert len(self.transition_index_list) == (self.num_steps - len(self.trajectories))*self.action_num, "Error: the number of transitions and steps do not match"

    def total_transition_num(self):
        return len(self.transition_index_list)

    def generate_transition_index(self):
        if self.relabel_actions:
            self.generate_relabel_action_transition_index()
        else:
            self.generate_original_transition_index()

        self.shuffle_transition_dataset()

        print("Loaded %d actual environment steps"%(self.num_steps)) # real interaction steps
        print("Loaded %d transitions"%(len(self.transition_index_list)))
    
    def generate_trajectory_index(self):
        self.trajectory_index_list = list(range(self.num_trajectories))
        self.shuffle_trajectory_dataset()

    def advance_index_one_transition_batch(self, batch_size):
        stride = min(batch_size, len(self.transition_index_list) - self.transition_index)
        batch_inds = self.transition_index_list[self.transition_index:self.transition_index+stride]
        
        # advance index
        self.transition_index += stride

        if self.transition_index >= len(self.transition_index_list):
            self.transition_index = 0
        
        return batch_inds
    
    def advance_index_one_trajectory_batch(self, batch_size):
        stride = min(batch_size, len(self.trajectory_index_list) - self.trajectory_index)
        batch_inds = self.trajectory_index_list[self.trajectory_index:self.trajectory_index+stride]
        
        # advance index
        self.trajectory_index += stride

        if self.trajectory_index >= len(self.trajectory_index_list):
            self.trajectory_index = 0
        
        return batch_inds

    # 239 batches for 2000 trajectories for batch size 512
    def get_transition_batch_num(self, batch_size):
        batch_num = int(math.ceil(len(self.transition_index_list) / batch_size))
        return batch_num
    
    # 500 batches for 2000 trajectories for batch size 4
    def get_trajectory_batch_num(self, batch_size):
        batch_num = int(math.ceil(len(self.trajectory_index_list) / batch_size))
        return batch_num

    # sample a transition batch 
    # o: (B,C,H,W)
    # g: (B,goal_dim)
    # a: (B)
    # next_o: (B,C,H,W)
    # next_g: (B,goal_dim)
    # next_a: (B)
    # d: (B)
    def get_transition_batch(self, batch_size):
        if self.relabel_actions:
            return self.get_relabel_action_transition_batch(batch_size)
        else:
            return self.get_original_transition_batch(batch_size)
    
    def get_original_transition_batch(self, batch_size):
        batch_inds = self.advance_index_one_transition_batch(batch_size)
        real_batch_size = len(batch_inds)
        observation_space_shape = self.trajectories[0]['observations'][0].shape
        rel_goal_space_shape = self.trajectories[0]['rel_goals'][0].shape
        abs_goal_space_shape = self.trajectories[0]['abs_goals'][0].shape
        state_space_shape = self.trajectories[0]['state_positions'][0].shape
        
        # print(observation_space_shape)
        # print(rel_goal_space_shape)
        # print(abs_goal_space_shape)
        # exit()
        o = torch.zeros(
            real_batch_size,
            *observation_space_shape, dtype=torch.float32, device=self.device)
        
        s = torch.zeros(
            real_batch_size,
            *state_space_shape, dtype=torch.float32, device=self.device)

        rel_g = torch.zeros(
            real_batch_size,
            *rel_goal_space_shape, dtype=torch.float32, device=self.device)
        
        abs_g = torch.zeros(
            real_batch_size,
            *abs_goal_space_shape, dtype=torch.float32, device=self.device)

        a = torch.zeros(
            real_batch_size, dtype=torch.long, device=self.device)
        
        r = torch.zeros(
            real_batch_size, dtype=torch.float, device=self.device)

        
        d = torch.zeros(
            real_batch_size, dtype=torch.bool, device=self.device)

        next_o =  torch.zeros(
            real_batch_size,
            *observation_space_shape, dtype=torch.float32, device=self.device) 

        next_s = torch.zeros(
            real_batch_size,
            *state_space_shape, dtype=torch.float32, device=self.device)  
        
        next_rel_g = torch.zeros(
            real_batch_size,
            *rel_goal_space_shape, dtype=torch.float32, device=self.device)
        
        next_abs_g = torch.zeros(
            real_batch_size,
            *abs_goal_space_shape, dtype=torch.float32, device=self.device)
        
        next_a = torch.zeros(
            real_batch_size, dtype=torch.long, device=self.device)

        for batch_index, (traj_index, step_index) in enumerate(batch_inds):
            # memory id has changed by converting to tensor
            a[batch_index] = torch.tensor(self.trajectories[traj_index]['actions'][step_index], dtype=torch.long, device=self.device)
            o[batch_index] = torch.tensor(self.trajectories[traj_index]['observations'][step_index], dtype=torch.float, device=self.device)
            s[batch_index] = torch.tensor(self.trajectories[traj_index]['state_positions'][step_index], dtype=torch.float, device=self.device)
            next_o[batch_index] = torch.tensor(self.trajectories[traj_index]['observations'][step_index+1], dtype=torch.float, device=self.device)
            next_s[batch_index] = torch.tensor(self.trajectories[traj_index]['state_positions'][step_index+1], dtype=torch.float, device=self.device)
            rel_g[batch_index] = torch.tensor(self.trajectories[traj_index]['rel_goals'][step_index], dtype=torch.float, device=self.device)
            abs_g[batch_index] = torch.tensor(self.trajectories[traj_index]['abs_goals'][step_index], dtype=torch.float, device=self.device)
            done = self.trajectories[traj_index]['dones'][step_index+1]
            d[batch_index] = torch.tensor(done, dtype=torch.bool, device=self.device)
            if self.reward_type == "original":
                r[batch_index] = torch.tensor(self.trajectories[traj_index]['rewards'][step_index+1], dtype=torch.float, device=self.device)
            elif self.reward_type == "minus_one_zero":
                if done:
                    r[batch_index] = torch.tensor(0, dtype=torch.float, device=self.device)
                else:
                    r[batch_index] = torch.tensor(-1, dtype=torch.float, device=self.device) * self.reward_scale
            else:
                print("Error: undefined reward type: %s"%(self.reward_type))
                exit()
            next_rel_g[batch_index] = torch.tensor(self.trajectories[traj_index]['rel_goals'][step_index+1], dtype=torch.float, device=self.device)
            next_abs_g[batch_index] = torch.tensor(self.trajectories[traj_index]['abs_goals'][step_index+1], dtype=torch.float, device=self.device)
            next_a[batch_index] = torch.tensor(self.trajectories[traj_index]['actions'][step_index+1], dtype=torch.long, device=self.device)

        if self.goal_form == "rel_goal":
            output_goal = rel_g
            output_next_goal = next_rel_g
        elif self.goal_form == "abs_goal":
            output_goal = abs_g
            output_next_goal = next_abs_g
        else:
            print("Undefined goal form: %s"%(self.goal_form))
            exit()  
        
        if self.state_form == "state":
            output_obs = s
            output_next_obs = next_s
        elif self.state_form == "observation":
            output_obs = o
            output_next_obs = next_o
        else:
            print("Undefined state form: %s"%(self.state_form))
            exit()
        
        return output_obs, output_goal, a, r, output_next_obs, output_next_goal, d, next_a, None

    def get_relabel_action_transition_batch(self, batch_size):
        batch_inds = self.advance_index_one_transition_batch(batch_size)
        real_batch_size = len(batch_inds)
        observation_space_shape = self.trajectories[0]['observations'][0].shape
        rel_goal_space_shape = self.trajectories[0]['rel_goals'][0].shape
        abs_goal_space_shape = self.trajectories[0]['abs_goals'][0].shape
        state_space_shape = self.trajectories[0]['state_positions'][0].shape
        
        # print(observation_space_shape)
        # print(rel_goal_space_shape)
        # print(abs_goal_space_shape)
        # exit()
        o = torch.zeros(
            real_batch_size,
            *observation_space_shape, dtype=torch.float32, device=self.device)
        
        s = torch.zeros(
            real_batch_size,
            *state_space_shape, dtype=torch.float32, device=self.device)

        rel_g = torch.zeros(
            real_batch_size,
            *rel_goal_space_shape, dtype=torch.float32, device=self.device)
        
        abs_g = torch.zeros(
            real_batch_size,
            *abs_goal_space_shape, dtype=torch.float32, device=self.device)

        a = torch.zeros(
            real_batch_size, dtype=torch.long, device=self.device)
        
        r = torch.zeros(
            real_batch_size, dtype=torch.float, device=self.device)

        d = torch.zeros(
            real_batch_size, dtype=torch.bool, device=self.device)
        
        optimal_actions = torch.zeros(
            real_batch_size, dtype=torch.bool, device=self.device)

        next_o =  torch.zeros(
            real_batch_size,
            *observation_space_shape, dtype=torch.float32, device=self.device) 

        next_s = torch.zeros(
            real_batch_size,
            *state_space_shape, dtype=torch.float32, device=self.device)  
        
        next_rel_g = torch.zeros(
            real_batch_size,
            *rel_goal_space_shape, dtype=torch.float32, device=self.device)
        
        next_abs_g = torch.zeros(
            real_batch_size,
            *abs_goal_space_shape, dtype=torch.float32, device=self.device)
        
        next_a = torch.zeros(
            real_batch_size, dtype=torch.long, device=self.device)

        for batch_index, (traj_index, step_index, relabel_action, optimal_action) in enumerate(batch_inds):
            # memory id has changed by converting to tensor
            # use relabel action instead of original action
            a[batch_index] = torch.tensor(relabel_action, dtype=torch.long, device=self.device)
            optimal_actions[batch_index] = torch.tensor(optimal_action, dtype=torch.bool, device=self.device)
            
            o[batch_index] = torch.tensor(self.trajectories[traj_index]['observations'][step_index], dtype=torch.float, device=self.device)
            s[batch_index] = torch.tensor(self.trajectories[traj_index]['state_positions'][step_index], dtype=torch.float, device=self.device)
            next_o[batch_index] = torch.tensor(self.trajectories[traj_index]['observations'][step_index+1], dtype=torch.float, device=self.device)
            next_s[batch_index] = torch.tensor(self.trajectories[traj_index]['state_positions'][step_index+1], dtype=torch.float, device=self.device)
            rel_g[batch_index] = torch.tensor(self.trajectories[traj_index]['rel_goals'][step_index], dtype=torch.float, device=self.device)
            abs_g[batch_index] = torch.tensor(self.trajectories[traj_index]['abs_goals'][step_index], dtype=torch.float, device=self.device)
            done = self.trajectories[traj_index]['dones'][step_index+1]
            d[batch_index] = torch.tensor(done, dtype=torch.bool, device=self.device)
            if self.reward_type == "original":
                r[batch_index] = torch.tensor(self.trajectories[traj_index]['rewards'][step_index+1], dtype=torch.float, device=self.device)
            elif self.reward_type == "minus_one_zero":
                if done:
                    r[batch_index] = torch.tensor(0, dtype=torch.float, device=self.device)
                else:
                    r[batch_index] = torch.tensor(-1, dtype=torch.float, device=self.device) * self.reward_scale
            else:
                print("Error: undefined reward type: %s"%(self.reward_type))
                exit()
            next_rel_g[batch_index] = torch.tensor(self.trajectories[traj_index]['rel_goals'][step_index+1], dtype=torch.float, device=self.device)
            next_abs_g[batch_index] = torch.tensor(self.trajectories[traj_index]['abs_goals'][step_index+1], dtype=torch.float, device=self.device)
            next_a[batch_index] = torch.tensor(self.trajectories[traj_index]['actions'][step_index+1], dtype=torch.long, device=self.device)

        if self.goal_form == "rel_goal":
            output_goal = rel_g
            output_next_goal = next_rel_g
        elif self.goal_form == "abs_goal":
            output_goal = abs_g
            output_next_goal = next_abs_g
        else:
            print("Undefined goal form: %s"%(self.goal_form))
            exit()  
        
        if self.state_form == "state":
            output_obs = s
            output_next_obs = next_s
        elif self.state_form == "observation":
            output_obs = o
            output_next_obs = next_o
        else:
            print("Undefined state form: %s"%(self.state_form))
            exit()
        
        return output_obs, output_goal, a, r, output_next_obs, output_next_goal, d, next_a, optimal_actions


    # sample a trajectory batch  
    def get_trajectory_batch(self, batch_size):
        if self.batch_mode == "random_segment": 
            return self.get_batch_random_segment(batch_size=batch_size)
        elif self.batch_mode == "whole_trajectory":
            return self.get_batch_unequal_trajectory(batch_size=batch_size, whole_trajectory=True)
        elif self.batch_mode == "partial_trajectory": # random start
            return self.get_batch_unequal_trajectory(batch_size=batch_size, whole_trajectory=False)
        else:
            print("Undefined batch mode: %s"%(self.batch_mode))
            exit()    

    

    # sample a batch of whole trajectory or partial trajectory
    # reorganize it to fit rnn history format
    # o: (T,C,H,W), where T is the total number of steps in the batch
    # a: (T)
    # prev_a: (T)
    # g: (T,goal_dim)
    # dtg: (T,1)
    # ag: (T,goal_dim)
    def get_batch_unequal_trajectory(self, batch_size, whole_trajectory):
        # sample batch_size trajectories from the trajectory pool with replacement
        # batch_inds = np.random.choice(
        #     np.arange(self.num_trajectories),
        #     size=batch_size,
        #     replace=True
        # )

        batch_inds = self.advance_index_one_trajectory_batch(batch_size)
        # real batch size
        real_batch_size = len(batch_inds)

        o, a, g, dtg, ag, prev_a, seq_lengths = [], [], [], [], [], [], []
        for i in range(real_batch_size):
            # current trajectory
            traj = self.trajectories[int(batch_inds[i])]
            # starting index
            if whole_trajectory:
                si = 0
            else:    
                si = random.randint(0, len(traj['observations']) - 1) # trajectory length
            
            # stack current sequence into a numpy array
            obs_seg = np.expand_dims(np.stack(traj['observations'][si:]), axis=0)
            act_seg = np.expand_dims(np.stack(traj['actions'][si:]), axis=0)
            rel_goal_seg = np.expand_dims(np.stack(traj['rel_goals'][si:]), axis=0)
            dist_to_goal_seg = np.expand_dims(np.stack(traj['distance_to_goals'][si:]), axis=(0,2))
            # [1,seg_len, goal_dim]
            abs_goal_seg = np.expand_dims(np.stack(traj['abs_goals'][si:]), axis=0)
            # create prev action segment
            prev_act_seg = [-1]
            # note that extend will use the reference of a list thus will change its content
            prev_act_seg.extend(copy.deepcopy(traj['actions'][si:-1]))
            #print(prev_act_seg)
            prev_act_seg = np.expand_dims(np.stack(prev_act_seg), axis=0)

            #print(prev_act_seg.shape)
            #print(act_seg.shape)
            #print(abs_goal_seg.shape)

            o.append(obs_seg)
            a.append(act_seg)
            prev_a.append(prev_act_seg)
            g.append(rel_goal_seg)
            dtg.append(dist_to_goal_seg)
            ag.append(abs_goal_seg)
            seq_lengths.append(obs_seg.shape[1])

        # print(seq_lengths)

        # sort seqs in decreading order 
        o, a, g, dtg, ag, prev_a, batch_sizes, sorted_lengths = self.sort_seqs(o, a, g, dtg, ag, prev_a, seq_lengths)

        # for o_seg in o:
        #     print(o_seg.shape)

        o, a, g, dtg, ag, prev_a = self.concat_seqs_columnwise(o, a, g, dtg, ag, prev_a, batch_sizes)

        # concate elements in the list and convert numpy to torch tensor
        o = torch.from_numpy(np.concatenate(o, axis=0)).to(dtype=torch.float32, device=self.device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.long, device=self.device)
        g = torch.from_numpy(np.concatenate(g, axis=0)).to(dtype=torch.float32, device=self.device)
        dtg_numpy = np.concatenate(dtg, axis=0)
        dtg = torch.from_numpy(dtg_numpy).to(dtype=torch.float32, device=self.device)
        ag = torch.from_numpy(np.concatenate(ag, axis=0)).to(dtype=torch.float32, device=self.device)
        prev_a = torch.from_numpy(np.concatenate(prev_a, axis=0)).to(dtype=torch.long, device=self.device)
        batch_sizes = torch.from_numpy(batch_sizes).to(dtype=torch.long, device="cpu")
        value = torch.from_numpy(copy.deepcopy(dtg_numpy)).to(dtype=torch.float32, device=self.device)
        # print("====================")
        # print(o.shape) 
        # print(a.shape)
        # print(prev_a.shape)
        # print(g.shape)
        # print(dtg.shape)
        # print(ag.shape)
        # exit()

        if self.goal_form == "rel_goal":
            return o, a, prev_a, g, value, batch_sizes, sorted_lengths
        elif self.goal_form == "distance_to_goal":
            return o, a, prev_a, dtg, value, batch_sizes, sorted_lengths
        elif self.goal_form == "abs_goal":
            return o, a, prev_a, ag, value, batch_sizes, sorted_lengths
        else:
            print("Undefined goal form: %s"%(self.goal_form))
            exit() 

    # o: [(1,tlen,C,H,W)]
    # a: [(1,tlen)]
    # g: [(1,tlen,goal_dim)]
    # dtg: [(1,tlen,1)]
    # ag: [(1,tlen,goal_dim)]
    # prev_a: [(1,tlen)]
    # seq_lengths: a list of length of sequences
    # sort seqs from long to short
    def sort_seqs(self, o, a, g, dtg, ag, prev_a, seq_lengths):
        sorted_indices = np.argsort(-np.array(seq_lengths))
        sorted_lengths = -np.sort(-np.array(seq_lengths))
        o = [o[i] for i in sorted_indices]
        a = [a[i] for i in sorted_indices]
        g = [g[i] for i in sorted_indices]
        dtg = [dtg[i] for i in sorted_indices]
        ag = [ag[i] for i in sorted_indices]
        prev_a = [prev_a[i] for i in sorted_indices]

        #print(sorted_lengths)
        batch_sizes = np.zeros(sorted_lengths[0], dtype=int)
        for length in sorted_lengths:
            batch_sizes[:length] += 1
        #print(batch_sizes)
        #print(batch_sizes.shape)
        return o, a, g, dtg, ag, prev_a, batch_sizes, sorted_lengths
    
    def concat_seqs_columnwise(self, o, a, g, dtg, ag, prev_a, batch_sizes):
        new_o, new_a, new_g, new_dtg, new_ag, new_prev_a = [], [], [], [], [], []
        for column_index, batch_size in enumerate(batch_sizes):
            for i in range(batch_size):
                new_o.append(o[i][:,column_index,:,:,:])
                new_a.append(a[i][:,column_index])
                new_g.append(g[i][:,column_index,:])
                new_dtg.append(dtg[i][:,column_index,:])
                new_ag.append(ag[i][:,column_index,:])
                new_prev_a.append(prev_a[i][:,column_index])

        return new_o, new_a, new_g, new_dtg, new_ag, new_prev_a

    # sample a batch of segments of length K
    def get_batch_random_segment(self, batch_size):
        # sample batch_size trajectories from the trajectory pool with replacement
        # batch_inds = np.random.choice(
        #     np.arange(self.num_trajectories),
        #     size=batch_size,
        #     replace=True
        # )
        batch_inds = self.advance_index_one_trajectory_batch(batch_size)
        # real batch size
        real_batch_size = len(batch_inds)

        # organize a batch into observation, action, goal, distance to goal, timestep, mask
        # each element in the new batch is a trjectory segment, max_len: segment length which will be used to train sequence model
        o, a, g, dtg, timesteps, mask = [], [], [], [], [], []
        for i in range(real_batch_size):
            # current trajectory
            traj = self.trajectories[int(batch_inds[i])]
            # randomly pick a segment of context length from current trajectory starting from index si
            #print(len(traj['observations']))
            si = random.randint(0, len(traj['observations']) - 1) # trajectory length
            #si = 60

            # add batch dimension
            obs_seg = np.expand_dims(np.stack(traj['observations'][si:si + self.context_length]), axis=0)
            act_seg = np.expand_dims(np.stack(traj['actions'][si:si + self.context_length]), axis=0)
            # print("==============")
            # a = traj['rel_goals'][si:si + self.context_length]
            # for item in a:
            #     print(item.shape)
            # #print()
            # print("==============")
            rel_goal_seg = np.expand_dims(np.stack(traj['rel_goals'][si:si + self.context_length]), axis=0)
            dist_to_goal_seg = np.expand_dims(np.stack(traj['distance_to_goals'][si:si + self.context_length]), axis=(0,2))

            # print(obs_seg.shape) # (1,tlen,C,H,W)
            # print(act_seg.shape) # (1,tlen)
            # print(rel_goal_seg.shape) # (1,tlen,goal_dim)
            # print(dist_to_goal_seg.shape) # (1,tlen,1)
            
            # Note that if si+self.context_length exceed current traj length, only get elements until the episode ends
            o.append(obs_seg)
            a.append(act_seg)
            g.append(rel_goal_seg)
            dtg.append(dist_to_goal_seg)

            # tlen is the true length of current segment
            # tlen <= self.context_length
            tlen = o[-1].shape[1]
            #print(tlen)
            
            # each timestep is the step index inside this segment: e.g. [5,6,7]
            timesteps.append(np.arange(si, si + tlen).reshape(1, -1))
            # if actual index exceeds predefined max episode length, use the last step index (i.e. index max_ep_len - 1) instead
            # if timesteps in current segment >= self.max_ep_len: for each step in current segment, check whether it exceeds self.max_ep_len
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  
            #print(timesteps[-1].shape) # (1, tlen)

            # mask = 1 (attend to not paddding part) until tlen
            mask.append(np.ones((1, tlen)))
            #print(mask[-1].shape) # (1, tlen)

            # padding current segment to self.context_length if shorter than self.context_length
            op, ap, gp, dtgp, tp, mp = self.get_padding(self.context_length - tlen)

            # print(op.shape) # (1,K-tlen,C,H,W)
            # print(ap.shape) # (1,K-tlen)
            # print(gp.shape) # (1,K-tlen,goal_dim)
            # print(dtgp.shape) # (1,K-tlen,1)
            # print(tp.shape) # (1,K-tlen)
            # print(mp.shape) # (1,K-tlen)
            

            # left padding
            if self.pad_mode == "left":
                o[-1] = np.concatenate([op, o[-1]],  axis=1)
                a[-1] = np.concatenate([ap, a[-1]],  axis=1)
                g[-1] = np.concatenate([gp, g[-1]], axis=1)
                dtg[-1] = np.concatenate([dtgp, dtg[-1]], axis=1)
                timesteps[-1] = np.concatenate([tp, timesteps[-1]], axis=1)
                mask[-1] = np.concatenate([mp, mask[-1]], axis=1)
            # right padding
            elif self.pad_mode == "right":
                o[-1] = np.concatenate([o[-1], op],  axis=1)
                a[-1] = np.concatenate([a[-1], ap],  axis=1)
                g[-1] = np.concatenate([g[-1], gp], axis=1)
                dtg[-1] = np.concatenate([dtg[-1], dtgp], axis=1)
                timesteps[-1] = np.concatenate([timesteps[-1], tp], axis=1)
                mask[-1] = np.concatenate([mask[-1], mp], axis=1)
            else:
                print("Error: undefined padding mode: %s"%(self.pad_mode))
                exit()

            # print(o[-1].shape) # (1,K,C,H,W)
            # print(a[-1].shape) # (1,K)
            # print(g[-1].shape) # (1,K,goal_dim)
            # print(dtg[-1].shape) # (1,K,1)
            # print(timesteps[-1].shape) # (1,K)
            # print(mask[-1].shape) # (1,K)

            # if i>2:
            #     break
            

        # concate elements in the list and convert numpy to torch tensor
        o = torch.from_numpy(np.concatenate(o, axis=0)).to(dtype=torch.float32, device=self.device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.long, device=self.device)
        g = torch.from_numpy(np.concatenate(g, axis=0)).to(dtype=torch.float32, device=self.device)
        dtg = torch.from_numpy(np.concatenate(dtg, axis=0)).to(dtype=torch.float32, device=self.device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=self.device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=self.device)

        # print(o.size()) # (B,K,C,H,W)
        # print(a.size()) # (B,K)
        # print(g.size()) # (B,K,goal_dim)
        # print(dtg.size()) # (B,K,1)
        # print(timesteps.size()) # (B,K)
        # print(mask.size()) # (B,K)
        
        batch_shape = np.array([self.context_length] * real_batch_size, dtype=np.int32)

        if self.goal_form == "rel_goal":
            return o, a, g, timesteps, mask, batch_shape
        elif self.goal_form == "distance_to_goal":
            return o, a, dtg, timesteps, mask, batch_shape
        else:
            print("Undefined goal form: %s"%(self.goal_form))
            exit()  

    # get padding as numpy array
    # padding_length >= 0
    def get_padding(self, padding_length):
        # pad observation with 0
        op = np.zeros((1, padding_length, self.obs_channel, self.obs_height, self.obs_width))
        # pad action with 0 (stop)
        ap = np.zeros((1, padding_length))
        # pad goal with 0 
        gp = np.zeros((1, padding_length, self.goal_dim))
        # pad dtg with 0
        dtgp = np.zeros((1, padding_length, 1))
        # pad timestep with 0
        tp = np.zeros((1, padding_length))
        # pad mask with 0 (not attend to)
        mp = np.zeros((1, padding_length))

        return op, ap, gp, dtgp, tp, mp
    
    def shuffle_transition_dataset(self):
        random.shuffle(self.transition_index_list)
        # reset transtion index pointer
        self.transition_index = 0

        print("Transition dataset shuffled")
    
    def shuffle_trajectory_dataset(self):
        random.shuffle(self.trajectory_index_list)
        # reset transtion index pointer
        self.trajectory_index = 0

        print("Trajectory dataset shuffled")


if __name__ == "__main__":
    set_seed_except_env_seed(seed=1)
    config_file = os.path.join(config_path, "imitation_learning_dqn.yaml")
    config = parse_config(config_file)
    dataset = BehaviorDataset(config)

    batch_size = 512
    #batch_size = 4
    transition_batch_num = dataset.get_transition_batch_num(batch_size)
    #trajectory_batch_num = dataset.get_trajectory_batch_num(batch_size)
    
    for i in range(transition_batch_num):
    #for i in range(trajectory_batch_num):
        #output = dataset.get_trajectory_batch(batch_size=batch_size)
        #print(output[0].size()) # pytorch tensor
        #print(type(output[-1])) # numpy array
        #print(output[-1])
        o, g, a, r, next_o, next_g, d, next_a, optimal_action = dataset.get_transition_batch(batch_size=batch_size)
        #print(dataset.trajectories[0]["dones"][-1])
        #print(dataset.trajectories[0]["rewards"][-1])
        #print(o.size())
        #break
        # print(g.size())
        # print(a.size())
        # print(next_o.size())
        #break
        print("Batch %d Done"%(i+1))
        #print("Batch size: %d"%(o.size()[0]))
        print("Transition index: %d"%dataset.transition_index)
        #print("Trajectory index: %d"%dataset.trajectory_index)
        print("=========================")

    print("Batch size: %d"%(batch_size))
    print("Total number of transition batches: %d"%transition_batch_num)
    #print("Total number of trajectory batches: %d"%trajectory_batch_num)