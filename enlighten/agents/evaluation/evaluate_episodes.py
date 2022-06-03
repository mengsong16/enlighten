import numpy as np
import torch

from enlighten.utils.path import *
from enlighten.utils.config_utils import parse_config
from enlighten.agents.common.seed import set_seed_except_env_seed
from enlighten.agents.common.other import get_device
from enlighten.envs.multi_nav_env import MultiNavEnv
from enlighten.agents.common.other import get_obs_channel_num
from enlighten.datasets.il_data_gen import load_behavior_dataset_meta, extract_observation
from enlighten.agents.models.decision_transformer import DecisionTransformer

class MeasureHistory:
    def __init__(self, id):
        self.id = id
        self.data = []
    
    def add(self, a):
        self.data.append(a)
    
    def len(self):
        return len(self.data)

    def mean(self): 
        data = np.array(self.data, dtype=np.float32) 
        return np.mean(data, axis=0)

    def max(self):
        data = np.array(self.data, dtype=np.float32)
        return np.max(data, axis=0)
    
    def min(self):
        data = np.array(self.data, dtype=np.float32)
        return np.min(data, axis=0)

    def std(self):
        data = np.array(self.data, dtype=np.float32)
        return np.std(data, axis=0)
    
    def print_full_summary(self):
        print("================  %s  ======================"%(self.id))
        print("Num: %f"%self.len())
        print("Min: %f"%self.min())
        print("Mean: %f"%self.mean())
        print("Max: %f"%self.max())
        print("Std: %f"%self.std())
        print("==============================================")



# evaluate decision transformer for one episode
def evaluate_one_episode_dt(
        episode,
        env,
        model,
        goal_form,
        sample,
        max_ep_len,
        device
    ):

    # turn model into eval mode and move to desired device
    model.eval()
    model.to(device=device)

    # reset env
    obs = env.reset(episode=episode, plan_shortest_path=False)
    obs_array = extract_observation(obs, env.observation_space.spaces)
    if goal_form == "rel_goal":
        goal = np.array(obs["pointgoal"], dtype="float32")
    elif goal_form == "distance_to_goal":
        goal = env.get_current_distance()
    # a0 is 0, shape (1,1)
    actions = torch.zeros((1, 1), device=device, dtype=torch.long)

    print("Scene id: %s"%(episode.scene_id))
    print("Goal position: %s"%(env.goal_position))
    print("Start position: %s"%(env.start_position))

    # change shape and convert to torch tensor
    # (C,H,W) --> (1,1,C,H,W)
    obs_array = np.expand_dims(np.expand_dims(obs_array, axis=0), axis=0)
    observations = torch.from_numpy(obs_array).to(device=device, dtype=torch.float32)
    # (goal_dim,) --> (1,1,goal_dim)
    if goal_form == "rel_goal":
        goal = np.expand_dims(np.expand_dims(goal, axis=0), axis=0)
        goals = torch.from_numpy(goal).to(device=device, dtype=torch.float32)
    # float --> (1,1,1)
    elif goal_form == "distance_to_goal":
        goals = torch.tensor(goal, device=device, dtype=torch.float32).reshape(1, 1, 1)

    # t0 is 0, shape (1,1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    # run under policy for max_ep_len step
    # keep all history steps, but only use context length to predict current action
    for t in range(max_ep_len):
        # predict according to the sequence from (s0,a0,r0) up to now (context)
        # need to input timesteps as positional embedding
        action = model.get_action(
            observations.to(dtype=torch.float32),
            actions.to(dtype=torch.float32),
            goals.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            sample=sample,
        )
        
        # append new action
        action = action.detach().cpu().item()
        new_action = torch.tensor(action, device=device, dtype=torch.long).reshape(1, 1)
        actions = torch.cat([actions, new_action], dim=1)

        # step the env according to the action, get new observation and goal
        obs, _, done, _ = env.step(action)
        obs_array = extract_observation(obs, env.observation_space.spaces)
        if goal_form == "rel_goal":
            goal = np.array(obs["pointgoal"], dtype="float32")
        elif goal_form == "distance_to_goal":
            goal = env.get_current_distance()

        # change shape and convert to torch tensor
        # (C,H,W) --> (1,1,C,H,W)
        obs_array = np.expand_dims(np.expand_dims(obs_array, axis=0), axis=0)
        new_obs = torch.from_numpy(obs_array).to(device=device, dtype=torch.float32)
        # (goal_dim,) --> (1,1,goal_dim)
        if goal_form == "rel_goal":
            goal = np.expand_dims(np.expand_dims(goal, axis=0), axis=0)
            new_goal = torch.from_numpy(goal).to(device=device, dtype=torch.float32)
        # float --> (1,1,1)
        elif goal_form == "distance_to_goal":
            new_goal = torch.tensor(goal, device=device, dtype=torch.float32).reshape(1, 1, 1)
            
        # append new observation and goal
        observations = torch.cat([observations, new_obs], dim=1)
        goals = torch.cat([goals, new_goal], dim=1)

        # append new timestep
        timesteps = torch.cat(
            [timesteps,
            torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        if done:
            break

    # collect measures
    episode_length = env.get_current_step()
    success = env.is_success()
    spl = env.get_spl()
    softspl = env.get_softspl()

    return episode_length, success, spl, softspl


# evaluate an agent in across scene env
class MultiEnvEvaluator:
    # eval_split: ["across_scene_test", "same_scene_test", "across_scene_val", "same_scene_val"]
    def __init__(self, eval_split, config_filename="imitation_learning.yaml", 
        env: MultiNavEnv = None, device=None):

        assert config_filename is not None, "needs config file to initialize trainer"
        
        config_file = os.path.join(config_path, config_filename)
        self.config = parse_config(config_file)

        # seed everything except env
        self.seed = int(self.config.get("seed"))
        set_seed_except_env_seed(self.seed)

        # create env if None
        if env is None:
            self.env = MultiNavEnv(config_file=config_filename) 
        else:
            self.env = env

        # device
        if device is None:
            self.device = get_device(self.config)
        else:
            self.device = device 

        # max episode length
        self.max_ep_len = int(self.config.get("max_ep_len"))  

        # goal_form
        self.goal_form = self.config.get("goal_form") 
        if self.goal_form not in ["rel_goal", "distance_to_goal"]:
            print("Undefined goal form: %s"%(self.goal_form))
            exit()

        # get name of evaluation folder
        self.experiment_name_to_load = self.config.get("eval_experiment_folder")
          
        # load episodes of behavior dataset for evaluation
        self.eval_episodes = load_behavior_dataset_meta(yaml_name=config_filename, 
            split_name=eval_split)

    # load dt model to be evaluated
    def load_dt_model(self):
        # create model
        model = DecisionTransformer(
            obs_channel = get_obs_channel_num(self.config),
            obs_width = int(self.config.get("image_width")), 
            obs_height = int(self.config.get("image_height")),
            goal_dim=int(self.config.get("goal_dimension")),
            goal_form=self.config.get("goal_form"),
            act_num=int(self.config.get("action_number")),
            context_length=int(self.config.get('K')),
            max_ep_len=int(self.config.get("max_ep_len")),  
            hidden_size=int(self.config.get('embed_dim')), # parameters starting from here will be passed to gpt2
            n_layer=int(self.config.get('n_layer')),
            n_head=int(self.config.get('n_head')),
            n_inner=int(4*self.config.get('embed_dim')),
            activation_function=self.config.get('activation_function'),
            n_positions=1024,
            resid_pdrop=float(self.config.get('dropout')),
            attn_pdrop=float(self.config.get('dropout')),
        )
        
        # get checkpoint path
        checkpoint_path = os.path.join(checkpoints_path, self.experiment_name_to_load, self.config.get("eval_checkpoint_file"))
        if os.path.exists(checkpoint_path):
            print("Loading checkpoint at: "+str(checkpoint_path))
        else:
            print("Error: checkpoint path does not exist: %s"%(checkpoint_path))
            exit()  
        
        # load checkpoint
        model.load_state_dict(torch.load(checkpoint_path))

        return model

    def evaluate_over_dataset(self, model=None, sample=True):
        if model is None:
            model = self.load_dt_model()
        
        episode_length_array = MeasureHistory("episode_length")
        success_array = MeasureHistory("success")
        spl_array = MeasureHistory("soft_spl")
        soft_spl_array = MeasureHistory("soft_spl")

        for i, episode in enumerate(self.eval_episodes):
            print('Episode: {}'.format(i+1))
            episode_length, success, spl, softspl = evaluate_one_episode_dt(
                episode,
                self.env,
                model,
                self.goal_form,
                sample,
                self.max_ep_len,
                self.device)
            
            episode_length_array.add(episode_length)
            success_array.add(float(success))
            spl_array.add(spl)
            soft_spl_array.add(softspl)
        
        print("==============================================")
        print("Episodes in total: %d"%(success_array.len()))
        print("Success rate: %d"%(success_array.mean()))
        print("SPL mean: %d"%(spl_array.mean()))
        print("Soft SPL mean: %d"%(soft_spl_array.mean()))
        print("==============================================")

if __name__ == "__main__":
    evaluator = MultiEnvEvaluator(eval_split="same_scene_test") # across_scene_test
    evaluator.evaluate_over_dataset()

        

    