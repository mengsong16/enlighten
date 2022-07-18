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
from enlighten.agents.models.rnn_seq_model import RNNSequenceModel
from enlighten.agents.evaluation.ppo_eval import *
from enlighten.agents.common.tensor_related import (
    ObservationBatchingCache,
    batch_obs,
)
from enlighten.datasets.il_data_gen import goal_position_to_abs_goal

# evaluate an agent across scene single env
class AcrossEnvBaseEvaluator:
    # eval_splits: ["across_scene_test", "same_scene_test", "across_scene_val", "same_scene_val", "same_start_goal_test", "same_start_goal_val"]
    def __init__(self, eval_splits, config_filename="imitation_learning_dt.yaml", device=None):

        assert config_filename is not None, "needs config file to initialize trainer"
        
        config_file = os.path.join(config_path, config_filename)
        self.config = parse_config(config_file)

        # seed everything except env
        self.seed = int(self.config.get("seed"))
        set_seed_except_env_seed(self.seed)


        # create env
        self.create_env(config_filename=config_filename)

        # device
        if device is None:
            self.device = get_device(self.config)
        else:
            self.device = device 

        # max episode length
        self.max_ep_len = int(self.config.get("max_ep_len"))  

        # goal_form
        self.goal_form = self.config.get("goal_form") 
        if self.goal_form not in ["rel_goal", "distance_to_goal", "abs_goal"]:
            print("Undefined goal form: %s"%(self.goal_form))
            exit()
        
        # algorithm
        self.algorithm_name = self.config.get("algorithm_name")

        # get name of evaluation folder
        self.experiment_name_to_load = self.config.get("eval_experiment_folder")
          
        # load episodes of behavior datasets for evaluation
        print("====> Evaluation splits during training: %s"%(eval_splits))
        self.eval_dataset_episodes = {}
        for eval_split in eval_splits:
            episodes = load_behavior_dataset_meta(
                behavior_dataset_path=self.config.get("behavior_dataset_path"), 
                split_name=eval_split)
            self.eval_dataset_episodes[eval_split] = episodes

    # dummy
    def create_env(self, config_filename):
        return 

    # load dt model to be evaluated
    def load_model(self):
        # create model
        if self.algorithm_name == "dt":
            model = DecisionTransformer(
                obs_channel = get_obs_channel_num(self.config),
                obs_width = int(self.config.get("image_width")), 
                obs_height = int(self.config.get("image_height")),
                goal_dim=int(self.config.get("goal_dimension")),
                goal_form=self.config.get("goal_form"),
                act_num=int(self.config.get("action_number")),
                context_length=int(self.config.get('K')),
                max_ep_len=int(self.config.get("max_ep_len")),  
                pad_mode = str(self.config.get("pad_mode")),
                hidden_size=int(self.config.get('embed_dim')), # parameters starting from here will be passed to gpt2
                n_layer=int(self.config.get('n_layer')),
                n_head=int(self.config.get('n_head')),
                n_inner=int(4*self.config.get('embed_dim')),
                activation_function=self.config.get('activation_function'),
                n_positions=1024,
                resid_pdrop=float(self.config.get('dropout')),
                attn_pdrop=float(self.config.get('dropout')),
            )
        elif self.algorithm_name == "rnn":
            model = RNNSequenceModel(
                obs_channel = get_obs_channel_num(self.config),
                obs_width = int(self.config.get("image_width")), 
                obs_height = int(self.config.get("image_height")),
                goal_dim=int(self.config.get("goal_dimension")),
                goal_form=self.config.get("goal_form"),
                act_num=int(self.config.get("action_number")),
                max_ep_len=int(self.config.get("max_ep_len")),  
                rnn_hidden_size=int(self.config.get('rnn_hidden_size')), 
                obs_embedding_size=int(self.config.get('obs_embedding_size')), #512
                goal_embedding_size=int(self.config.get('goal_embedding_size')), #32
                act_embedding_size=int(self.config.get('act_embedding_size')), #32
                rnn_type=self.config.get('rnn_type'),
                supervise_value=self.config.get('supervise_value')
            )
        elif self.algorithm_name == "ppo":
            self.obs_transforms = get_active_obs_transforms(self.config)
            self.cache = ObservationBatchingCache()
            # assume a single env
            model = load_ppo_model(config=self.config, 
                observation_space=self.env.observation_space, 
                goal_observation_space=self.env.get_goal_observation_space(), 
                action_space=self.env.action_space,
                device=self.device,
                obs_transforms=self.obs_transforms)
            return model
        else:
            print("Error: undefined algorithm name: %s"%(self.algorithm_name))
            exit()
        
        # move model to correct device
        model.to(self.device)
        
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
    
    def get_metric_string(self, logs, eval_splits):
        print_str_dict = {}
        for split_name in eval_splits:
            print_str = ""
            print_str += "================== %s ======================\n"%(split_name)
            print_str += "Episodes in total: %d\n"%(logs[f"{split_name}/total_episodes"])
            print_str += "Success rate: %.4f\n"%(logs[f"{split_name}/success_rate"])
            print_str += "SPL mean: %.4f\n"%(logs[f"{split_name}/mean_spl"])
            #print("Soft SPL mean: %f"%(logs[f"{split_name}/mean_soft_spl"]))
            print_str += "==============================================\n"

            print_str_dict[split_name] = print_str

        return print_str_dict

    def print_metrics(self, logs, eval_splits):
        print_str_dict = self.get_metric_string(logs, eval_splits)
        for v in print_str_dict.values():
            print(v)

    def save_eval_logs(self, logs, eval_splits):
        # get metric string from logs
        print_str_dict = self.get_metric_string(logs, eval_splits)
    
        # get save folder
        save_folder = os.path.join(root_path, self.config.get("eval_dir"), self.config.get("eval_experiment_folder"))
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        # get checkpoint name
        checkpoint_name = self.config.get("eval_checkpoint_file")
        checkpoint_name = os.path.splitext(checkpoint_name)[0]
        
        for split_name, print_str in print_str_dict.items():
            txt_name =  f"{checkpoint_name}-{split_name}-eval_results.txt"
            with open(os.path.join(save_folder, txt_name), 'w') as outfile:
                    outfile.write(print_str)

            print("Saved evaluation file: %s"%(txt_name)) 

        

    