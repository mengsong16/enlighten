import numpy as np
import torch

from enlighten.utils.path import *
from enlighten.utils.config_utils import parse_config
from enlighten.agents.common.seed import set_seed_except_env_seed
from enlighten.agents.common.other import get_device
from enlighten.envs.multi_nav_env import MultiNavEnv
from enlighten.agents.common.other import get_obs_channel_num
from enlighten.datasets.common import load_behavior_dataset_meta
from enlighten.agents.models.decision_transformer import DecisionTransformer
from enlighten.agents.models.rnn_seq_model import RNNSequenceModel
from enlighten.agents.models.mlp_policy_model import MLPPolicy
from enlighten.agents.evaluation.ppo_eval import *
from enlighten.agents.common.tensor_related import (
    ObservationBatchingCache,
    batch_obs,
)
from enlighten.datasets.common import goal_position_to_abs_goal
import pickle
import matplotlib.pyplot as plt
from enlighten.agents.models.rnn_seq_model import DDBC
from enlighten.agents.models.q_network import QNetwork

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

        # max evaluation episode length
        self.max_ep_len = int(self.config.get("max_steps_per_episode"))  

        # goal form
        self.goal_form = self.config.get("goal_form") 
        if self.goal_form not in ["rel_goal", "distance_to_goal", "abs_goal"]:
            print("Undefined goal form: %s"%(self.goal_form))
            exit()
        
        # state form
        self.state_form = self.config.get("state_form", "observation") 
        if self.state_form not in ["state", "observation"]:
            print("Undefined state form: %s"%(self.goal_form))
            exit()
        
        # algorithm
        self.algorithm_name = self.config.get("algorithm_name")

        # get name of evaluation folder
        self.experiment_name_to_load = self.config.get("eval_experiment_folder")

        # action space
        self.action_type = self.config.get("action_type", "cartesian")
        print("=========> Action type: %s"%(self.action_type))
        
          
        # load episodes of behavior datasets for evaluation
        self.eval_splits = eval_splits
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

    # load dt or rnn model to be evaluated
    def load_model(self, checkpoint_file):
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
                max_ep_len=int(self.config.get("dt_max_ep_len")),  
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
        elif self.algorithm_name == "rnn_bc":
            model = RNNSequenceModel(
                obs_channel = get_obs_channel_num(self.config),
                obs_width = int(self.config.get("image_width")), 
                obs_height = int(self.config.get("image_height")),
                goal_dim=int(self.config.get("goal_dimension")),
                goal_form=self.config.get("goal_form"),
                act_num=int(self.config.get("action_number")),
                rnn_hidden_size=int(self.config.get('rnn_hidden_size')), 
                obs_embedding_size=int(self.config.get('obs_embedding_size')), #512
                goal_embedding_size=int(self.config.get('goal_embedding_size')), #32
                act_embedding_size=int(self.config.get('act_embedding_size')), #32
                rnn_type=self.config.get('rnn_type'),
                supervise_value=self.config.get('supervise_value'),
                domain_adaptation=self.config.get('domain_adaptation'),
                temperature=float(self.config.get('temperature', 1.0))
            )
        elif self.algorithm_name == "rnn_bc_online":
            model = DDBC(
                obs_channel = get_obs_channel_num(self.config),
                obs_width = int(self.config.get("image_width")), 
                obs_height = int(self.config.get("image_height")),
                goal_dim=int(self.config.get("goal_dimension")),
                goal_form=self.config.get("goal_form"),
                act_num=int(self.config.get("action_number")),
                rnn_hidden_size=int(self.config.get('rnn_hidden_size')), 
                obs_embedding_size=int(self.config.get('obs_embedding_size')), #512
                goal_embedding_size=int(self.config.get('goal_embedding_size')), #32
                act_embedding_size=int(self.config.get('act_embedding_size')), #32
                rnn_type=self.config.get('rnn_type'),
                supervise_value=self.config.get('supervise_value'),
                device=self.device,
                temperature=float(self.config.get('temperature', 1.0))
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
                obs_transforms=self.obs_transforms,
                checkpoint_file=checkpoint_file)
            # return here because we already load the model and move it to the correct device
            return model
        elif self.algorithm_name == "mlp_bc":
            model = MLPPolicy(
                obs_channel = get_obs_channel_num(self.config),
                obs_width = int(self.config.get("image_width")), 
                obs_height = int(self.config.get("image_height")),
                goal_dim=int(self.config.get("goal_dimension")),
                goal_form=self.config.get("goal_form"),
                act_num=int(self.config.get("action_number")),
                obs_embedding_size=int(self.config.get('obs_embedding_size')), #512
                goal_embedding_size=int(self.config.get('goal_embedding_size')), #32
                hidden_size=int(self.config.get('hidden_size')),
                hidden_layer=int(self.config.get('hidden_layer')),
                state_form=self.config.get('state_form'),
                state_dimension=int(self.config.get('state_dimension')),
                temperature=float(self.config.get('temperature', 1.0))
            )
        elif "dqn" in self.algorithm_name or "mlp_sqn" in self.algorithm_name:
             # action type
            self.action_type = self.config.get("action_type", "cartesian")
            print("=========> Action type: %s"%(self.action_type))
            if self.action_type == "polar":
                self.action_number = 37
            else:
                self.action_number = int(self.config.get("action_number"))

            model = QNetwork(
                obs_channel = get_obs_channel_num(self.config),
                obs_width = int(self.config.get("image_width")), 
                obs_height = int(self.config.get("image_height")),
                goal_dim=int(self.config.get("goal_dimension")),
                goal_form=self.config.get("goal_form"),
                act_num=self.action_number,
                obs_embedding_size=int(self.config.get('obs_embedding_size')), #512
                goal_embedding_size=int(self.config.get('goal_embedding_size')), #32
                hidden_size=int(self.config.get('hidden_size')),
                hidden_layer=int(self.config.get('hidden_layer')),
                state_form=self.config.get('state_form'),
                state_dimension=int(self.config.get('state_dimension'))
            )
        else:
            print("Error: undefined algorithm name: %s"%(self.algorithm_name))
            exit()
        
        # move model to correct device
        model.to(self.device)
        
        # get checkpoint path
        checkpoint_path = os.path.join(checkpoints_path, self.experiment_name_to_load, checkpoint_file)
        if os.path.exists(checkpoint_path):
            print("Loading checkpoint at: "+str(checkpoint_path))
        else:
            print("Error: checkpoint path does not exist: %s"%(checkpoint_path))
            exit()  
        
        # load checkpoint
        checkpoint = torch.load(checkpoint_path)
        # load weights
        #model.load_state_dict(checkpoint["model_state_dict"])
        if "state_dict" in checkpoint.keys():
            model.load_state_dict(checkpoint["state_dict"])
        elif "model_state_dict" in checkpoint.keys():
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            print("Error: unknown model state dict key")
            exit()

        return model
    
    def extract_int_from_string(self, r):
        s = ''.join(x for x in r if x.isdigit())
        return int(s)

    def evaluate_over_checkpoints(self, sample):
        checkpoint_files = list(self.config.get("eval_checkpoint_file"))
        # evaluate all checkpoints
        if "*" in checkpoint_files:
            checkpoint_files = []
            checkpoint_folder = os.path.join(checkpoints_path, self.experiment_name_to_load)
            for file in os.listdir(checkpoint_folder):
                if file.endswith(".pth") and file.startswith("ckpt."):
                   checkpoint_files.append(file) 
        
        # sort files according to checkpoint index
        checkpoint_indices = [self.extract_int_from_string(r) for r in checkpoint_files]
        sort_indices = np.argsort(np.array(checkpoint_indices, dtype=np.int32))
        checkpoint_files = [checkpoint_files[i] for i in sort_indices]
        
        success_rate = {}
        spl = {}

        for split_name in self.eval_splits:
            success_rate[split_name] = []
            spl[split_name] = []

        for checkpoint_file in checkpoint_files:
            print("================== %s evaluation Start ==================="%(checkpoint_file))
            logs, current_checkpoint_results = self.evaluate_over_datasets(checkpoint_file=checkpoint_file, model=None, sample=sample)

            # record current checkpoint result
            for split_name in self.eval_splits:
                success_rate[split_name].append(current_checkpoint_results[split_name]["success_rate"])
                spl[split_name].append(current_checkpoint_results[split_name]["spl"])            


            self.print_metrics(logs, self.eval_splits)
            self.save_eval_logs(logs, self.eval_splits, checkpoint_file)
            print("================== %s evaluation Done ==================="%(checkpoint_file))
        
        # dump results
        dump_folder = os.path.join(root_path, self.config.get("eval_dir"), self.config.get("eval_experiment_folder"))
        if not os.path.exists(dump_folder):
            os.mkdir(dump_folder)
        
        with open(os.path.join(dump_folder, "success_rate.pickle"), 'wb') as handle:
            pickle.dump(success_rate, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(os.path.join(dump_folder, "spl.pickle"), 'wb') as handle:
            pickle.dump(spl, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(os.path.join(dump_folder, "checkpoint_list.pickle"), 'wb') as handle:
            pickle.dump(checkpoint_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print("Evaluated checkpoints: %s"%str(checkpoint_indices))
        print("Done")

    def get_metric_string(self, logs, eval_splits):
        print_str_dict = {}
        for split_name in eval_splits:
            print_str = ""
            print_str += "================== %s ======================\n"%(split_name)
            print_str += "Episodes in total: %d\n"%(logs[f"{split_name}/total_episodes"])
            print_str += "Success rate: %.4f\n"%(logs[f"{split_name}/success_rate"])
            print_str += "SPL: %.4f\n"%(logs[f"{split_name}/spl"])
            #print("Soft SPL mean: %f"%(logs[f"{split_name}/mean_soft_spl"]))
            print_str += "==============================================\n"

            print_str_dict[split_name] = print_str

        return print_str_dict

    def print_metrics(self, logs, eval_splits):
        print_str_dict = self.get_metric_string(logs, eval_splits)
        for v in print_str_dict.values():
            print(v)

    def save_eval_logs(self, logs, eval_splits, checkpoint_file):
        # get metric string from logs
        print_str_dict = self.get_metric_string(logs, eval_splits)
    
        # get save folder
        save_folder = os.path.join(root_path, self.config.get("eval_dir"), self.config.get("eval_experiment_folder"))
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        # get checkpoint name
        checkpoint_name = os.path.splitext(checkpoint_file)[0]
        
        txt_name =  f"{checkpoint_name}-eval_results.txt"
        with open(os.path.join(save_folder, txt_name), 'w') as outfile:
            for print_str in print_str_dict.values():
                outfile.write(print_str)

        print("Saved evaluation file: %s"%(txt_name)) 

    def plot_checkpoint_one_graph(self, x, curves, eval_metric, save_folder, x_unit):
        # replace "_" with space
        eval_metric_name = eval_metric.replace("_", " ")

        # plotting the curves 
        for eval_split, curve in curves.items():
            plt.plot(x, curve, label=eval_split)
        
        # x, y axis start from 0
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)

        # naming the x axis
        plt.xlabel('number of %s'%(x_unit))
       
        # naming the y axis
        plt.ylabel(eval_metric_name)
        
        # giving a title to the graph
        title = eval_metric_name
        plt.title(title)

        # show a legend on the plot
        plt.legend()

        # save plot
        plt.savefig(os.path.join(save_folder, eval_metric+'_plot.png'))

        plt.close()  

    def plot_checkpoint_graphs(self):

        if self.config.get("algorithm_name") == "mlp_bc":
            checkpoint_interval = int(self.config.get("save_every_epochs"))
            x_unit = "epochs"
        elif "rnn_bc" in self.config.get("algorithm_name"):
            checkpoint_interval = int(self.config.get("save_every_iterations")) * int(self.config.get("num_steps_per_iter"))
            x_unit = "updates"
        elif "ppo" in self.config.get("algorithm_name"):
            if int(self.config.get("checkpoint_interval")) >= 0:
                checkpoint_interval = int(self.config.get("checkpoint_interval"))
            else:
                checkpoint_interval = int(int(self.config.get("total_num_steps")) / int(self.config.get("num_checkpoints")))
            x_unit = "environment steps"
        else:
            print("Error: undefined algorithm")
            exit()

        load_folder = os.path.join(root_path, self.config.get("eval_dir"), self.config.get("eval_experiment_folder"))

        checkpoint_index_path = os.path.join(load_folder, "checkpoint_list.pickle")
        print("Loading checkpoint indices from %s"%(checkpoint_index_path))
        with open(checkpoint_index_path, 'rb') as f:
            checkpoint_index_array = pickle.load(f)
            # convert start indexing from 0 to 1
            checkpoint_index_array = np.array(checkpoint_index_array, dtype=int) + 1

        # x axis values 
        x = checkpoint_index_array * checkpoint_interval

        eval_metrics = ["success_rate", "spl"]
        for eval_metric in eval_metrics:
            # load results
            eval_result_path = os.path.join(load_folder, "%s.pickle"%(eval_metric))
            print("Loading evaluation results from %s"%(eval_result_path))
            with open(eval_result_path, 'rb') as f:
                eval_results = pickle.load(f)
            
            curves = eval_results
            self.plot_checkpoint_one_graph(x, curves, eval_metric, load_folder, x_unit)

        print("Done.")

        

    