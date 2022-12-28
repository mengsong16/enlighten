from enlighten.envs.multi_nav_env import MultiNavEnv
from enlighten.agents.common.replay_buffer import EnvReplayBuffer
from enlighten.agents.common.data_collector import MdpPathCollector
from enlighten.agents.algorithms.sac_trainer import SACTrainer
from enlighten.agents.algorithms.online_rl_algorithms import TorchBatchRLAlgorithm
from enlighten.utils.path import *
from enlighten.utils.config_utils import parse_config
from enlighten.agents.common.seed import set_seed_except_env_seed
from enlighten.agents.models.mlp_network import MLPNetwork
from enlighten.agents.models.dt_encoder import ObservationEncoder, GoalEncoder
from enlighten.agents.common.other import get_obs_channel_num, get_device
from enlighten.datasets.common import load_behavior_dataset_meta
from enlighten.agents.common.data_collector import rollout, get_random_action
import torch
import torch.nn as nn

class SimpleMLPPolicy(nn.Module):
    def __init__(
            self,
            act_num,
            input_dim, #512+32
            hidden_size, #512
            hidden_layer, #2
    ):
        super().__init__()

        self.policy_network = MLPNetwork(
                input_dim=input_dim, 
                output_dim=act_num, 
                hidden_dim=hidden_size, 
                hidden_layer=hidden_layer)
        
        
        # acton logits --> action prob
        self.softmax = nn.Softmax(dim=-1)
    
    # for training, return distributions (not logits)
    def forward(self, input_embeddings):  
        # feed the input embeddings into the mlp policy
        # output: [B, act_num]
        pred_action_logits = self.policy_network(input_embeddings)

        # apply softmax to convert to probabilities
        # probs: [B, action_num]
        probs = self.softmax(pred_action_logits)

        return probs
    
    

class Encoder(nn.Module):
    def __init__(
            self,
            goal_dim,
            obs_channel,
            obs_embedding_size, #512
            goal_embedding_size, #32
    ):
        super().__init__()

        self.goal_encoder = GoalEncoder(goal_dim, goal_embedding_size)
        self.obs_encoder = ObservationEncoder(obs_channel, obs_embedding_size)

    # input: observations: [B, C, H, W]
    #        goals: [B,goal_dim]
    # output: [B, action_number]
    def forward(self, observations, goals):
        # (B,C,H,W) ==> (B,obs_embedding_size)
        observation_embeddings = self.obs_encoder(observations)
        
        # (B,goal_dim) ==> (B,goal_embedding_size)
        goal_embeddings = self.goal_encoder(goals)
        
         # (o,g) ==> [B,input_size]
        input_embeddings = torch.cat([observation_embeddings, goal_embeddings], dim=1)

        return input_embeddings


class SACExperiment():
    def __init__(self, config_filename):
        # get config
        config_file = os.path.join(config_path, config_filename)
        self.config = parse_config(config_file)

        # seed everything except env
        self.seed = int(self.config.get("seed"))
        set_seed_except_env_seed(self.seed)

        # set device
        self.device = get_device(self.config)

        # # set experiment name
        # self.set_experiment_name()

        # # init wandb
        # self.log_to_wandb = self.config.get("log_to_wandb")
        # if self.log_to_wandb:
        #     self.init_wandb()

        # create env and dataset
        self.create_env_dataset(config_filename)

        # create models
        self.create_models()
        
        # create path collector, replay buffer, trainer, algorithm
        self.expl_path_collector = MdpPathCollector(
            env=self.env,
            get_action_fn=self.get_action,
        )
        self.replay_buffer = EnvReplayBuffer(
            self.config
        )
        self.trainer = SACTrainer(
            env=self.env,
            encoder=self.encoder,
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            discount=float(self.config.get("discount")),
            encoder_lr=float(self.config.get("encoder_lr")),
            policy_lr=float(self.config.get("policy_lr")),
            qf_lr=float(self.config.get("qf_lr")),
            soft_target_tau=float(self.config.get("soft_target_tau")),
            target_update_period=int(self.config.get("target_update_period"))
            
        )
        self.algorithm = TorchBatchRLAlgorithm(
            trainer=self.trainer,
            exploration_env=self.env,
            exploration_data_collector=self.expl_path_collector,
            replay_buffer=self.replay_buffer,
            batch_size=int(self.config.get("batch_size")),
            max_path_length=int(self.config.get("max_steps_per_episode")),
            num_epochs=int(self.config.get("num_epochs")),
            num_expl_steps_per_train_loop=int(self.config.get("num_expl_steps_per_train_loop")),
            num_trains_per_train_loop=int(self.config.get("num_trains_per_train_loop")),
            num_expl_steps_before_training=int(self.config.get("num_expl_steps_before_training")),
        )
        self.algorithm.to(self.device)
        
    
    def create_env_dataset(self, config_filename):
        self.env = MultiNavEnv(config_file=config_filename)

        self.env.seed(self.seed)

        train_episodes = load_behavior_dataset_meta(
            behavior_dataset_path=self.config.get("behavior_dataset_path"), 
            split_name="train")

        self.env.set_episode_dataset(episodes=train_episodes)


    def create_models(self):
        obs_channel = get_obs_channel_num(self.config)
        self.goal_dim = int(self.config.get("goal_dimension"))
        self.act_num = int(self.config.get("action_number"))
        self.obs_embedding_size = int(self.config.get('obs_embedding_size')) #512
        self.goal_embedding_size = int(self.config.get('goal_embedding_size')) #32
        self.hidden_size = int(self.config.get('hidden_size'))
        self.hidden_layer = int(self.config.get('hidden_layer'))
        
        # shared by the following 5 networks
        self.encoder = Encoder(goal_dim=self.goal_dim,
            obs_channel=obs_channel,
            obs_embedding_size=self.obs_embedding_size,
            goal_embedding_size=self.goal_embedding_size)
        
        # two hidden layer MLPs
        self.qf1 = MLPNetwork(
            input_dim=self.obs_embedding_size+self.goal_embedding_size, 
            output_dim=self.act_num, 
            hidden_dim=self.hidden_size, 
            hidden_layer=self.hidden_layer)
        
        # two hidden layer MLPs
        self.qf2 = MLPNetwork(
            input_dim=self.obs_embedding_size+self.goal_embedding_size, 
            output_dim=self.act_num, 
            hidden_dim=self.hidden_size, 
            hidden_layer=self.hidden_layer)

        # two hidden layer MLPs
        self.target_qf1 = MLPNetwork(
            input_dim=self.obs_embedding_size+self.goal_embedding_size, 
            output_dim=self.act_num, 
            hidden_dim=self.hidden_size, 
            hidden_layer=self.hidden_layer)
        
        # two hidden layer MLPs
        self.target_qf2 = MLPNetwork(
            input_dim=self.obs_embedding_size+self.goal_embedding_size, 
            output_dim=self.act_num, 
            hidden_dim=self.hidden_size, 
            hidden_layer=self.hidden_layer)
        
        # two hidden layer MLPs
        self.policy = SimpleMLPPolicy(
            act_num=self.act_num, 
            input_dim=self.obs_embedding_size+self.goal_embedding_size,
            hidden_size=self.hidden_size, 
            hidden_layer=self.hidden_layer)
    
    # for evaluation
    # observations: [B,C,H,W]
    # goals: [B, goal_dim]
    # return actions:[B, 1]
    def get_action(self, observations, goals, sample=True):
        # forward the sequence with no grad
        with torch.no_grad():
            # get input embeddings
            input_embeddings = self.encoder(observations, goals)
            
            # get distributions
            probs = self.policy(input_embeddings)

            # sample from the distribution
            if sample:
                # each row is an independent distribution, draw 1 sample per distribution
                actions = torch.multinomial(probs, num_samples=1)
            # take the most likely action
            else:
                _, actions = torch.topk(probs, k=1, dim=-1)

        return actions

    def train(self):
        self.algorithm.train()
    
    def test_rollouts(self):
        data = rollout(
            env=self.env,
            #get_action_fn=get_random_action,
            get_action_fn=self.get_action,
            sample=True,
            max_path_length=int(self.config.get("max_steps_per_episode")),
            render=False,
        )

        print("="*30)
        print(data["observations"].shape)
        print(data["goals"].shape)
        print(data["actions"].shape)
        print(data["rewards"].shape)
        print(data["next_observations"].shape)
        print(data["next_goals"].shape)
        print(data["dones"].shape)
        #print(data["dones"])
        print("="*30)
    
    def test_path_collector(self):
        # collect m steps
        new_expl_paths = self.expl_path_collector.collect_new_paths(
            max_path_length=int(self.config.get("max_steps_per_episode")),
            num_steps=1000
        )
        print("="*30)
        print(self.expl_path_collector._num_paths_total)
        print(self.expl_path_collector._num_steps_total)
        print("="*30)
    
    def test_replay_buffer(self):
        # collect m steps
        new_expl_paths = self.expl_path_collector.collect_new_paths(
            max_path_length=int(self.config.get("max_steps_per_episode")),
            num_steps=1000
        )
        self.replay_buffer.add_paths(new_expl_paths)
        print("="*30)
        print(self.replay_buffer.num_steps_can_sample())
        print("="*30)



if __name__ == "__main__":
    exp = SACExperiment(config_filename=os.path.join(config_path, "sac_multi_envs.yaml"))
    #exp.test_rollouts()
    #exp.test_path_collector()
    exp.test_replay_buffer()
    #exp.train()