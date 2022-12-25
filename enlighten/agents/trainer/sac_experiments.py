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

import torch
import torch.nn as nn

class SimpleMLPPolicy(nn.Module):
    def __init__(
            self,
            act_num,
            obs_embedding_size, #512
            goal_embedding_size, #32
            hidden_size, #512
            hidden_layer, #2
            goal_encoder,
            obs_encoder
    ):
        super().__init__()
        self.goal_encoder = goal_encoder
        self.obs_encoder = obs_encoder
        self.policy_network = MLPNetwork(
                input_dim=obs_embedding_size+goal_embedding_size, 
                output_dim=act_num, 
                hidden_dim=hidden_size, 
                hidden_layer=hidden_layer)
        
        # acton logits --> action prob
        self.softmax = nn.Softmax(dim=-1)

    # input: observations: [B, C, H, W]
    #        goals: [B,goal_dim]
    # output: [B, action_number]
    def encoder_forward(self, observations, goals):
        # (B,C,H,W) ==> (B,obs_embedding_size)
        observation_embeddings = self.obs_encoder(observations)
        
        # (B,goal_dim) ==> (B,goal_embedding_size)
        goal_embeddings = self.goal_encoder(goals)
        
         # (o,g) ==> [B,input_size]
        input_embeddings = torch.cat([observation_embeddings, goal_embeddings], dim=1)

        return input_embeddings

    # for training
    def forward(self, observations, goals):
        # embed each input modality with a different head
        input_embeddings = self.encoder_forward(observations, goals)
            
        # feed the input embeddings into the mlp policy
        # output: [B, act_num]
        pred_action_logits = self.policy_network(input_embeddings)

        return pred_action_logits
    
    # for evaluation
    def get_action(self, observations, goals, sample=True):
        # forward the sequence with no grad
        with torch.no_grad():
            # embed each input modality with a different head
            pred_action_logits = self.forward(observations, goals)

            # apply softmax to convert to probabilities
            # probs: [B, action_num]
            probs = self.softmax(pred_action_logits)

            # sample from the distribution
            if sample:
                # each row is an independent distribution, draw 1 sample per distribution
                actions = torch.multinomial(probs, num_samples=1)
            # take the most likely action
            else:
                _, actions = torch.topk(probs, k=1, dim=-1)

        
        return actions



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

        # set experiment name
        self.set_experiment_name()

        # init wandb
        self.log_to_wandb = self.config.get("log_to_wandb")
        if self.log_to_wandb:
            self.init_wandb()

        # create env and dataset
        self.create_env_dataset(config_filename)

        # create models
        self.create_models()
        
        # create path collector, replay buffer, trainer, algorithm
        self.expl_path_collector = MdpPathCollector(
            self.env,
            self.policy,
        )
        self.replay_buffer = EnvReplayBuffer(
            self.config
        )
        self.trainer = SACTrainer(
            env=self.env,
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            **variant['trainer_kwargs']
        )
        self.algorithm = TorchBatchRLAlgorithm(
            trainer=self.trainer,
            exploration_env=self.env,
            exploration_data_collector=self.expl_path_collector,
            replay_buffer=self.replay_buffer,
            **variant['algorithm_kwargs']
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
        self.goal_encoder = GoalEncoder(self.goal_dim, self.goal_embedding_size)
        self.obs_encoder = ObservationEncoder(obs_channel, self.obs_embedding_size)

        # two hidden layers
        self.qf1 = MLPNetwork(
            input_dim=self.obs_embedding_size+self.goal_embedding_size, 
            output_dim=self.act_num, 
            hidden_dim=self.hidden_size, 
            hidden_layer=self.hidden_layer)
        
        # two hidden layers
        self.qf2 = MLPNetwork(
            input_dim=self.obs_embedding_size+self.goal_embedding_size, 
            output_dim=self.act_num, 
            hidden_dim=self.hidden_size, 
            hidden_layer=self.hidden_layer)

        # two hidden layers
        self.target_qf1 = MLPNetwork(
            input_dim=self.obs_embedding_size+self.goal_embedding_size, 
            output_dim=self.act_num, 
            hidden_dim=self.hidden_size, 
            hidden_layer=self.hidden_layer)
        
        # two hidden layers
        self.target_qf2 = MLPNetwork(
            input_dim=self.obs_embedding_size+self.goal_embedding_size, 
            output_dim=self.act_num, 
            hidden_dim=self.hidden_size, 
            hidden_layer=self.hidden_layer)
        
        # two hidden layers
        self.policy = SimpleMLPPolicy(
            act_num=self.act_num, 
            obs_embedding_size=self.obs_embedding_size,
            goal_embedding_size=self.goal_embedding_size,
            hidden_size=self.hidden_size, 
            hidden_layer=self.hidden_layer,
            goal_encoder=self.goal_encoder,
            obs_encoder=self.obs_encoder)

    
    def train(self):
        self.algorithm.train()

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    
    experiment(variant)
