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
from enlighten.agents.models.sac_agent import SACAgent
#from enlighten.agents.trainer.seq_trainer import SequenceTrainer
import torch
import torch.nn as nn
import datetime
import wandb

class SACExperiment:
    def __init__(self, config_filename):
        # get config
        config_file = os.path.join(config_path, config_filename)
        self.config = parse_config(config_file)

        # seed everything except env
        self.seed = int(self.config.get("seed"))
        set_seed_except_env_seed(self.seed)

        # set device
        self.device = get_device(self.config)

        # create env and dataset
        self.create_env_dataset(config_filename)

        # create agent (models)
        self.agent = SACAgent(self.config)
        
        
        # create path collector, replay buffer, trainer, algorithm
        self.expl_path_collector = MdpPathCollector(
            env=self.env,
            get_action_fn=self.agent.get_action,
        )
        self.replay_buffer = EnvReplayBuffer(
            self.config
        )
        self.trainer = SACTrainer(
            env=self.env,
            encoder=self.agent.encoder,
            policy=self.agent.policy,
            qf1=self.agent.qf1,
            qf2=self.agent.qf2,
            target_qf1=self.agent.target_qf1,
            target_qf2=self.agent.target_qf2,
            discount=float(self.config.get("discount")),
            encoder_lr=float(self.config.get("encoder_lr")),
            policy_lr=float(self.config.get("policy_lr")),
            qf_lr=float(self.config.get("qf_lr")),
            soft_target_tau=float(self.config.get("soft_target_tau")),
            target_update_period=int(self.config.get("target_update_period"))
        )

        # set experiment name (must before instantiate TorchBatchRLAlgorithm)
        self.set_experiment_name()

        # init wandb (must before instantiate TorchBatchRLAlgorithm)
        self.log_to_wandb = self.config.get("log_to_wandb")
        if self.log_to_wandb:
            self.init_wandb()

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
            save_every_epochs=int(self.config.get("save_every_epochs")),
            checkpoint_folder_name = self.project_name + "-" + self.group_name + "-" + self.experiment_name,
            log_to_wandb=self.log_to_wandb
        )
        # move all networks to the device
        #self.algorithm.to(self.device)
        
    def create_env_dataset(self, config_filename):
        self.env = MultiNavEnv(config_file=config_filename)

        self.env.seed(self.seed)

        train_episodes = load_behavior_dataset_meta(
            behavior_dataset_path=self.config.get("behavior_dataset_path"), 
            split_name="train")

        self.env.set_episode_dataset(episodes=train_episodes)


    def train(self):
        print("======> Start training from epoch 0 to epoch %d"%(int(self.config.get('num_epochs'))-1))
        self.algorithm.train()
    
    
    def test_rollouts(self):
        data = rollout(
            env=self.env,
            #get_action_fn=get_random_action,
            get_action_fn=self.agent.get_action,
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

    def set_experiment_name(self):
        self.project_name = self.config.get("algorithm_name").lower()
        self.group_name = self.config.get("experiment_name").lower()

        # experiment_name: seed-YearMonthDay-HourMiniteSecond
        # experiment name should be the same config run with different stochasticity
        now = datetime.datetime.now()
        self.experiment_name = "s%d-"%(self.seed)+now.strftime("%Y%m%d-%H%M%S").lower() 
        
    def init_wandb(self):
        wandb.init(
            name=self.experiment_name,
            group=self.group_name,
            project=self.project_name,
            config=self.config,
            dir=os.path.join(root_path),
        )


if __name__ == "__main__":
    exp = SACExperiment(config_filename=os.path.join(config_path, "sac_multi_envs.yaml"))
    #exp.test_rollouts()
    #exp.test_path_collector()
    #exp.test_replay_buffer()
    exp.train()