from gym.envs.mujoco import HalfCheetahEnv

from enlighten.envs.multi_nav_env import MultiNavEnv
from enlighten.agents.common.replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from enlighten.agents.common.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from enlighten.agents.algorithms.sac_trainer import SACTrainer
from enlighten.agents.algorithms.online_rl_algorithms import TorchBatchRLAlgorithm
from enlighten.utils.path import *
from enlighten.utils.config_utils import parse_config
from enlighten.agents.common.seed import set_seed_except_env_seed
from enlighten.agents.models.mlp_network import MLPNetwork
from enlighten.agents.models.dt_encoder import ObservationEncoder, GoalEncoder
from enlighten.agents.common.other import get_obs_channel_num
from enlighten.datasets.common import load_behavior_dataset_meta

class SACExperiment():
    def __init__(self, config_filename):
        # get config
        config_file = os.path.join(config_path, config_filename)
        self.config = parse_config(config_file)

        # seed everything except env
        self.seed = int(self.config.get("seed"))
        set_seed_except_env_seed(self.seed)

        self.create_env_dataset(config_filename)

        expl_env = NormalizedBoxEnv(HalfCheetahEnv())
        eval_env = NormalizedBoxEnv(HalfCheetahEnv())

        self.create_models()
        
        eval_policy = MakeDeterministic(self.policy)
        eval_path_collector = MdpPathCollector(
            eval_env,
            eval_policy,
        )
        expl_path_collector = MdpPathCollector(
            expl_env,
            self.policy,
        )
        replay_buffer = EnvReplayBuffer(
            self.config
        )
        trainer = SACTrainer(
            env=eval_env,
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            **variant['trainer_kwargs']
        )
        algorithm = TorchBatchRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            **variant['algorithm_kwargs']
        )
        algorithm.to(ptu.device)
        algorithm.train()
    
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
        self.policy = MLPNetwork(
            input_dim=self.obs_embedding_size+self.goal_embedding_size, 
            output_dim=self.act_num, 
            hidden_dim=self.hidden_size, 
            hidden_layer=self.hidden_layer)

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
    setup_logger('name-of-experiment', variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
