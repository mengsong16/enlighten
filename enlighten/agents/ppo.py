import torch

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed, get_seed
from garage.sampler import RaySampler, LocalSampler, FragmentWorker
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

from enlighten.envs import NavEnv, create_gym_env
  


@wrap_experiment
def ppo_nav(ctxt=None, seed=1):
    """Train PPO with NavEnv environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    
    set_seed(seed)

    env = create_gym_env(config_filename="navigate_with_flashlight.yaml") 

    #print(env.spec)

    
    trainer = Trainer(ctxt)

    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    
    sampler = RaySampler(agents=policy,
                         envs=env,
                         max_episode_length=env.spec.max_episode_length)
    
    #print('-------ppo-----')
    #print(get_seed())
    
    
    #sampler = LocalSampler(agents=policy,
    #                       envs=env,
    #                       max_episode_length=env.spec.max_episode_length,
    #                       worker_class=FragmentWorker)                     
    
    algo = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               sampler=sampler,
               discount=0.99,
               center_adv=False)

    trainer.setup(algo, env)
    trainer.train(n_epochs=10, batch_size=10000)
    
if __name__ == "__main__": 
    ppo_nav(seed=1)
