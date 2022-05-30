import numpy as np
import torch


# evaluate one episode
# used by decision transformer
# compare with evaluate_episode:
#   has extra input scale
#   target_return has different input location in bc and transformer
#   need to input timesteps
#   has extra input mode
def evaluate_episode_dt(
        env,
        state_dim,
        act_dim,
        model,
        sample,
        target_return,
        max_ep_len,
        device
    ):

    # turn policy model into eval mode
    model.eval()
    model.to(device=device)

    # reset env
    obs = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    observations = torch.from_numpy(obs).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    # run under policy for max_ep_len step
    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # post pad a 0 to action sequence and reward sequence
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

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
        actions[-1] = action
        
        # step the env according to action, get new state and new reward
        action = action.detach().cpu().numpy()
        obs, reward, done, _ = env.step(action)

        # append new observation
        obs = torch.from_numpy(obs).to(device=device).reshape(1, state_dim)
        observations = torch.cat([observations, obs], dim=0)
        
        # append new reward 
        rewards[-1] = reward

        pred_return = target_return[0,-1]
        
        # append target return
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
       
        # append new timestep
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        # update episode return and length
        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length

# evaluate during training
def eval_episodes(target_rew, num_eval_episodes):
    # collect returns and lengths for num_eval_episodes episodes
    
    returns, lengths = [], []
    # for _ in range(num_eval_episodes):
    #     with torch.no_grad():
    #         ret, length = evaluate_episode_dt(
    #             env,
    #             state_dim,
    #             act_dim,
    #             model,
    #             sample=sample,
    #             target_return,
    #             max_ep_len,
    #             device=device,
    #         )
            
    #     returns.append(ret)
    #     lengths.append(length)
    return {
        f'target_{target_rew}_return_mean': np.mean(returns),
        f'target_{target_rew}_return_std': np.std(returns),
        f'target_{target_rew}_length_mean': np.mean(lengths),
        f'target_{target_rew}_length_std': np.std(lengths),
    }
    
