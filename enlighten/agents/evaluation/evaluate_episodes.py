import numpy as np
import torch


# evaluate one episode
# used by decision transformer
# compare with evaluate_episode:
#   has extra input scale
#   target_return has different input location in bc and transformer
#   need to input timesteps
#   has extra input mode
def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # add noise to state
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # post pad a 0 to action sequence and reward sequence
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        # predict according to the sequence from (s0,a0,r0) up to now (context)
        # need to input timesteps as positional embedding
        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            #rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        # append new action
        actions[-1] = action
        
        # step the env according to action, get new state and new reward
        action = action.detach().cpu().numpy()
        state, reward, done, _ = env.step(action)

        # append new state
        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        
        # append new reward 
        rewards[-1] = reward

        # if delayed reward, target_return should minus scaled current reward
        # note that pred_return is not a return predicted by the model
        # reward is a true return given by the env
        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
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
def eval_episodes(target_rew):
    # collect returns and lengths for num_eval_episodes episodes
    
    returns, lengths = [], []
    for _ in range(num_eval_episodes):
        with torch.no_grad():
            ret, length = evaluate_episode_rtg(
                env,
                state_dim,
                act_dim,
                model,
                max_ep_len=max_ep_len,
                scale=scale,
                target_return=target_rew/scale,
                mode=mode,
                state_mean=state_mean,
                state_std=state_std,
                device=device,
            )
            
        returns.append(ret)
        lengths.append(length)
    return {
        f'target_{target_rew}_return_mean': np.mean(returns),
        f'target_{target_rew}_return_std': np.std(returns),
        f'target_{target_rew}_length_mean': np.mean(lengths),
        f'target_{target_rew}_length_std': np.std(lengths),
    }
    
