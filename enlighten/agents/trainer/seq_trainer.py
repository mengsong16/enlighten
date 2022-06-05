import numpy as np
import torch
import time
import random
from torch.nn import functional as F

# train seq2seq imitation learning
class SequenceTrainer():
    def __init__(self, model, optimizer, batch_size, train_dataset, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        
        
        self.scheduler = scheduler
        self.train_dataset = train_dataset

        self.start_time = time.time()

    # train for one iteration
    def train_one_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        # switch model to training mode
        self.model.train()

        # train for num_steps
        for _ in range(num_steps):
            train_loss = self.train_one_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start
        logs['time/total'] = time.time() - self.start_time
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    # train for one step
    def train_one_step(self):
        # observation # (B,K,C,H,W)
        # action # (B,K)
        # goal # (B,K,goal_dim)
        # dtg # (B,K,1)
        # timestep # (B,K)
        # mask # (B,K)
        observations, actions, goals, timesteps, attention_mask = self.train_dataset.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        # [B,K, action_num]
        action_preds = self.model.forward(
            observations, actions, goals, timesteps, attention_mask=attention_mask,
        )

        # loss is computed over the whole sequence (K action tokens)
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1)[attention_mask.reshape(-1) > 0]

        #print(action_preds.size())  #[sum of seq_len, act_num]
        #print(action_target.size()) #[sum seq_len]
        
        # loss is evaluated only on actions
        # action_target are ground truth action indices (not one-hot vectors)
        loss =  F.cross_entropy(action_preds, action_target)

        #print(loss) # a float number

        self.optimizer.zero_grad()
        # compute weight gradients
        loss.backward()
        # clip weight grad
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        # optimize for one step
        self.optimizer.step()

        return loss.detach().cpu().item()
    
    

    
