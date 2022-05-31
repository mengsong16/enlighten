import numpy as np
import torch
import time
import random
from torch.nn import functional as F

# train seq2seq imitation learning
class SequenceTrainer():
    def __init__(self, model, optimizer, batch_size, train_dataset, scheduler=None, eval_fns=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        
        
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
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

        eval_start = time.time()

        # switch model to evaluation mode
        self.model.eval()
        
        # evaluate by each evaluation function
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    # train for one step
    def train_one_step(self):
        observations, actions, goals, timesteps, attention_mask = self.train_dataset.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        
        action_preds = self.model.forward(
            observations, actions, goals, timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        # loss is evaluated only on actions
        # action_target are ground truth action indices (not one-hot vectors)
        loss =  F.cross_entropy(action_preds, action_target)

        self.optimizer.zero_grad()
        # compute weight gradients
        loss.backward()
        # clip weight grad
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        # optimize for one step
        self.optimizer.step()

        # compute action prediction error
        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()
    
    

    
