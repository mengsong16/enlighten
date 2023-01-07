from collections import OrderedDict, namedtuple
from typing import Tuple
import abc
from typing import Iterable

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

from enlighten.agents.common.other import add_prefix, np_to_pytorch_batch, soft_update_from_to, zeros, get_numpy, create_stats_ordered_dict, ones
import gtimer as gt

# 4 losses
SACLosses = namedtuple(
    'SACLosses',
    'policy_loss qf1_loss qf2_loss alpha_loss',
)

class SACTrainer(metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            encoder,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            discount,
            encoder_lr,
            policy_lr,
            qf_lr,
            soft_target_tau,
            target_update_period,
            optimizer_class=optim.Adam,
            target_entropy=None,
            
    ):
        #self._num_train_steps = 0

        self.env = env
        self.encoder = encoder
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        
        if target_entropy is None:
            # continuous actions space: Use heuristic value from SAC paper
            # self.target_entropy = -np.prod(
            #     self.env.action_space.shape).item()

            # discrete actions space
            # 1.386 when there are 4 actions
            self.target_entropy = np.log(self.env.action_space.n)
            # print(-np.log(1/self.env.action_space.n))
        else:
            self.target_entropy = target_entropy

        # Initialize log alpha and alpha
        self.log_alpha = zeros(1, requires_grad=True, torch_device=self.get_device()) # tensor
        self.alpha = self.log_alpha.exp()

        # print(self.get_device())
        # print(self.alpha.device)
        # exit()

        self.alpha_optimizer = optimizer_class(
            [self.log_alpha],
            lr=policy_lr,
        )

        self.qf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.encoder_optimizer = optimizer_class(
            self.encoder.parameters(),
            lr=encoder_lr,
        )

        self.discount = discount
        self._n_train_steps_total = 0

        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()
    
    def get_device(self):
        return self.qf1.get_device()

    def update_actor_parameters(self, losses):
        """Updates the parameters for the actor"""
        self.policy_optimizer.zero_grad()
        losses.policy_loss.backward()
        self.policy_optimizer.step()
    
    def update_alpha_parameters(self, losses):
        """Upadate the log alpha"""
        self.alpha_optimizer.zero_grad()
        losses.alpha_loss.backward()
        self.alpha_optimizer.step()

        # update alpha 
        self.alpha = self.log_alpha.exp()

    def update_critic_parameters(self, losses):
        """Updates the parameters for both critics"""
        # update q1
        self.qf1_optimizer.zero_grad()
        losses.qf1_loss.backward()
        self.qf1_optimizer.step()

        # update q2
        self.qf2_optimizer.zero_grad()
        losses.qf2_loss.backward()
        self.qf2_optimizer.step()

    def train_from_torch(self, batch):
        gt.blank_stamp()

        # forward
        losses, stats = self.compute_loss(
            batch,
            #skip_statistics=not self._need_to_update_eval_statistics,
            skip_statistics=True
        )
        #print(losses)
        
        """
        Update networks
        """
        # clear encoder gradients
        self.encoder_optimizer.zero_grad()

        # update alpha 
        self.update_alpha_parameters(losses)

        # update policy weights
        self.update_actor_parameters(losses)

        # update q weights
        self.update_critic_parameters(losses)

        # update encoder weights
        self.encoder_optimizer.step()

        # update weights of target q (after updating q networks)
        self.try_update_target_networks()

        # update step counter
        self._n_train_steps_total += 1

        print("Done")
        exit()

        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False
        
        gt.stamp('sac training', unique=False)

        

    def try_update_target_networks(self):
        # copy q to target q at the very beginning
        if self._n_train_steps_total % self.target_update_period == 0:
            self.update_target_networks()

    def update_target_networks(self):
        soft_update_from_to(
            self.qf1, self.target_qf1, self.soft_target_tau
        )
        soft_update_from_to(
            self.qf2, self.target_qf2, self.soft_target_tau
        )

    # pi, log_pi: # [B,4]
    def get_policy_distribution(self, obs_embeddings):
        pi = self.policy(obs_embeddings)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = (pi == 0.0)
        z = z.float() * 1e-8
        log_pi = torch.log(pi+z)

        return pi, log_pi

    def calculate_entropy_tuning_loss(self, obs_embeddings):
        """Calculates the loss for the entropy temperature parameter."""
        # alpha_loss is a scalar
        # torch.sum part has shape [B,1]
        pi, log_pi = self.get_policy_distribution(obs_embeddings)
        alpha_loss = -(self.log_alpha * torch.sum(pi.detach() * (log_pi + self.target_entropy).detach(), dim=1, keepdim=True)).mean()

        return alpha_loss
    
    def calculate_actor_loss(self, obs_embeddings):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        # [B,4]
        q = torch.min(
            self.qf1(obs_embeddings),
            self.qf2(obs_embeddings),
        )
        
        # [B,4]
        pi, log_pi = self.get_policy_distribution(obs_embeddings)
        # policy_loss is a scalar
        # torch.sum part has shape [B,1]
        policy_loss = (torch.sum(pi * (self.alpha * log_pi - q), dim=1, keepdim=True)).mean()
        log_pi_pi = torch.sum(log_pi * pi, dim=1)

        return policy_loss, log_pi_pi

    def calculate_critic_losses(self, obs_embeddings, next_obs_embeddings, rewards, actions, dones):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""

        # compute predicted Q
        q1_output = self.qf1(obs_embeddings)
        q2_output = self.qf2(obs_embeddings)
        
        # actions: [B,1]
        # q1_output: [B,4]
        # q1_pred: [B,1]
        q1_pred = torch.gather(q1_output, dim=1, index=actions.long()) # [B,1]
        q2_pred = torch.gather(q2_output, dim=1, index=actions.long()) # [B,1]

        # [B,4]
        next_pi, next_log_pi = self.get_policy_distribution(next_obs_embeddings)

        next_target_q = torch.min(
            self.target_qf1(next_obs_embeddings),
            self.target_qf2(next_obs_embeddings),
        ) # [B,4]

        # [B,1]
        target_v_values = torch.sum(next_pi * (next_target_q - self.alpha * next_log_pi), dim=1, keepdim=True) 
        # [B,1]
        q_target = rewards + (1. - dones) * self.discount * target_v_values

        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        return qf1_loss, qf2_loss

    def compute_loss(
        self,
        batch,
        skip_statistics=False,
    ) -> Tuple[SACLosses, OrderedDict]:
        rewards = batch['rewards']
        dones = batch['dones']
        obs = batch['observations']
        goals = batch['goals']
        actions = batch['actions']
        next_obs = batch['next_observations']
        next_goals = batch['next_goals']

        """
        Encoder forward
        """
        # get input embeddings
        obs_embeddings = self.encoder(obs, goals)
        next_obs_embeddings = self.encoder(next_obs, next_goals)

        """
        Alpha Loss forward
        """
        alpha_loss = self.calculate_entropy_tuning_loss(obs_embeddings)
        
        """
        Policy Loss forward
        """
        policy_loss, log_pi_pi = self.calculate_actor_loss(obs_embeddings)

        """
        QF Loss forward
        """
        qf1_loss, qf2_loss = self.calculate_critic_losses(obs_embeddings, next_obs_embeddings, rewards, actions, dones)

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics['QF1 Loss'] = np.mean(get_numpy(qf1_loss))
            eval_statistics['QF2 Loss'] = np.mean(get_numpy(qf2_loss))
            eval_statistics['Policy Loss'] = np.mean(get_numpy(
                policy_loss
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                get_numpy(q1_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                get_numpy(q2_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                get_numpy(q_target),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Log Pi',
                get_numpy(log_pi),
            ))
            policy_statistics = add_prefix(pi.get_diagnostics(), "policy/")
            eval_statistics.update(policy_statistics)
            
            eval_statistics['Alpha'] = self.alpha.item()
            eval_statistics['Alpha Loss'] = alpha_loss.item()

        loss = SACLosses(
            policy_loss=policy_loss,
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            alpha_loss=alpha_loss,
        )

        return loss, eval_statistics

    # get what to print
    def get_diagnostics(self):
        # stats = OrderedDict([
        #     ('num train calls', self._num_train_steps),
        # ])
        stats = OrderedDict([
            ('num train calls', self._n_train_steps_total),
        ])

        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True
    
    def train(self, np_batch):
        #self._num_train_steps += 1
        tensor_batch = np_to_pytorch_batch(np_batch, self.get_device())
        # for k, np_array in tensor_batch.items():
        #     print("------------------")
        #     print(k)
        #     print(np_array.dtype)
        #     print(np_array.device)
        # print("------------------")
        # print(self.get_device())
        # exit()

        self.train_from_torch(tensor_batch)

    # 6 networks
    @property
    def networks(self):
        return [
            self.encoder,
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    # 5 optimizers
    @property
    def optimizers(self):
        return [
            self.alpha_optimizer,
            self.qf1_optimizer,
            self.qf2_optimizer,
            self.policy_optimizer,
            self.encoder_optimizer
        ]

    # 6 networks
    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            encoder=self.encoder
        )
