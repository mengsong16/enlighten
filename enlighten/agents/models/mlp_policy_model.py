import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

from enlighten.agents.models.dt_encoder import ObservationEncoder, DistanceToGoalEncoder, GoalEncoder, DiscreteActionEncoder, ValueDecoder, DiscreteActionDecoder, BinaryDiscriminator, AdversarialLayer
from enlighten.agents.models.mlp_encoder import MLPEncoder


class MLPPolicy(nn.Module):

    def __init__(
            self,
            obs_channel,
            obs_width,
            obs_height,
            goal_dim,
            goal_form, # ["rel_goal", "distance_to_goal", "abs_goal"]
            act_num,
            obs_embedding_size, #512
            goal_embedding_size, #32
            hidden_size, #512
            hidden_layer #2
    ):
        super().__init__()
        
        self.obs_channel = obs_channel
        self.obs_width = obs_width
        self.obs_height = obs_height
        self.goal_dim = goal_dim
        self.act_num = act_num
        self.goal_form = goal_form
        
        
        self.hidden_size = hidden_size
        self.obs_embedding_size = obs_embedding_size
        self.goal_embedding_size = goal_embedding_size
        self.hidden_layer = hidden_layer
        
        # three heads for input (training): o,a,g
        if self.goal_form == "rel_goal" or self.goal_form == "abs_goal":
            self.goal_encoder = GoalEncoder(self.goal_dim, goal_embedding_size)
        elif self.goal_form == "distance_to_goal":
            self.distance_to_goal_encoder = DistanceToGoalEncoder(goal_embedding_size)
        else:
            print("Undefined goal form: %s"%(self.goal_form))
            exit()    
    
        self.obs_encoder = ObservationEncoder(obs_channel, obs_embedding_size)
        
        self.policy = MLPEncoder(input_dim=self.obs_embedding_size+self.goal_embedding_size, 
            output_dim=self.act_num, hidden_dim=self.hidden_size, hidden_layer=self.hidden_layer)
        
        # acton logits --> action prob
        self.softmax = nn.Softmax(dim=-1)


    def encoder_forward(self, observations, goals):
        # (T,C,H,W) ==> (T,obs_embedding_size)
        observation_embeddings = self.obs_encoder(observations)
        
        # (T,goal_dim) ==> (T,goal_embedding_size)
        if self.goal_form == "rel_goal" or self.goal_form == "abs_goal":
            goal_embeddings = self.goal_encoder(goals)
        elif self.goal_form == "distance_to_goal":
            goal_embeddings = self.distance_to_goal_encoder(goals)
        else:
            print("Undefined goal form: %s"%(self.goal_form))
            exit()    

        
        # (o,g) ==> [T,input_size]
        input_embeddings = torch.cat([observation_embeddings, goal_embeddings], dim=1)

        return input_embeddings


    # input: B sequence of (o,a,g) of variant lengths, T steps in total
    # input: h_0: [1, B, hidden_size]
    # input: batch_sizes: batch_size of each step in the longest sequence
    # output: B sequence of pred_action_logits of variant lengths, T steps in total
    # for training
    def forward(self, observations, goals):

        # print(observations.size()) # (T,C,H,W)
        # print(goals.size()) # (T,goal_dim)

        # embed each input modality with a different head
        input_embeddings = self.encoder_forward(observations, goals)
        
        # feed the input embeddings into the mlp policy
        # output: [B, act_num]
        pred_action_logits = self.policy(input_embeddings)

        return pred_action_logits

    # input: observations: [B, C, H, W]
    #        goals: [B,goal_dim]
    # output: actions: [B,1]  
    # for evaluation
    def get_action(self, observations, goals, sample):
        # forward the sequence with no grad
        with torch.no_grad():
            # embed each input modality with a different head
            pred_action_logits = self.forward(observations, goals)

            # apply softmax to convert to probabilities
            # probs: [B, action_num]
            probs = self.softmax(pred_action_logits)

            
            # sample from the distribution or take the most likely
            if sample:
                # each row is an independent distribution, draw 1 sample per distribution
                actions = torch.multinomial(probs, num_samples=1)
            else:
                _, actions = torch.topk(probs, k=1, dim=-1)
            
            #print(actions.size())
            #print("=========")
        
        return actions