import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

from enlighten.agents.models.dt_encoder import ObservationEncoder, DistanceToGoalEncoder, GoalEncoder, DiscreteActionEncoder, ValueDecoder, DiscreteActionDecoder, BinaryDiscriminator, AdversarialLayer
from enlighten.agents.models.mlp_network import MLPNetwork


class QNetwork(nn.Module):

    def __init__(
            self,
            obs_channel,
            obs_width,
            obs_height,
            goal_dim, # 2
            goal_form, # ["rel_goal", "distance_to_goal", "abs_goal"]
            act_num,
            obs_embedding_size, #512
            goal_embedding_size, #32
            hidden_size, #512
            hidden_layer, #2
            state_form,
            state_dimension, #2
            policy_type,
            greedy_policy,
            temperature,
            prob_convert_method

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
    
        self.state_form = state_form
        self.state_dimension = state_dimension
        if self.state_form == "observation":
            self.obs_encoder = ObservationEncoder(obs_channel, obs_embedding_size)
            
            self.q_network = MLPNetwork(input_dim=self.obs_embedding_size+self.goal_embedding_size, 
                output_dim=self.act_num, hidden_dim=self.hidden_size, hidden_layer=self.hidden_layer)
        else:
            self.q_network = MLPNetwork(input_dim=self.state_dimension+self.goal_dim, 
                output_dim=self.act_num, hidden_dim=self.hidden_size, hidden_layer=self.hidden_layer)
        
        self.policy_type = policy_type
        self.temperature = temperature
        self.prob_convert_method = prob_convert_method

        assert self.policy_type in ["max_q", "boltzmann"], "Unknown policy type: %s"%self.policy_type
        if self.policy_type == "max_q":
            print("=========> Q network uses max Q policy")
        elif self.policy_type == "boltzmann":
            print("=========> Q network uses Boltzmann policy")
            print("=========> Q to probability convert method: %s"%(self.prob_convert_method))
            if self.prob_convert_method == "softmax":
                print("=========> Policy temperature: %f"%(self.temperature))

        self.greedy_policy = greedy_policy
        if self.greedy_policy:
            print("=========> Use greedy Q policy")
        else:
            print("=========> Samle from Q policy")
        
            
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


    # input: observations: [B, C, H, W]
    #        goals: [B,goal_dim]
    # output: [B, action_number]
    # for training
    def forward(self, observations, goals):

        # print(observations.size()) # (T,C,H,W)
        # print(goals.size()) # (T,goal_dim)

        if self.state_form == "observation":
            # embed each input modality with a different head
            input_embeddings = self.encoder_forward(observations, goals)
            
            # feed the input embeddings into the mlp q function
            # output: [B, act_num]
            q_values = self.q_network(input_embeddings)
        else:
            concat_inputs = torch.cat((observations, goals), dim=1)
            q_values = self.q_network(concat_inputs)

        return q_values

    # input: observations: [B, C, H, W]
    #        goals: [B,goal_dim]
    # output: actions: [B,1]  
    # for evaluation
    def get_action(self, observations, goals):
        # forward the sequence with no grad
        with torch.no_grad():
            # embed each input modality with a different head
            q_values = self.forward(observations, goals)
            # greedy policy = arg max Q
            if self.policy_type == "max_q": # must be greedy
                # If there are multiple maximal values then the indices of the first maximal value are returned
                actions = torch.argmax(q_values, dim=1, keepdim=True)
            # Boltzmann policy = softmax (Q / temperature)
            else:
                if self.prob_convert_method == "softmax":
                    q_values = q_values / self.temperature
                    probs = self.softmax(q_values)
                elif self.prob_convert_method == "normalize":
                    min_q = torch.min(q_values)
                    normalize_q = q_values - min_q
                    sum_q = torch.sum(normalize_q)
                    probs = normalize_q  / sum_q
                else:
                    print("Error: unknown q to prob convert method: %s"%self.prob_convert_method)
                    exit()


                if self.greedy_policy:
                    # get max prob action
                    _, actions = torch.topk(probs, k=1, dim=-1)
                else:
                    # sample action according to the policy distribution
                    actions = torch.multinomial(probs, num_samples=1)
            
        
        return actions