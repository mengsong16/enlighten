
from enlighten.utils.path import *
from enlighten.utils.config_utils import parse_config
from enlighten.agents.models.mlp_network import MLPNetwork
from enlighten.agents.models.dt_encoder import ObservationEncoder, GoalEncoder
from enlighten.agents.common.other import get_obs_channel_num, get_device
import torch
import torch.nn as nn


class SimpleMLPPolicy(nn.Module):
    def __init__(
            self,
            act_num,
            input_dim, #512+32
            hidden_size, #512
            hidden_layer, #2
    ):
        super().__init__()

        self.policy_network = MLPNetwork(
                input_dim=input_dim, 
                output_dim=act_num, 
                hidden_dim=hidden_size, 
                hidden_layer=hidden_layer)
        
        
        # acton logits --> action prob
        self.softmax = nn.Softmax(dim=-1)
    
    # for training, return distributions (not logits)
    def forward(self, input_embeddings):  
        # feed the input embeddings into the mlp policy
        # output: [B, act_num]
        pred_action_logits = self.policy_network(input_embeddings)

        # apply softmax to convert to probabilities
        # probs: [B, action_num]
        probs = self.softmax(pred_action_logits)

        return probs
    
class Encoder(nn.Module):
    def __init__(
            self,
            goal_dim,
            obs_channel,
            obs_embedding_size, #512
            goal_embedding_size, #32
    ):
        super().__init__()

        self.goal_encoder = GoalEncoder(goal_dim, goal_embedding_size)
        self.obs_encoder = ObservationEncoder(obs_channel, obs_embedding_size)

    # input: observations: [B, C, H, W]
    #        goals: [B,goal_dim]
    # output: [B, action_number]
    def forward(self, observations, goals):
        # (B,C,H,W) ==> (B,obs_embedding_size)
        observation_embeddings = self.obs_encoder(observations)
        
        # (B,goal_dim) ==> (B,goal_embedding_size)
        goal_embeddings = self.goal_encoder(goals)
        
         # (o,g) ==> [B,input_size]
        input_embeddings = torch.cat([observation_embeddings, goal_embeddings], dim=1)

        return input_embeddings
    
    def to(self, device):
        self.goal_encoder.to(device)
        self.obs_encoder.to(device)
        #self.device = device
    
class SACAgent:
    def __init__(self, config_filename):
        # get config
        if isinstance(config_filename, str):
            config_file = os.path.join(config_path, config_filename)
            self.config = parse_config(config_file)
        else:
            self.config = config_filename
        
        # set device
        self.device = get_device(self.config)

        # create models
        self.create_models()
    
    def create_models(self):
        obs_channel = get_obs_channel_num(self.config)
        self.goal_dim = int(self.config.get("goal_dimension"))
        self.act_num = int(self.config.get("action_number"))
        self.obs_embedding_size = int(self.config.get('obs_embedding_size')) #512
        self.goal_embedding_size = int(self.config.get('goal_embedding_size')) #32
        self.hidden_size = int(self.config.get('hidden_size'))
        self.hidden_layer = int(self.config.get('hidden_layer'))
        
        # shared by the following 5 networks
        self.encoder = Encoder(goal_dim=self.goal_dim,
            obs_channel=obs_channel,
            obs_embedding_size=self.obs_embedding_size,
            goal_embedding_size=self.goal_embedding_size)
        
        # two hidden layer MLPs
        self.qf1 = MLPNetwork(
            input_dim=self.obs_embedding_size+self.goal_embedding_size, 
            output_dim=self.act_num, 
            hidden_dim=self.hidden_size, 
            hidden_layer=self.hidden_layer)
        
        # two hidden layer MLPs
        self.qf2 = MLPNetwork(
            input_dim=self.obs_embedding_size+self.goal_embedding_size, 
            output_dim=self.act_num, 
            hidden_dim=self.hidden_size, 
            hidden_layer=self.hidden_layer)

        # two hidden layer MLPs
        self.target_qf1 = MLPNetwork(
            input_dim=self.obs_embedding_size+self.goal_embedding_size, 
            output_dim=self.act_num, 
            hidden_dim=self.hidden_size, 
            hidden_layer=self.hidden_layer)
        
        # two hidden layer MLPs
        self.target_qf2 = MLPNetwork(
            input_dim=self.obs_embedding_size+self.goal_embedding_size, 
            output_dim=self.act_num, 
            hidden_dim=self.hidden_size, 
            hidden_layer=self.hidden_layer)
        
        # two hidden layer MLPs
        self.policy = SimpleMLPPolicy(
            act_num=self.act_num, 
            input_dim=self.obs_embedding_size+self.goal_embedding_size,
            hidden_size=self.hidden_size, 
            hidden_layer=self.hidden_layer)
        
        # move all networks to the device (must be done before instantiating the algorithm and trainer)
        self.to(self.device)
    
    def to(self, device):
        self.encoder.to(device)
        self.qf1.to(device)
        self.qf2.to(device)
        self.target_qf1.to(device)
        self.target_qf2.to(device)
        self.policy.to(device)

    # for evaluation
    # observations: [B,C,H,W]
    # goals: [B, goal_dim]
    # return actions:[B, 1]
    def get_action(self, observations, goals, sample=True):
        # forward the sequence with no grad
        with torch.no_grad():
            # get input embeddings
            input_embeddings = self.encoder(observations, goals)
            
            # get distributions
            probs = self.policy(input_embeddings)

            # sample from the distribution
            if sample:
                # each row is an independent distribution, draw 1 sample per distribution
                actions = torch.multinomial(probs, num_samples=1)
            # take the most likely action
            else:
                _, actions = torch.topk(probs, k=1, dim=-1)

        return actions
    
    def load(self, checkpoint):
        self.policy.load_state_dict(checkpoint["policy"])
        self.qf1.load_state_dict(checkpoint["qf1"])
        self.qf2.load_state_dict(checkpoint["qf2"])
        self.target_qf1.load_state_dict(checkpoint["target_qf1"])
        self.target_qf2.load_state_dict(checkpoint["target_qf2"])
        self.encoder.load_state_dict(checkpoint["encoder"])