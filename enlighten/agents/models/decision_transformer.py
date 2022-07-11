import numpy as np
import torch
import torch.nn as nn

import transformers

from enlighten.agents.models.gpt2 import GPT2Model
from enlighten.agents.models.dt_encoder import ObservationEncoder, DistanceToGoalEncoder, GoalEncoder, DiscreteActionEncoder, TimestepEncoder, DiscreteActionDecoder

# based on GPT2
class DecisionTransformer(nn.Module):

    """
    This model uses GPT2 model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            obs_channel,
            obs_width,
            obs_height,
            goal_dim,
            goal_form, # ["rel_goal", "distance_to_goal", "abs_goal"]
            act_num,
            hidden_size,
            max_ep_len,
            pad_mode,
            context_length,
            **kwargs
    ):
        super().__init__()
        
        self.obs_channel = obs_channel
        self.obs_width = obs_width
        self.obs_height = obs_height
        self.goal_dim = goal_dim
        self.act_num = act_num
        self.context_length = context_length  # context length
        self.goal_form = goal_form
        self.pad_mode = pad_mode # left or right

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        # four heads for input (training): o,a,g,t
        # timestep is used for positional embedding
        self.timestep_encoder = TimestepEncoder(max_ep_len, hidden_size)
        
        if self.goal_form == "rel_goal":
            self.goal_encoder = GoalEncoder(self.goal_dim, hidden_size)
        elif self.goal_form == "distance_to_goal":
            self.distance_to_goal_encoder = DistanceToGoalEncoder(hidden_size)
        else:
            print("Undefined goal form: %s"%(self.goal_form))
            exit()    
    
        self.obs_encoder = ObservationEncoder(obs_channel, hidden_size)
        self.action_encoder = DiscreteActionEncoder(self.act_num, hidden_size)
       
        # used to embed the concatenated input
        self.concat_embed_ln = nn.LayerNorm(hidden_size)

        # one heads for output (training)
        self.action_decoder = DiscreteActionDecoder(hidden_size, self.act_num)

        # acton logits --> action prob
        self.softmax = nn.Softmax(dim=-1)

    # input: a sequence of (o,a,g,t) of length context_length
    # output: a sequence of predicted (o,a,g) of length context_length
    # for training
    def forward(self, observations, actions, goals, timesteps, attention_mask=None):

        # print(observations.size()) # (B,K,C,H,W)
        # print(actions.size()) # (B,K)
        # print(goals.size()) # (B,K,goal_dim)
        # print(timesteps.size()) # (B,K)
        # print(attention_mask.size()) # (B,K)

        batch_size, seq_length = observations.shape[0], observations.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        #print("===========================")
        # embed each input modality with a different head

        # (B,K,C,H,W) ==> (B*K,C,H,W)
        observation_embeddings = self.obs_encoder(observations.reshape(-1, self.obs_channel, self.obs_height, self.obs_width).type(torch.float32).contiguous())
        # (B*K,C,H,W) ==> (B,K,C,H,W)
        observation_embeddings = observation_embeddings.reshape(batch_size, seq_length, self.hidden_size) 
        
        #print(observation_embeddings.size())
        action_embeddings = self.action_encoder(actions)
        #print(action_embeddings.size())
        

        if self.goal_form == "rel_goal":
            goal_embeddings = self.goal_encoder(goals)
        elif self.goal_form == "distance_to_goal":
            goal_embeddings = self.distance_to_goal_encoder(goals)
        else:
            print("Undefined goal form: %s"%(self.goal_form))
            exit()    

        #print(goal_embeddings.size())
        
        time_embeddings = self.timestep_encoder(timesteps)

        #print(time_embeddings.size())

        # time embeddings are treated similar to positional embeddings
        # append positional embedding to each input modality
        observation_embeddings = observation_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        goal_embeddings = goal_embeddings + time_embeddings

        # this makes the sequence look like (g_1, o_1, a_1, g_2, o_2, a_2, ...)
        # which works nice in an autoregressive sense since observations predict actions
        # stack (g,o,a) for each step
        # before permutation: [batch_size, 3, seq_length, hidden_size] (dim 1 is a new dim)
        # after permutation: [batch_size, seq_length, 3, hidden_size]
        # after reshape: sequence length becomes 3*seq_length
        stacked_inputs = torch.stack(
            (goal_embeddings, observation_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        
        # embed the concatenated input
        stacked_inputs = self.concat_embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the transformer
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )

        # note that for a general transformer, input has shape [batch_size, seq_length, input_size]
        # output has shape [batch_size, seq_length, hidden_size]
        # x is the output of the transformer, hidden-states of the model at the output of the last layer 
        # x has shape [batch_size, 3*seq_length, hidden_size]
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original tuple (g,o,a)
        # after reshape: [batch_size, 3, seq_length, hidden_size]
        # returns goals (0), observations (1) and actions (2)
        # before permutation: [batch_size, seq_length, 3, hidden_size]
        # after permutation: [batch_size, 3, seq_length, hidden_size]
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        #print(x.shape)

        # get prediction logits from observations
        # x[:,1] = x[:,1,:,:] = sequence of observations
        # i.e. x[:,1,:,:] is the embedding for o_t
        pred_action_logits = self.action_decoder(x[:,1])  # predict next action given state (policy)
        
        #print(pred_action_logits.size()) # [batch_size, seq_length, action_num]

        return pred_action_logits

    # get padding as numpy array
    # called by get_action
    # padding_length >= 0
    # if padding_length = 0, return 0 size tensor
    def get_padding(self, batch_size, padding_length, device):
        # pad observation with 0
        op = torch.zeros((batch_size, padding_length, self.obs_channel, self.obs_height, self.obs_width), device=device)
        # pad action with 0 (stop)
        ap = torch.zeros((batch_size, padding_length), device=device)
        # pad goal with 0
        if self.goal_form == "rel_goal":
            gp = torch.zeros((batch_size, padding_length, self.goal_dim), device=device)
        elif self.goal_form == "distance_to_goal":
            gp = torch.zeros((batch_size, padding_length, 1), device=device)
        else:
            print("Undefined goal form: %s"%(self.goal_form))
            exit()    
        # pad timestep with 0
        tp = torch.zeros((batch_size, padding_length), device=device)
        # pad mask with 0 (not attend to)
        mp = torch.zeros((1, padding_length), device=device)

        return op, ap, gp, tp, mp


    # input a sequence of (g,s,t) of length context_length K
    # only return the last action
    # for evaluation
    def get_action(self, observations, actions, goals, timesteps, sample, **kwargs):
        # print(observations.size()) # (B,K,C,H,W)
        # print(actions.size()) # (B,K)
        # print(goals.size()) # (B,K,goal_dim) or (B,K,1)
        # print(timesteps.size()) # (B,K)
        # print(attention_mask.size()) # (B,K)
        
        # B = 1
        observations = observations.reshape(1, -1, self.obs_channel, self.obs_height, self.obs_width)
        actions = actions.reshape(1, -1)
        if self.goal_form == "rel_goal":
            goals = goals.reshape(1, -1, self.goal_dim)
        else:    
            goals = goals.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)


        # create attention mask according to the input
        if self.context_length is not None:
            # crop context_length sequence starting from the rightmost
            observations = observations[:,-self.context_length:]
            actions = actions[:,-self.context_length:]
            goals = goals[:,-self.context_length:]
            timesteps = timesteps[:,-self.context_length:]

            batch_size = observations.shape[0]
            seq_length = observations.shape[1] # already <=K

            # only attend to the valid part (non padding part)
            # 0 - not attend, 1 - attend
            attention_mask = torch.ones((1, seq_length), device=observations.device)
            # pad all tokens to context length
            op, ap, gp, tp, mp = self.get_padding(batch_size, self.context_length-seq_length, observations.device)
            
            # left padding
            if self.pad_mode == "left":
                observations = torch.cat([op, observations], dim=1).to(dtype=torch.float32)  
                actions = torch.cat([ap, actions], dim=1).to(dtype=torch.long) 
                goals = torch.cat([gp, goals], dim=1).to(dtype=torch.float32)
                timesteps = torch.cat([tp, timesteps], dim=1).to(dtype=torch.long)
                attention_mask = torch.cat([mp, attention_mask], dim=1).to(dtype=torch.long)
            # right padding
            elif self.pad_mode == "right": 
                observations = torch.cat([observations, op], dim=1).to(dtype=torch.float32)  
                actions = torch.cat([actions, ap], dim=1).to(dtype=torch.long) 
                goals = torch.cat([goals, gp], dim=1).to(dtype=torch.float32)
                timesteps = torch.cat([timesteps, tp], dim=1).to(dtype=torch.long)
                attention_mask = torch.cat([attention_mask, mp], dim=1).to(dtype=torch.long)
            else:
                print("Error: undefined padding mode: %s"%(self.pad_mode))
                exit()
        else:
            attention_mask = None

        # forward the sequence with no grad
        with torch.no_grad():
            pred_action_seq_logits = self.forward(
                observations, actions, goals, timesteps, attention_mask=attention_mask, **kwargs)
            
            # pluck the logits at the final step (rightmost)
            # [B,K,act_num]
            pred_last_action_logits = pred_action_seq_logits[:,-1,:] 

            #print("============================")
            #print(pred_last_action_logit.size())
            
            # apply softmax to convert to probabilities
            probs = self.softmax(pred_last_action_logits)
            # greedily pick the action
            #return torch.argmax(probs, dim=1)

            # sample from the distribution or take the most likely
            if sample:
                action = torch.multinomial(probs, num_samples=1)
            else:
                _, action = torch.topk(probs, k=1, dim=-1)
        
        return action
