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
            state_dim,
            goal_dim,
            act_num,
            goal_input, # True when using goal vector, False when using distance to goal (goal_dim=1)
            hidden_size,
            max_ep_len,
            context_length=None,
            **kwargs
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.act_num = act_num
        self.context_length = context_length  # context length
        self.goal_input = goal_input

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
        
        if self.goal_input:
            self.goal_encoder = GoalEncoder(self.goal_dim, hidden_size)
        else:
            self.distance_to_goal_encoder = DistanceToGoalEncoder(hidden_size)
    
        self.obs_encoder = ObservationEncoder(self.observation_space, hidden_size)
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

        batch_size, seq_length = observations.shape[0], observations.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each input modality with a different head
        observation_embeddings = self.obs_encoder(observations)
        action_embeddings = self.action_encoder(actions)
        if self.goal_input:
            goal_embeddings = self.goal_encoder(goals)
        else:
            goal_embeddings = self.distance_to_goal_encoder(goals)

        time_embeddings = self.timestep_encoder(timesteps)

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

        # note that for a general transformer, input has shape [batch_size, seq_length, input_size], output has shape [batch_size, seq_length, hidden_size]
        # x is the output of the transformer
        # x has shape [batch_size, 3*seq_length, hidden_size]
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original tuple (r,s,a)
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        # before permutation: [batch_size, seq_length, 3, hidden_size]
        # after permutation: [batch_size, 3, seq_length, hidden_size]
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get prediction logits
        # x[:,1] = x[:,1,:,:]
        pred_action_logits = self.action_decoder(x[:,1])  # predict next action given state (policy)

        return pred_action_logits

    # get padding as numpy array
    def get_padding(self, batch_size, padding_length, device):
        # pad observation with 0
        op = torch.zeros((batch_size, padding_length, self.obs_channel, self.obs_height, self.obs_width), device=device)
        # pad action with 0 (stop)
        ap = torch.ones((batch_size, padding_length), device=device)
        # pad goal with 0
        gp = torch.zeros((batch_size, padding_length, self.goal_dim), device=device)
        # pad timestep with 0
        tp = torch.zeros((batch_size, padding_length), device=device)
        # pad mask with 0 (not attend to)
        mp = torch.zeros((1, padding_length), device=device)

        return op, ap, gp, tp, mp


    # input a sequence of (g,s,t) of length context_length
    # only return the last action
    # for evaluation
    def get_action(self, observations, actions, goals, timesteps, sample, **kwargs):
        observations = observations.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1)
        goals = goals.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.context_length is not None:
            observations = observations[:,-self.context_length:]
            actions = actions[:,-self.context_length:]
            goals = goals[:,-self.context_length:]
            timesteps = timesteps[:,-self.context_length:]

            batch_size = observations.shape[0]
            seq_length = observations.shape[1]

            # only attend to the valid part (non padding part)
            # 0 - not attend, 1 - attend
            attention_mask = torch.ones(1, seq_length)
            # left pad all tokens to context length
            op, ap, gp, tp, mp = self.get_padding(batch_size, self.context_length-seq_length, observations.device)
            observations = torch.cat([op, observations], dim=1).to(dtype=torch.float32)  
            actions = torch.cat([ap, actions], dim=1).to(dtype=torch.float32) 
            goals = torch.cat([gp, goals], dim=1).to(dtype=torch.float32)
            timesteps = torch.cat([tp, timesteps], dim=1).to(dtype=torch.long)
            attention_mask = torch.cat([mp, attention_mask], dim=1).to(dtype=torch.long)
        else:
            attention_mask = None

        # forward the sequence with no grad
        with torch.no_grad():
            pred_action_seq_logits = self.forward(
                observations, actions, goals, timesteps, attention_mask=attention_mask, **kwargs)
            
            # pluck the logits at the final step and scale by temperature 1.0
            pred_last_action_logit = pred_action_seq_logits[0,-1] 
            # apply softmax to convert to probabilities
            probs = self.softmax(pred_last_action_logit)
            # greedily pick the action
            #return torch.argmax(probs, dim=1)

            # sample from the distribution or take the most likely
            if sample:
                action = torch.multinomial(probs, num_samples=1)
            else:
                _, action = torch.topk(probs, k=1, dim=-1)
        
        return action
