import numpy as np
import torch
import torch.nn as nn

import transformers

from enlighten.agents.models.gpt2 import GPT2Model
from enlighten.agents.models.dt_encoder import ObservationEncoder, ReturnToGoEncoder, GoalEncoder, ActionEncoder, TimestepEncoder, DiscreteActionDecoder

# based on GPT2
class DecisionTransformer(nn.Module):

    """
    This model uses GPT2 model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            **kwargs
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length  # context length

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        # four heads for input (training)
        # timestep is used as positional embedding
        self.timestep_encoder = TimestepEncoder(max_ep_len, hidden_size)
        self.return_to_go_encoder = ReturnToGoEncoder(hidden_size)
        self.obs_encoder = ObservationEncoder(self.observation_space, hidden_size)
        self.action_decoder = ActionEncoder(self.act_dim, hidden_size)
        
        # used to embed the concatenated input
        self.concat_embed_ln = nn.LayerNorm(hidden_size)

        # one heads for output (training)
        self.action_decoder = DiscreteActionDecoder(hidden_size, self.act_dim)

        # acton logits --> action prob
        self.softmax = nn.Softmax(dim=-1)

    # input: a sequence of (s,a,r,t) of length max_length
    # output: a sequence of predicted (s,a,r) of length max_length
    # for training
    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each input modality with a different head
        state_embeddings = self.obs_encoder(states)
        action_embeddings = self.action_encoder(actions)
        returns_embeddings = self.return_to_go_encoder(returns_to_go)
        time_embeddings = self.timestep_encoder(timesteps)

        # time embeddings are treated similar to positional embeddings
        # append positional embedding to each input modality
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        # stack (r,s,a) for each step
        # before permutation: [batch_size, 3, seq_length, hidden_size] (dim 1 is a new dim)
        # after permutation: [batch_size, seq_length, 3, hidden_size]
        # after reshape: sequence length becomes 3*seq_length
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
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

    # input a sequence of (r,s,t) of length max_length
    # only return the last action
    # for evaluation
    def get_action(self, states, actions, returns_to_go, timesteps, sample, **kwargs):

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # only attend to the valid part (non padding part)
            # 0 - not attend, 1 - attend
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            # pre-pad all tokens to sequence length
            # pad state with 0 if shorter than max_length
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            # pad action with 0 if shorter than max_length    
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            # pad rtg with 0 if shorter than max_length    
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            # pad timestep with 0 if shorter than max_length
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        # forward the sequence with no grad
        with torch.no_grad():
            pred_action_seq_logits = self.forward(
                states, actions, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)
            
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
