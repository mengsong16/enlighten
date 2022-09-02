import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

from enlighten.agents.models.dt_encoder import ObservationEncoder, DistanceToGoalEncoder, GoalEncoder, DiscreteActionEncoder, ValueDecoder, DiscreteActionDecoder, BinaryDiscriminator, AdversarialLayer


class RNNModel(nn.Module):
    def __init__(self, rnn_type, rnn_input_size, rnn_hidden_size):
        super().__init__()

        # create model
        if rnn_type == "gru":
            self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=1,
            batch_first=False)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=1,
            batch_first=False)
        else:
            raise RuntimeError(f"Did not recognize rnn type '{rnn_type}'")
 
        # initialize model weights
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
        
        self.rnn_type = rnn_type


    # input x: [1, B, input_size] (when batch_first=False)
    # input hidden_states (h_{t-1}): [1, B, hidden_size]
    # output next_hidden_states (h_{t}): [1, B, hidden_size] 
    def single_forward(self, x, hidden_states):
        r"""Forward a single step"""

        h_seq, h_n = self.rnn(
            x, self.unpack_hidden(hidden_states)
        )

        next_hidden_states = self.pack_hidden(h_n)

        return next_hidden_states

    # input x: 
    # 1. PackedSequence of [T, input_size] for variable lengths
    # 2. [T, B, input_size] for equal lengths (when batch_first=False) 
    # input h_0: [1, B, hidden_size]
    # output h_seq: 
    # 1. PackedSequence of [T, hidden_size] for variable lengths
    # 2. [T, B, input_size] for equal lengths (when batch_first=False)
    # output h_n: [1, B, hidden_size]
    def seq_forward(self, x, h_0, batch_sizes):
        r"""Forward for a sequence"""

        x_packed = PackedSequence(x, batch_sizes, None, None)

        h_seq, h_n = self.rnn(
            x_packed, self.unpack_hidden(h_0)
        )

        h_n = self.pack_hidden(h_n)

        return h_seq, h_n
     
    def pack_hidden(self, hidden_states):
        if self.rnn_type == "lstm":
            # hidden_states:  --> [2, N, hidden_size]
            # h: [1, N, hidden_size]
            # c: [1, N, hidden_size]
            # concatenate along dim 0
            return torch.cat(hidden_states, 0)
        else:
            return hidden_states

    def unpack_hidden(self, hidden_states):
        if self.rnn_type == "lstm":
            # hidden_states: [2, N, hidden_size] --> (h,c)
            # h: [1, N, hidden_size]
            # c: [1, N, hidden_size] 
            # split a tensor into the specified number of chunks
            lstm_states = torch.chunk(hidden_states, 2, 0)
            return (lstm_states[0], lstm_states[1])
        else:
            return hidden_states

    # hidden_states: [N, 2, hidden_size]
    # h: [N, 1, hidden_size]
    def extract_h(self, hidden_states):
        lstm_states = torch.chunk(hidden_states, 2, 1)
        return lstm_states[0]


class RNNSequenceModel(nn.Module):

    def __init__(
            self,
            obs_channel,
            obs_width,
            obs_height,
            goal_dim,
            goal_form, # ["rel_goal", "distance_to_goal", "abs_goal"]
            act_num,
            max_ep_len,
            obs_embedding_size, #512
            goal_embedding_size, #32
            act_embedding_size, #32
            rnn_hidden_size, #512
            rnn_type,
            supervise_value,
            domain_adaptation
    ):
        super().__init__()
        
        self.obs_channel = obs_channel
        self.obs_width = obs_width
        self.obs_height = obs_height
        self.goal_dim = goal_dim
        self.act_num = act_num
        self.goal_form = goal_form
        self.max_ep_len = max_ep_len
        
        self.rnn_hidden_size = rnn_hidden_size
        self.obs_embedding_size = obs_embedding_size
        self.goal_embedding_size = goal_embedding_size
        self.act_embedding_size = act_embedding_size
        rnn_input_size = obs_embedding_size + goal_embedding_size + act_embedding_size

        
        # three heads for input (training): o,a,g
        if self.goal_form == "rel_goal" or self.goal_form == "abs_goal":
            self.goal_encoder = GoalEncoder(self.goal_dim, goal_embedding_size)
        elif self.goal_form == "distance_to_goal":
            self.distance_to_goal_encoder = DistanceToGoalEncoder(goal_embedding_size)
        else:
            print("Undefined goal form: %s"%(self.goal_form))
            exit()    
    
        self.obs_encoder = ObservationEncoder(obs_channel, obs_embedding_size)
        # action=0 is a place holder for a_0 = -1
        # input vector has size as action number + 1 
        self.action_encoder = DiscreteActionEncoder(self.act_num+1, act_embedding_size)
       
        self.rnn = RNNModel(rnn_type, rnn_input_size, rnn_hidden_size)
        
        # policy head for output, output vector has size as action number
        self.action_decoder = DiscreteActionDecoder(rnn_hidden_size, self.act_num)

        # acton logits --> action prob
        self.softmax = nn.Softmax(dim=-1)

        # value head for output (optional)
        self.supervise_value = supervise_value
        if self.supervise_value:
            self.value_decoder = ValueDecoder(rnn_hidden_size)
        
        # domain adaptation
        self.domain_adaptation = domain_adaptation
        if self.domain_adaptation:
            self.adversarial_discriminator = BinaryDiscriminator(obs_embedding_size)

        # gradient reversal layer
        self.grl = AdversarialLayer()

    def encoder_forward(self, observations, prev_actions, goals):
        # (T,C,H,W) ==> (T,obs_embedding_size)
        observation_embeddings = self.obs_encoder(observations)
        # (T) ==> (T,act_embedding_size)
        # input action + 1
        prev_action_embeddings = self.action_encoder(prev_actions+1)
        # (T,goal_dim) ==> (T,goal_embedding_size)
        if self.goal_form == "rel_goal" or self.goal_form == "abs_goal":
            goal_embeddings = self.goal_encoder(goals)
        elif self.goal_form == "distance_to_goal":
            goal_embeddings = self.distance_to_goal_encoder(goals)
        else:
            print("Undefined goal form: %s"%(self.goal_form))
            exit()    

        
        if self.domain_adaptation:
            # (o[:source],g,a) ==> [T,rnn_input_size]
            source_batch_size = prev_action_embeddings.size(0)
            input_embeddings = torch.cat([observation_embeddings[:source_batch_size], goal_embeddings, prev_action_embeddings], dim=1)

            return input_embeddings, observation_embeddings
        else:
            # (o,g,a) ==> [T,rnn_input_size]
            input_embeddings = torch.cat([observation_embeddings, goal_embeddings, prev_action_embeddings], dim=1)

            return input_embeddings


    # input: B sequence of (o,a,g) of variant lengths, T steps in total
    # input: h_0: [1, B, hidden_size]
    # input: batch_sizes: batch_size of each step in the longest sequence
    # output: B sequence of pred_action_logits of variant lengths, T steps in total
    # for training
    def forward(self, observations, prev_actions, goals, h_0, batch_sizes):

        # print(observations.size()) # (T,C,H,W)
        # print(prev_actions.size()) # (T)
        # print(goals.size()) # (T,goal_dim)

        # embed each input modality with a different head
        if self.domain_adaptation:
            input_embeddings, observation_embeddings = self.encoder_forward(observations, prev_actions, goals)
            da_logits = self.adversarial_discriminator(self.grl.apply(observation_embeddings))
        else:
            input_embeddings = self.encoder_forward(observations, prev_actions, goals)
        
        # feed the input embeddings into the rnn
        # h_seq.data: [T, hidden_size]
        # h_n: [1, B, hidden_size]
        h_seq, h_n = self.rnn.seq_forward(x=input_embeddings, h_0=h_0, batch_sizes=batch_sizes)


        # pred_action_logits: [T, action_num]
        pred_action_logits = self.action_decoder(h_seq.data)

        # pred_values: [T,1]
        if self.supervise_value:
            pred_values = self.value_decoder(h_seq.data)
        
        if self.supervise_value == True and self.domain_adaptation == True:
            return pred_action_logits, pred_values, da_logits
        elif self.supervise_value == False and self.domain_adaptation == True:
            return pred_action_logits, da_logits
        elif self.supervise_value == True and self.domain_adaptation == False:
            return pred_action_logits, pred_values
        else:
            return pred_action_logits

    # input: observations: [B, C, H, W]
    #        prev_actions: [B]
    #        goals: [B,goal_dim]
    #        h_{t-1}: [1, B, hidden_size]
    # output: [B,1] actions, h_t 
    # B could be 1 or the number of vector environments
    # for evaluation
    def get_action(self, observations, prev_actions, goals, h_prev, sample):
        # forward the sequence with no grad
        with torch.no_grad():
            # embed each input modality with a different head
            if self.domain_adaptation:
                input_embeddings, _ = self.encoder_forward(observations, prev_actions, goals)
            else:
                input_embeddings = self.encoder_forward(observations, prev_actions, goals)

            # input_embeddings: [B, input_size] --> [1, B, input_size]
            input_embeddings = torch.unsqueeze(input_embeddings, 0)
            # h_cur: [1, B, hidden_size] 
            h_cur = self.rnn.single_forward(
                input_embeddings, h_prev)
            
            # h_cur: [1, B, hidden_size] 
            # pred_action_logits: [B, action_num]
            # h_cur': [B, hidden_size]
            pred_action_logits = self.action_decoder(torch.squeeze(h_cur, 0))

            #print("=========")
            #print(pred_action_logits.size())
            
            # apply softmax to convert to probabilities
            # probs: [B, action_num]
            probs = self.softmax(pred_action_logits)

            #print(probs.size())
            
            
            # sample from the distribution or take the most likely
            if sample:
                # each row is an independent distribution, draw 1 sample per distribution
                actions = torch.multinomial(probs, num_samples=1)
            else:
                _, actions = torch.topk(probs, k=1, dim=-1)
            
            #print(actions.size())
            #print("=========")
        
        # actions: [B, 1]
        # h_cur: [1, B, hidden_size]
        
        return actions, h_cur

# ddbc
class _GetActionWrapper(torch.nn.Module):
    r"""Wrapper on get_action that allows that to be called from forward.
    This is needed to interface with DistributedDataParallel's forward call
    """

    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, *args, **kwargs):
        return self.actor.get_action(*args, **kwargs)


class DDBC(nn.Module):
    def __init__(
            self,
            obs_channel,
            obs_width,
            obs_height,
            goal_dim,
            goal_form, # ["rel_goal", "distance_to_goal", "abs_goal"]
            act_num,
            max_ep_len,
            obs_embedding_size, #512
            goal_embedding_size, #32
            act_embedding_size, #32
            rnn_hidden_size, #512
            rnn_type,
            supervise_value,
            device
    ):
        super().__init__()

        self.actor = RNNSequenceModel(obs_channel,
            obs_width,
            obs_height,
            goal_dim,
            goal_form, # ["rel_goal", "distance_to_goal", "abs_goal"]
            act_num,
            max_ep_len,
            obs_embedding_size, #512
            goal_embedding_size, #32
            act_embedding_size, #32
            rnn_hidden_size, #512
            rnn_type,
            supervise_value,
            domain_adaptation=False)
        
        self.device = device
        self.actor.to(self.device)

    def get_action(self, observations, prev_actions, goals, h_prev, sample):
        return self.actor.get_action(observations, prev_actions, goals, h_prev, sample)

    def init_distributed(self, find_unused_params: bool = True) -> None:
        r"""Initializes distributed training for the model

        1. Broadcasts the model weights from world_rank 0 to all other workers
        2. Adds gradient hooks to the model

        :param find_unused_params: Whether or not to filter out unused parameters
                                   before gradient reduction.  This *must* be True if
                                   there are any parameters in the model that where unused in the
                                   forward pass, otherwise the gradient reduction
                                   will not work correctly.
        """
        # NB: Used to hide the hooks from the nn.Module,
        # so they don't show up in the state_dict
        class Guard:  # noqa: SIM119
            def __init__(self, model, device):
                if torch.cuda.is_available():
                    self.ddp = torch.nn.parallel.DistributedDataParallel(
                        model,
                        device_ids=[device],
                        output_device=device,
                        find_unused_parameters=find_unused_params,
                    )
                else:
                    self.ddp = torch.nn.parallel.DistributedDataParallel(
                        model,
                        find_unused_parameters=find_unused_params,
                    )

        self._get_action_wrapper = Guard(_GetActionWrapper(self.actor), self.device)  # type: ignore

    def _get_action(
        self, observations, prev_actions, goals, h_prev, sample
    ):
        r"""Internal method that calls Policy.evaluate_actions.  This is used instead of calling
        that directly so that that call can be overrided with inheritance
        """
        return self.get_action_wrapper.ddp(
            observations, prev_actions, goals, h_prev, sample
        )