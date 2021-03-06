#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

from enlighten.agents.models import Attention

import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence


def _invert_permutation(permutation: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(permutation)
    output.scatter_(
        0,
        permutation,
        torch.arange(0, permutation.numel(), device=permutation.device),
    )
    return output

#dones: [T*N, 1]
def _build_pack_info_from_dones(
    dones: torch.Tensor,
    T: int,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    r"""Create the indexing info needed to make the PackedSequence
    based on the dones.

    PackedSequences are PyTorch's way of supporting a single RNN forward
    call where each input in the batch can have an arbitrary sequence length

    They work as follows: Given the sequences [c], [x, y, z], [a, b],
    we generate data [x, a, c, y, b, z] and batch_sizes [3, 2, 1].  The
    data is a flattened out version of the input sequences (the ordering in
    data is determined by sequence length).  batch_sizes tells you that
    for each index, how many sequences have a length of (index + 1) or greater.

    This method will generate the new index ordering such that you can
    construct the data for a PackedSequence from a (T*N, ...) tensor
    via x.index_select(0, select_inds)
    """

    dones = dones.view(T, -1)
    N = dones.size(1)

    rollout_boundaries = dones.clone().detach()
    # Force a rollout boundary for t=0.  We will use the
    # original dones for masking later, so this is fine
    # and simplifies logic considerably
    rollout_boundaries[0] = True
    rollout_boundaries = rollout_boundaries.nonzero(as_tuple=False)

    # The rollout_boundaries[:, 0]*N will make the episode_starts index into
    # the T*N flattened tensors
    episode_starts = rollout_boundaries[:, 0] * N + rollout_boundaries[:, 1]

    # We need to create a transposed start indexing so we can compute episode lengths
    # As if we make the starts index into a N*T tensor, then starts[1] - starts[0]
    # will compute the length of the 0th episode
    episode_starts_transposed = (
        rollout_boundaries[:, 1] * T + rollout_boundaries[:, 0]
    )
    # Need to sort so the above logic is correct
    episode_starts_transposed, sorted_indices = torch.sort(
        episode_starts_transposed, descending=False
    )

    # Calculate length of episode rollouts
    rollout_lengths = (
        episode_starts_transposed[1:] - episode_starts_transposed[:-1]
    )
    last_len = N * T - episode_starts_transposed[-1]
    rollout_lengths = torch.cat([rollout_lengths, last_len.unsqueeze(0)])
    # Undo the sort above
    rollout_lengths = rollout_lengths.index_select(
        0, _invert_permutation(sorted_indices)
    )

    # Resort in descending order of episode length
    lengths, sorted_indices = torch.sort(rollout_lengths, descending=True)

    # We will want these on the CPU for torch.unique_consecutive,
    # so move now.
    cpu_lengths = lengths.to(device="cpu", non_blocking=True)

    episode_starts = episode_starts.index_select(0, sorted_indices)
    select_inds = torch.empty((T * N), device=dones.device, dtype=torch.int64)

    max_length = int(cpu_lengths[0].item())
    # batch_sizes is *always* on the CPU
    batch_sizes = torch.empty((max_length,), device="cpu", dtype=torch.long)

    offset = 0
    prev_len = 0
    num_valid_for_length = lengths.size(0)

    unique_lengths = torch.unique_consecutive(cpu_lengths)
    # Iterate over all unique lengths in reverse as they sorted
    # in decreasing order
    for next_len in reversed(unique_lengths):
        valids = lengths[0:num_valid_for_length] > prev_len
        num_valid_for_length = int(valids.float().sum())

        batch_sizes[prev_len:next_len] = num_valid_for_length

        # Creates this array
        # [step * N + start for step in range(prev_len, next_len)
        #                   for start in episode_starts[0:num_valid_for_length]
        # * N because each step is separated by N elements
        new_inds = (
            torch.arange(
                prev_len, next_len, device=episode_starts.device
            ).view(next_len - prev_len, 1)
            * N
            + episode_starts[0:num_valid_for_length].view(
                1, num_valid_for_length
            )
        ).view(-1)

        select_inds[offset : offset + new_inds.numel()] = new_inds

        offset += new_inds.numel()

        prev_len = next_len

    # Make sure we have an index for all elements
    assert offset == T * N

    # This is used in conjunction with episode_starts to get
    # the RNN hidden states
    rnn_state_batch_inds = episode_starts % N
    # This indicates that a given episode is the last one
    # in that rollout.  In other words, there are N places
    # where this is True, and for each n, True indicates
    # that this episode is the last contiguous block of experience,
    # This is needed for getting the correct hidden states after
    # the RNN forward pass
    episode_index = torch.div(episode_starts + (lengths - 1) * N, N, rounding_mode='trunc')
    last_episode_in_batch_mask = (
        episode_index
    ) == (T - 1)

    # last_episode_in_batch_mask = (
    #     (episode_starts + (lengths - 1) * N) // N
    # ) == (T - 1)

    return (
        select_inds,
        batch_sizes,
        episode_starts,
        rnn_state_batch_inds,
        last_episode_in_batch_mask,
    )

def build_packed_input(x: torch.Tensor, select_inds: torch.Tensor, batch_sizes: torch.Tensor)-> PackedSequence:
    select_inds = select_inds.to(device=x.device)
    
    # print("========================")
    # print("x: %s"%(x))
    # print("Input x of packedseq: %s"%(x.index_select(0, select_inds)))
    # print("Input batch_sizes of packedseq: %s"%(batch_sizes))
    # print("Select inds: %s"%(select_inds))
    # print("========================")
    x_seq = PackedSequence(
        x.index_select(0, select_inds), batch_sizes, None, None
    )
    
    return x_seq
    

# x: [T*N, input_size] 
def build_rnn_inputs(
    x: torch.Tensor, not_dones: torch.Tensor, rnn_states: torch.Tensor
) -> Tuple[
    PackedSequence, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    r"""Create a PackedSequence input for an RNN such that each
    set of steps that are part of the same episode are all part of
    a batch in the PackedSequence.

    Use the returned select_inds and build_rnn_out_from_seq to invert this.

    :param x: A (T * N, -1) tensor of the data to build the PackedSequence out of
    :param not_dones: A (T * N) tensor where not_dones[i] == False indicates an episode is done
    :param rnn_states: A (-1, N, -1) tensor of the rnn_hidden_states

    :return: tuple(x_seq, rnn_states, select_inds, rnn_state_batch_inds, last_episode_in_batch_mask)
        WHERE
        x_seq is the PackedSequence version of x to pass to the RNN

        rnn_states are the corresponding rnn state

        select_inds can be passed to build_rnn_out_from_seq to retrieve the
            RNN output

        rnn_state_batch_inds indicates which of the rollouts in the batch a hidden
            state came from/is for

        last_episode_in_batch_mask indicates if an episode is the last in that batch.
            There will be exactly N places where this is True

    """
    N = rnn_states.size(1)
    T = x.size(0) // N

    dones = torch.logical_not(not_dones)

    (
        select_inds,
        batch_sizes,
        episode_starts,
        rnn_state_batch_inds,
        last_episode_in_batch_mask,
    ) = _build_pack_info_from_dones(dones.detach().to(device="cpu"), T)

    x_seq = build_packed_input(x, select_inds, batch_sizes)

    episode_starts = episode_starts.to(device=x.device)
    rnn_state_batch_inds = rnn_state_batch_inds.to(device=x.device)
    last_episode_in_batch_mask = last_episode_in_batch_mask.to(device=x.device)
    

    # Just select the rnn_states by batch index, the masking bellow will set things
    # to zero in the correct locations
    rnn_states = rnn_states.index_select(1, rnn_state_batch_inds)
    # Now zero things out in the correct locations
    # N++ if one rollout breaks into two episodes, a new h0 would be added
    rnn_states = torch.where(
        not_dones.view(1, -1, 1).index_select(1, episode_starts),
        rnn_states,
        rnn_states.new_zeros(()),
    )

    return (
        x_seq,
        rnn_states,
        select_inds,
        rnn_state_batch_inds,
        last_episode_in_batch_mask,
        batch_sizes,
    )

def build_rnn_hidden_inputs(
    x: torch.Tensor, not_dones: torch.Tensor, rnn_states: torch.Tensor
) -> Tuple[
    PackedSequence, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    N = rnn_states.size(1)
    T = x.size(0) // N

    dones = torch.logical_not(not_dones)

    (
        select_inds,
        batch_sizes,
        episode_starts,
        rnn_state_batch_inds,
        last_episode_in_batch_mask,
    ) = _build_pack_info_from_dones(dones.detach().to(device="cpu"), T)


    episode_starts = episode_starts.to(device=x.device)
    rnn_state_batch_inds = rnn_state_batch_inds.to(device=x.device)
    last_episode_in_batch_mask = last_episode_in_batch_mask.to(device=x.device)
    

    # Just select the rnn_states by batch index, the masking bellow will set things
    # to zero in the correct locations
    rnn_states = rnn_states.index_select(1, rnn_state_batch_inds)
    # Now zero things out in the correct locations
    # N++ if one rollout breaks into two episodes, a new h0 would be added
    rnn_states = torch.where(
        not_dones.view(1, -1, 1).index_select(1, episode_starts),
        rnn_states,
        rnn_states.new_zeros(()),
    )

    return (
        rnn_states,
        select_inds,
        rnn_state_batch_inds,
        last_episode_in_batch_mask,
        batch_sizes,
    )


def build_rnn_out_from_seq(
    x_seq: PackedSequence,
    hidden_states,
    select_inds,
    rnn_state_batch_inds,
    last_episode_in_batch_mask,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Construct the output of the rnn from a packed sequence returned by
        forward propping an RNN on the packed sequence returned by :ref:`build_rnn_inputs`.

    :param x_seq: The packed sequence output from the rnn
    :param hidden_statess: The hidden states output from the rnn
    :param select_inds: Returned from :ref:`build_rnn_inputs`
    :param rnn_state_batch_inds: Returned from :ref:`build_rnn_inputs`
    :param last_episode_in_batch_mask: Returned from :ref:`build_rnn_inputs`
    :param N: The number of simulator instances in the batch of experience.
    """

    
    select_inds = select_inds.to(device=x_seq.data.device)

    x = x_seq.data.index_select(0, _invert_permutation(select_inds))

    output_hidden_states = build_unpacked_h_n(hidden_states, rnn_state_batch_inds,
    last_episode_in_batch_mask, N)
    
    return x, output_hidden_states

def build_rnn_out_from_seq_tensor(
    x_seq, # tensor
    hidden_states,
    select_inds,
    rnn_state_batch_inds,
    last_episode_in_batch_mask,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Construct the output of the rnn from a packed sequence returned by
        forward propping an RNN on the packed sequence returned by :ref:`build_rnn_inputs`.

    :param x_seq: The packed sequence output from the rnn
    :param hidden_statess: The hidden states output from the rnn
    :param select_inds: Returned from :ref:`build_rnn_inputs`
    :param rnn_state_batch_inds: Returned from :ref:`build_rnn_inputs`
    :param last_episode_in_batch_mask: Returned from :ref:`build_rnn_inputs`
    :param N: The number of simulator instances in the batch of experience.
    """

    select_inds = select_inds.to(device=x_seq.device)

    #print(select_inds)
    #print(_invert_permutation(select_inds))
    # print("--------- x_seq -----------")
    # print(x_seq)
    # print("---------------------------")

    x = x_seq.index_select(0, _invert_permutation(select_inds))

    output_hidden_states = build_unpacked_h_n(hidden_states, rnn_state_batch_inds,
    last_episode_in_batch_mask, N)
    

    return x, output_hidden_states

def build_unpacked_h_n(hidden_states, rnn_state_batch_inds,
    last_episode_in_batch_mask, N):
    last_hidden_states = torch.masked_select(
        hidden_states,
        last_episode_in_batch_mask.view(1, hidden_states.size(1), 1),
    ).view(hidden_states.size(0), N, hidden_states.size(2))
    output_hidden_states = torch.empty_like(last_hidden_states)
    scatter_inds = (
        torch.masked_select(rnn_state_batch_inds, last_episode_in_batch_mask)
        .view(1, N, 1)
        .expand_as(output_hidden_states)
    )
    output_hidden_states.scatter_(1, scatter_inds, last_hidden_states)

    return output_hidden_states

class RNNStateEncoder(nn.Module):
    r"""RNN encoder for use with RL and possibly IL.

    The main functionality this provides over just using PyTorch's RNN interface directly
    is that it takes an addition masks input that resets the hidden state between two adjacent
    timesteps to handle episodes ending in the middle of a rollout.
    """

    def layer_init(self):
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    # dummy
    def pack_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    # dummy
    def unpack_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    # input is a single vector
    # N: batch size
    # input x: [1*N,input_size]
    # hidden_states (h_{t-1}): [1, N, hidden_size]
    # output x: [1*N,hidden_size]
    def single_forward(
        self, x, hidden_states, masks
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward for a non-sequence input"""

        # if masks=true, i.e. not done, keep h, otherwise, reset to 0
        hidden_states = torch.where(
            masks.view(1, -1, 1), hidden_states, hidden_states.new_zeros(())
        )

        x, hidden_states = self.rnn(
            x.unsqueeze(0), self.unpack_hidden(hidden_states)
        )

        hidden_states = self.pack_hidden(hidden_states)

        # remove the first dim
        # [L,N,hidden_size] --> [L*N, hidden_size]
        x = x.squeeze(0)
        return x, hidden_states

    # input is a sequence of vectors
    # T: sequence length
    # N: batch size
    # input x: [T*N,input_size]
    # hidden_states(h_0): [1, N, hidden_size]
    # output x: [T*N,hidden_size]
    def seq_forward(
        self, x, hidden_states, masks
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward for a sequence of length T

        Args:
            x: (T, N, -1) Tensor that has been flattened to (T * N, -1)
            hidden_states: The starting hidden state.
            masks: The masks to be applied to hidden state at every timestep.
                A (T, N) tensor flatten to (T * N)
        """
        N = hidden_states.size(1)

        (
            x_seq,
            hidden_states,
            select_inds,
            rnn_state_batch_inds,
            last_episode_in_batch_mask,
            _
        ) = build_rnn_inputs(x, masks, hidden_states)

        # print(x_seq.data.size())
        # exit()

        x_seq, hidden_states = self.rnn(
            x_seq, self.unpack_hidden(hidden_states)
        )

        hidden_states = self.pack_hidden(hidden_states)

        x, hidden_states = build_rnn_out_from_seq(
            x_seq,
            hidden_states,
            select_inds,
            rnn_state_batch_inds,
            last_episode_in_batch_mask,
            N,
        )

        # hidden_states.requires_grad == True
        # x.requires_grad == True

        return x, hidden_states

    # x: [T*N,input_size]
    # hidden(h_n): [N, 1, hidden_size]
    # T: seq length
    # N: number of sequences
    def forward(
        self, x, hidden_states, masks
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # hidden: changed to [1, N, hidden_size]
        hidden_states = hidden_states.permute(1, 0, 2)

        # single forward: only forward one step, used when collecting data or evaluation
        # T*N = N --> T=1
        if x.size(0) == hidden_states.size(1):
            x, hidden_states = self.single_forward(x, hidden_states, masks)

        # sequence forward: forward for a sequence, only used during training
        else:
            x, hidden_states = self.seq_forward(x, hidden_states, masks)

        # hidden: changed to [N, 1, hidden_size]
        hidden_states = hidden_states.permute(1, 0, 2)

        return x, hidden_states

class AttentionRNNStateEncoder(RNNStateEncoder):
    def __init__(self, attention, attention_type, visual_map_size, hidden_size):
        # need to do initialization here because RNNStateEncoder derives from a NN module
        super().__init__()

        self.attention = attention
        if self.attention:
            # Attention visual encoder output dimension = RNN hidden size
            self.attention_model = Attention(encoder_dim=visual_map_size, hidden_dim=hidden_size, output_dim=hidden_size, attention_type=attention_type)

            
    # masks: not_dones
    # hidden_states: [N, 1, hidden_size]
    # visual_input: [T*N, 49, visual_input_size]
    # other_input: [T*N, other_input_size]
    # forward a sequence column by column with attention
    def attention_loop_seq_forward(self, visual_input, other_input, hidden_states, masks):
        # hidden states: [N, 1, hidden_size] --> [1, N, hidden_size]
        hidden_states = hidden_states.permute(1, 0, 2) 
        N = hidden_states.size(1)

        (
            hidden_states,
            select_inds,
            rnn_state_batch_inds,
            last_episode_in_batch_mask,
            batch_sizes,
        ) = build_rnn_hidden_inputs(visual_input, masks, hidden_states) 

        # loop forward column by column
        start_index = 0
        patch_weights_list = []
        select_inds = select_inds.to(device=visual_input.device)
        h_seq = []

        for batch_size in batch_sizes:
            # select input of step i
            cur_step_visual_input = visual_input.index_select(0, select_inds[start_index:start_index+batch_size])
            cur_step_other_input = other_input.index_select(0, select_inds[start_index:start_index+batch_size])
            
            # [1, N, hidden_size] --> [N, 1, hidden_size]
            permutated_h = hidden_states.permute(1, 0, 2)
            cur_step_permutated_h = permutated_h[:batch_size,:,:]

            # get attention selected visual input
            cur_step_selected_visual_input, patch_weights = self.attention_model(
                img_features=cur_step_visual_input, hidden_states=self.extract_h(cur_step_permutated_h)) #[N, 1, hidden_size]   
            
            patch_weights_list.append(patch_weights)    

            # x: concatenated input: [T*N,visual_input_size+other_input]
            cur_step_input = torch.cat((cur_step_selected_visual_input, cur_step_other_input), dim=1).unsqueeze(0)

            h_final, hn_out = self.rnn(cur_step_input, self.unpack_hidden(hidden_states[:,:batch_size,:]))
            
            if hn_out.size()[1] < hidden_states.size()[1]:
                hn_cat = [hn_out, hidden_states[:,batch_size:,:]]
                hidden_states = torch.cat(hn_cat, dim=1)
            else:
                hidden_states = hn_out
            
            h_seq.append(h_final.squeeze(0))

            start_index += batch_size 

        # concatentate h history
        h_seq = torch.cat(h_seq, dim=0)

        # rebuild output
        h_seq_rebuilt, hidden_states_rebuilt = build_rnn_out_from_seq_tensor(
            h_seq,
            hidden_states,
            select_inds,
            rnn_state_batch_inds,
            last_episode_in_batch_mask,
            N,
        )    

        return h_seq_rebuilt, hidden_states_rebuilt, patch_weights_list

    # masks: not_dones
    # hidden_states: [N, 1, hidden_size]
    # visual_input: [T*N, visual_input_size]
    # forward a sequence column by column
    # no attention
    # no pad
    def loop_seq_forward(self, visual_input, hidden_states, masks):

        # hidden states: [N, 1, hidden_size] --> [1, N, hidden_size]
        hidden_states = hidden_states.permute(1, 0, 2) 
        N = hidden_states.size(1)

        (
            hidden_states,
            select_inds,
            rnn_state_batch_inds,
            last_episode_in_batch_mask,
            batch_sizes,
        ) = build_rnn_hidden_inputs(visual_input, masks, hidden_states) 

        
        start_index = 0
        select_inds = select_inds.to(device=visual_input.device)
        
        h_seq = []

        for batch_size in batch_sizes:
            cur_step_input = visual_input.index_select(0, select_inds[start_index:start_index+batch_size]).unsqueeze(0)
            
            h_final, hn_out = self.rnn(cur_step_input, self.unpack_hidden(hidden_states[:,:batch_size,:]))
            
            if hn_out.size()[1] < hidden_states.size()[1]:
                hn_cat = [hn_out, hidden_states[:,batch_size:,:]]
                hidden_states = torch.cat(hn_cat, dim=1)
            else:
                hidden_states = hn_out
                
            h_seq.append(h_final.squeeze(0))

            start_index += batch_size 

        h_seq = torch.cat(h_seq, dim=0)

        h_seq_rebuilt, hidden_states_rebuilt = build_rnn_out_from_seq_tensor(
            h_seq,
            hidden_states,
            select_inds,
            rnn_state_batch_inds,
            last_episode_in_batch_mask,
            N,
        ) 
       
        return h_seq_rebuilt, hidden_states_rebuilt

    # visual_input: 
    # -no attention: [T*N,visual_input_size]
    # -attention: [T*N, 49, visual_input_size=256]
    # other_input: [T*N,other_input_size]
    # hidden(h_n): [N, 1, hidden_size]
    # T: seq length
    # N: number of sequences
    def forward(
        self, visual_input, other_input, hidden_states, masks, loop_seq=False
        ) -> Tuple[torch.Tensor, torch.Tensor]:


        # number of batches equal
        if visual_input is not None:
            assert visual_input.size(0) == other_input.size(0)

        # single forward: only forward one step, used when collecting data or evaluation
        # T*N = N --> T=1
        #if visual_input.size(0) == hidden_states.size(0):
        if other_input.size(0) == hidden_states.size(0):
            if self.attention:
                # lstm: extract h from [h,c]
                # gru: no change
                # get attention selected visual input
                visual_input, patch_weights = self.attention_model(
                    img_features=visual_input, hidden_states=self.extract_h(hidden_states))
            else:
                patch_weights = None        

            # hidden: [N, 1, hidden_size] --> [1, N, hidden_size]
            hidden_states = hidden_states.permute(1, 0, 2)    

            # x: concatenated input: [N, visual_input_size+other_input]
            if visual_input is not None:
                x = torch.cat((visual_input, other_input), dim=1) 
            else:
                x = other_input    
            # single forward   
            x, hidden_states = self.single_forward(x, hidden_states, masks)
            
        # sequence forward: forward for a sequence, only used during training
        else:
            if self.attention:
                x, hidden_states, patch_weights = self.attention_loop_seq_forward(visual_input, 
                other_input, hidden_states, masks)
            else:    
                
                # x: concatenated input: [T*N,visual_input_size+other_input]
                if visual_input is not None:
                    x = torch.cat((visual_input, other_input), dim=1) 
                else:
                    x = other_input    
                # seq forward
                # input: x: [T*N, input_size]
                # input: hidden_states: [N, 1, hidden_size]
                # output: x: [T*N, hidden_size]
                # output: hidden_states: [1, N, hidden_size]
                if loop_seq:
                    x, hidden_states = self.loop_seq_forward(x, hidden_states, masks)
                else:
                    # hidden: [N, 1, hidden_size] --> [1, N, hidden_size]
                    hidden_states = hidden_states.permute(1, 0, 2)
                    x, hidden_states = self.seq_forward(x, hidden_states, masks)

                patch_weights = None

        # hidden: [1, N, hidden_size] --> [N, 1, hidden_size]
        hidden_states = hidden_states.permute(1, 0, 2)

        return x, hidden_states, patch_weights    

# LSTM
#class LSTMStateEncoder(RNNStateEncoder):
class LSTMStateEncoder(AttentionRNNStateEncoder):    
    def __init__(
        self,
        attention: bool,
        attention_type: str,
        visual_encoder_output_size: int,
        other_input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        visual_map_size: int = 256
    ):
        #super().__init__()
        super().__init__(attention, attention_type, visual_map_size, hidden_size)

        # h+c
        self.num_recurrent_layers = num_layers * 2

        if self.attention:
            visual_input_size = self.attention_model.output_size
        else:
            visual_input_size = visual_encoder_output_size   

        self.rnn = nn.LSTM(
            input_size=visual_input_size+other_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.layer_init()

    # (h,c) --> [h,c]
    # h: [1, N, hidden_size]
    # c: [1, N, hidden_size] 
    def pack_hidden(
        self, hidden_states: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        # concatenate along dim 0
        return torch.cat(hidden_states, 0)

    # [h,c] --> (h,c)
    # h: [1, N, hidden_size]
    # c: [1, N, hidden_size] 
    def unpack_hidden(
        self, hidden_states
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # split a tensor into the specified number of chunks
        lstm_states = torch.chunk(hidden_states, 2, 0)
        return (lstm_states[0], lstm_states[1])

    # hidden_states: [N, 2, hidden_size]
    # h: [N, 1, hidden_size]
    def extract_h(self, hidden_states):
        lstm_states = torch.chunk(hidden_states, 2, 1)
        return lstm_states[0]

    
# GRU
#class GRUStateEncoder(RNNStateEncoder):
class GRUStateEncoder(AttentionRNNStateEncoder):
    def __init__(
        self,
        attention: bool,
        attention_type: str,
        visual_encoder_output_size: int,
        other_input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        visual_map_size: int = 256,
    ):
        #super().__init__()
        super().__init__(attention, attention_type, visual_map_size, hidden_size)

        self.num_recurrent_layers = num_layers

        if self.attention:
            visual_input_size = self.attention_model.output_size
        else:
            visual_input_size = visual_encoder_output_size    

        
        self.rnn = nn.GRU(
            input_size=visual_input_size+other_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.layer_init()

    # hidden_states: [N, 1, hidden_size]
    def extract_h(self, hidden_states):
        return hidden_states
    

# build RNN unit: input vector --> hidden vector
# lstm or gru
# A RNN unit could be num_layers network
def build_rnn_state_encoder(
    input_size: int,
    hidden_size: int,
    rnn_type: str = "gru",
    num_layers: int = 1,
):
    r"""Factory for :ref:`RNNStateEncoder`.  Returns one with either a GRU or LSTM based on
        the specified RNN type.

    :param input_size: The input size of the RNN
    :param hidden_size: The hidden dimension of the RNN
    :param rnn_types: The type of the RNN cell.  Can either be GRU or LSTM
    :param num_layers: The number of RNN layers.
    """
    rnn_type = rnn_type.lower()
    if rnn_type == "gru":
        return GRUStateEncoder(input_size, hidden_size, num_layers)
    elif rnn_type == "lstm":
        return LSTMStateEncoder(input_size, hidden_size, num_layers)
    else:
        raise RuntimeError(f"Did not recognize rnn type '{rnn_type}'")

def build_attention_rnn_state_encoder(
    attention: bool,
    visual_encoder_output_size: int,
    other_input_size: int,
    hidden_size: int,
    visual_map_size: int,
    rnn_type: str,
    attention_type: str,
    num_layers: int = 1,
):
    
    rnn_type = rnn_type.lower()
    if rnn_type == "gru":
        return GRUStateEncoder(attention, attention_type, visual_encoder_output_size, other_input_size, hidden_size, num_layers, visual_map_size)
    elif rnn_type == "lstm":
        return LSTMStateEncoder(attention, attention_type, visual_encoder_output_size, other_input_size, hidden_size, num_layers, visual_map_size)
    else:
        raise RuntimeError(f"Did not recognize rnn type '{rnn_type}'")

# verified equivalance between looping over the sequence step by step and by the whole sequence
def test_rnn_loop_eq():
    input_size = 3
    hidden_size = 2
    T = 5
    N = 2
    gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)

    x = torch.rand(T, N, input_size)
    
    h = torch.rand(1, N, hidden_size)

    h_seq_1, h_final = gru(x, h)

    print("h_seq_1: %s"%(str(h_seq_1)))  # x_seq,data = x, nothing changed
    print("h_seq_1 size: %s"%(str(h_seq_1.size())))
    print("h_final_1: %s"%(h_final))
    print("h_final_1 size: %s"%(str(h_final.size())))

    h0 = h
    for i in range(T):
       x_in = x[i, :, :].unsqueeze(0)
       #print(x_in.size())
       h_seq_2, h0 = gru(x_in, h0)

    print("h_seq_2: %s"%(str(h_seq_2)))  # x_seq,data = x, nothing changed
    print("h_seq_2 size: %s"%(str(h_seq_2.size())))
    print("h_final_2: %s"%(h0))
    print("h_final_2 size: %s"%(str(h0.size())))   

def test_rnn():
    input_size = 3
    hidden_size = 2
    T = 5
    N = 2
    gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
    

    # x: [T,N,input_size]
    # h: [1,N,hidden_size]
    x = torch.rand(T, N, input_size)
    x = x.flatten(0,1)
    h = torch.rand(1, N, hidden_size)

    print("x: %s"%(x))
    print("x size: %s"%(str(x.size())))
    print("h: %s"%(h))
    print("h size: %s"%(str(h.size())))

    # dones: [N,T]
    dones = torch.torch.Tensor([[0,1,0,1,0], [0,0,0,1,0]]).bool()
    print("dones: %s"%(dones))
    # dones: [T,N]
    dones = dones.permute(1,0)
    # dones: [T*N,1]
    dones = dones.flatten(0,1).unsqueeze(1)
    print("dones: %s"%(dones))
    
    not_dones = torch.logical_not(dones)

    (
        select_inds,
        batch_sizes,
        episode_starts,
        rnn_state_batch_inds,
        last_episode_in_batch_mask,
    ) = _build_pack_info_from_dones(dones, T)

    print("select_inds: %s"%(str(select_inds)))
    print("batch_sizes: %s"%(str(batch_sizes)))
    #print(episode_starts)

    (
        x_seq,
        h,
        select_inds,
        rnn_state_batch_inds,
        last_episode_in_batch_mask,
        batch_sizes2
    )=build_rnn_inputs(x, not_dones, h) 

    #print("batch_sizes2: %s"%(batch_sizes2))
    assert torch.equal(batch_sizes2, batch_sizes)
    print("rnn_state_batch_inds: %s"%(str(rnn_state_batch_inds)))

    # scatter_inds = (
    #     torch.masked_select(rnn_state_batch_inds, last_episode_in_batch_mask)
    #     .view(1, N, 1)
    #     .expand_as(h)
    # )

    # print("scatter_inds: %s"%(str(scatter_inds)))
    

    print("*****  after padding:  *****")
    #print("x_seq: %s"%(str(x_seq.data)))  # x_seq,data = x, nothing changed
    print("x_seq size: %s"%(str(x_seq.data.size())))
    print("h: %s"%(h))
    print("h size: %s"%(str(h.size())))

    pad_x_seq, pad_batch_sizes = pad_packed_sequence(x_seq)

    print("pad_batch_sizes: %s"%(str(pad_batch_sizes)))
    #print("pad_x_seq: %s"%(str(pad_x_seq))) 
    print("pad_x_seq size: %s"%(str(pad_x_seq.size())))  # T,N,input_size

    print("*****  after seq rnn:  *****")
    #print(h)
    #h_seq_1, h_final = gru(pad_x_seq, h)
    h_seq_1, h_final_1 = gru(x_seq, h)
    #print("h_seq_1: %s"%(str(h_seq_1.data)))  # x_seq,data = x, nothing changed
    #print("h_seq_1 size: %s"%(str(h_seq_1.data.size())))
    print("h_final_1: %s"%(h_final_1))
    print("h_final_1 size: %s"%(str(h_final_1.size())))

    h_seq_1_rebuilt, h_final_1_rebuilt = build_rnn_out_from_seq(
            h_seq_1,
            h_final_1,
            select_inds,
            rnn_state_batch_inds,
            last_episode_in_batch_mask,
            N,
        )

    print("h_final_1_rebuilt: %s"%(h_final_1_rebuilt))
    print("h_final_1_rebuilt size: %s"%(str(h_final_1_rebuilt.size())))  
    print("h_seq_1 size: %s"%(str(h_seq_1.data.size())))
    print("h_seq_1_rebuilt size: %s"%(str(h_seq_1_rebuilt.size())))  
    print("h_seq_1_rebuilt: %s"%(str(h_seq_1_rebuilt))) 

    print("*****  after loop rnn:  *****")
    h0 = h
    #print(h0)

    # The following code try to reverse engineering pytorch implmentation of rnn on sequences of variable lengths
    # method 1: padding with 0
    # for i in range(len(batch_sizes)):
    #    x_in = pad_x_seq[i, :, :].unsqueeze(0)
    #    #print(x_in.size())
    #    h_seq_2, h0 = gru(x_in, h0)

    # method 2: only forward the relevant part
    start_index = 0
    h_seq_2 = []
    max_batch_size = torch.max(batch_sizes)
    for batch_size in batch_sizes:
       x_in = x.index_select(0, select_inds[start_index:start_index+batch_size]).unsqueeze(0) 
       #print(x_in.size())
       #print(batch_size)

       # h_final_2 size: [1, batch_sizes[-1], hidden_size]
       h0_clone = h0.clone()
       h_final_2, h0_clone[:,:batch_size,:] = gru(x_in, h0[:,:batch_size,:]) 
       h0 = h0_clone
       #h_final_2, h0[:,:batch_size,:] = gru(x_in, h0[:,:batch_size,:])

       #assert(torch.equal(h_final_2, t))

       h_final_2 = h_final_2.squeeze(0)
       size0, size1 = h_final_2.size()
       h_final_2_pad = torch.cat((h_final_2, torch.zeros(max_batch_size - size0, size1)),dim=0)
       #print(h_final_2_pad)
       #print("------------------")
       

       #h_seq_2.append(h0.squeeze(0))
       h_seq_2.append(h_final_2_pad)
       
       start_index += batch_size  

    h_seq_2 = torch.cat(h_seq_2, dim=0)

    print("h_seq_2 size: %s"%(str(h_seq_2.size()))) 
    
    
    print("h_final_2: %s"%(h0))
    print("h_final_2 size: %s"%(str(h0.size())))
    #print("h_final_2: %s"%(h_final_2))
    #print("h_final_2 size: %s"%(str(h_final_2.size())))
    #print("h_seq_2: %s"%(str(h_seq_2)))  # x_seq,data = x, nothing changed
    

    # h0_rebuilt = build_unpacked_h_n(
    #         h0,
    #         rnn_state_batch_inds,
    #         last_episode_in_batch_mask,
    #         N,
    #     )

    h_seq_2_rebuilt, h0_rebuilt = build_rnn_out_from_seq(
            h_seq_2,
            h0,
            select_inds,
            rnn_state_batch_inds,
            last_episode_in_batch_mask,
            N,
        )

    print("h_final_2_rebuilt: %s"%(h0_rebuilt))
    print("h_final_2_rebuilt size: %s"%(str(h0_rebuilt.size())))  
    print("h_seq_2 size: %s"%(str(h_seq_2.size())))  
    print("h_seq_2 rebuilt size: %s"%(str(h_seq_2_rebuilt.size())))
    print("h_seq_2 rebuilt: %s"%(str(h_seq_2_rebuilt)))

# visual_input: -no attention: [T*N,visual_input_size]
# other_input: [T*N,other_input_size]
# hidden(h_n): [N, 1, hidden_size]
# T: seq length
# N: number of sequences    
def test_gru():
    torch.set_printoptions(precision=10)

    visual_input_size = 2
    other_input_size = 1
    hidden_size = 2
    T = 5
    N = 2

    # create model
    gru = GRUStateEncoder(attention=False, visual_encoder_output_size=visual_input_size, other_input_size=other_input_size, 
        hidden_size=hidden_size, num_layers=1, visual_map_size=0)
    
    for i in range(10):
        # generate data
        visual_input = torch.rand(T*N, visual_input_size)
        other_input = torch.rand(T*N, other_input_size) 
        hidden_input = torch.rand(N, 1, hidden_size)

        # dones: [N,T]
        #dones = torch.torch.Tensor([[0,1,0,1,0], [0,0,0,1,0]]).bool()
        dones = torch.randint(0, 2, (2,5)).bool()
        # dones: [T,N]
        dones = dones.permute(1,0)
        # dones: [T*N,1]
        dones = dones.flatten(0,1).unsqueeze(1)
        # masks = not_dones
        masks = torch.logical_not(dones)

        # ----------forward: loop ---------------------
        x_output_loop, hidden_output_loop, _ = gru(visual_input, other_input, hidden_input, masks, loop_seq=True) # hidden: [N, 1, hidden_size]
        #print("h_loop: %s"%(hidden_output_loop))
        #print("h_loop size: %s"%(str(hidden_output_loop.size())))
        #print("x_output_loop: %s"%(x_output_loop))
        #print("x_output_loop size: %s"%(str(x_output_loop.size())))
        #print(hidden_output_loop.dtype)
        #exit()
        # ----------backward: loop ---------------------
        #hidden_output_loop.mean().backward()
        #p_loop = gru.rnn.parameters()
        # for p in p_loop:
        #     print(p.grad)

        #exit()
        # ----------forward: seq ---------------------
        #x_output_loop, hidden_output_loop, _ = loop_gru(visual_input, other_input, hidden_input, masks)
        x_output_seq, hidden_output_seq, _ = gru(visual_input, other_input, hidden_input, masks, loop_seq=False) # hidden: [N, 1, hidden_size]
        #print("h_seq: %s"%(hidden_output_seq))
        #print("h_seq size: %s"%(str(hidden_output_seq.size())))
        #print("x_output_seq: %s"%(x_output_seq))
        #print("x_output_seq size: %s"%(str(x_output_seq.size())))
        # ----------backward: seq ---------------------
        hidden_output_seq.mean().backward()
        #p_seq = gru.rnn.parameters()
        # for p in p_seq:
        #     print(p.grad)

        print("------------------ i = %d ---------------------------"%(i))
        # ----------compare forward ---------------------
        forward_h_result = (hidden_output_loop == hidden_output_seq).all().cpu().numpy()
        forward_x_result = (x_output_loop == x_output_seq).all().cpu().numpy()
        if forward_h_result == False:
            print("forward final hidden state equal: %s"%(forward_h_result))
            print("final loop: %s"%hidden_output_loop)
            print("final seq: %s"%hidden_output_seq) 
            print("-----------------------------------------------------")
        if forward_x_result == False: 
            print("forward hidden state sequence equal: %s"%(forward_x_result))
            print("history loop: %s"%x_output_loop)
            print("history seq: %s"%x_output_seq)   
        # ----------compare backward ---------------------  
    
        # for p1, p2 in zip(p_loop, p_seq):
        #     print("backward equal: %s"%((p1.grad == p2.grad).all()))
            #print('-------------------------------')
            #print(p1.grad)
            #print(p2.grad)
        #print('backward')    
        #print(hidden_output_seq.grad)    
        
def test_tensor_slice():
    h = torch.rand(2, 3)
    h.requires_grad = True
    h1 = h[0, :]
    h1.retain_grad()

    l=h1.mean()
    l.backward()

    #print(type(h1))
    #print(h1.requires_grad)  
    print(h.grad)
    print(h1.grad)

def test_build_rnn_inputs():
    input_size = 3
    hidden_size = 2
    T = 5
    N = 2
    gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
    

    # x: [T,N,input_size]
    # h: [1,N,hidden_size]
    x = torch.rand(T, N, input_size)
    print("x: %s"%(x))
    x = x.flatten(0,1)
    h = torch.rand(1, N, hidden_size)

    
    # print("x size: %s"%(str(x.size())))
    # print("h: %s"%(h))
    # print("h size: %s"%(str(h.size())))

    # dones: [N,T]
    dones = torch.torch.Tensor([[0,1,0,1,0], [0,0,0,1,0]]).bool()
    # print("dones: %s"%(dones))
    # dones: [T,N]
    dones = dones.permute(1,0)
    # dones: [T*N,1]
    dones = dones.flatten(0,1).unsqueeze(1)
    # print("dones: %s"%(dones))
    
    not_dones = torch.logical_not(dones)

    (
        select_inds,
        batch_sizes,
        episode_starts,
        rnn_state_batch_inds,
        last_episode_in_batch_mask,
    ) = _build_pack_info_from_dones(dones, T)

    # print("select_inds: %s"%(str(select_inds)))
    # print("batch_sizes: %s"%(str(batch_sizes)))
    #print(episode_starts)

    (
        x_seq,
        h,
        select_inds,
        rnn_state_batch_inds,
        last_episode_in_batch_mask,
        batch_sizes2
    )=build_rnn_inputs(x, not_dones, h) 

    # print("==============================")
    # print("x: %s"%(x))
    # print("x_seq: %s"%(x_seq.data))
    # print("batch_sizes1: %s"%(batch_sizes))
    # print("batch_sizes2: %s"%(batch_sizes2))

if __name__ == "__main__":
    #test_rnn()
    #test_rnn_loop_eq()
    #test_gru()
    #test_tensor_slice()
    test_build_rnn_inputs()