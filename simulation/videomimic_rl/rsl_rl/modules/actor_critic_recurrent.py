# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from .actor_critic import ActorCritic, get_activation
from rsl_rl.utils import unpad_trajectories


SKIP_CONNECTIONS = True

class ActorCriticRecurrent(ActorCritic):
    is_recurrent = True

    def __init__(
        self,
        # num_actor_obs,
        # num_critic_obs,
        obs_shapes,
        num_actions,
        obs_proc_actor,
        obs_proc_critic,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation='elu',
        rnn_type='lstm',
        rnn_hidden_size=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        super().__init__(
            # num_actor_obs=rnn_hidden_size+ (num_actor_obs if SKIP_CONNECTIONS else 0),
            # num_critic_obs=rnn_hidden_size+ (num_critic_obs if SKIP_CONNECTIONS else 0),
            obs_shapes=obs_shapes,
            # num_actor_obs=rnn_hidden_size,
            # num_critic_obs=rnn_hidden_size,
            num_actions=num_actions,
            obs_proc_actor=obs_proc_actor,
            obs_proc_critic=obs_proc_critic,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            lstm_dim=rnn_hidden_size,
        )

        activation = get_activation(activation)

        # self.memory_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        # self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_a = Memory(self.actor_input_net.output_shape, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(self.critic_input_net.output_shape, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    # def act(self, observations, masks=None, hidden_states=None):
    #     input_a = self.memory_a(observations, masks, hidden_states)
    #     return super().act(input_a.squeeze(0))

    # def act_inference(self, observations):
    #     input_a = self.memory_a(observations)
    #     return super().act_inference(input_a.squeeze(0))

    # def evaluate(self, critic_observations, masks=None, hidden_states=None):
    #     input_c = self.memory_c(critic_observations, masks, hidden_states)
    #     return super().evaluate(input_c.squeeze(0))

    # def act(self, observations, masks=None, hidden_states=None):
    #     input_a = self.memory_a(observations, masks, hidden_states)
    #     if masks is not None:
    #         observations = unpad_trajectories(observations, masks)
    #     if SKIP_CONNECTIONS:
    #         return super().act(torch.cat([input_a.squeeze(0), observations], dim=-1))
    #     else:
    #         return super().act(torch.cat([input_a.squeeze(0)], dim=-1))

    # def act_inference(self, observations):
    #     input_a = self.memory_a(observations)
    #     if SKIP_CONNECTIONS:
    #         return super().act_inference(torch.cat([input_a.squeeze(0), observations], dim=-1))
    #     else:
    #         return super().act_inference(torch.cat([input_a.squeeze(0)], dim=-1))

    # def evaluate(self, critic_observations, masks=None, hidden_states=None):
    #     input_c = self.memory_c(critic_observations, masks, hidden_states)
    #     if masks is not None:
    #         critic_observations = unpad_trajectories(critic_observations, masks)
    #     if SKIP_CONNECTIONS:
    #         return super().evaluate(torch.cat([input_c.squeeze(0), critic_observations], dim=-1))
    #     else:
    #         return super().evaluate(torch.cat([input_c.squeeze(0)], dim=-1))

    def act(self, observations, masks=None, hidden_states=None):
        # TODO fix because the low level also will want to pre process
        observations = self.actor_input_net(observations)
        input_a = self.memory_a(observations, masks, hidden_states)
        if masks is not None:
            observations = unpad_trajectories(observations, masks)
        return super().act(torch.cat([input_a.squeeze(0), observations], dim=-1), call_input_net=False)
        # return super().act(input_a.squeeze(0) + observations, call_input_net=False)

    def act_inference(self, observations):
        observations = self.actor_input_net(observations)
        input_a = self.memory_a(observations)
        return super().act_inference(torch.cat([input_a.squeeze(0), observations], dim=-1), call_input_net=False)
        # return super().act_inference(input_a.squeeze(0) + observations, call_input_net=False)

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        # copy dict
        # critic_observations = {k:v for k, v in critic_observations.items()}
        # if self.mul_env_type is None:
        #     self.mul_env_type = 0.05 / torch.arange(1, critic_observations['policy'].shape[1]+1, device=critic_observations['env_type'].device).float().view(-1, 281) 
        #     # hack to add var that wasn't in there before
        # critic_observations['policy'] = critic_observations['policy'] + (self.mul_env_type + critic_observations['env_type'])
        critic_observations = self.critic_input_net(critic_observations)
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        if masks is not None:
            critic_observations = unpad_trajectories(critic_observations, masks)
        return super().evaluate(torch.cat([input_c.squeeze(0), critic_observations], dim=-1), call_input_net=False)
        # return super().evaluate(input_c.squeeze(0) + critic_observations, call_input_net=False)

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states


class Memory(torch.nn.Module):
    def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None
    
    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0
