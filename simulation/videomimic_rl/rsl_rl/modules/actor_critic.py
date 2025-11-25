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
from typing import Dict, List, Optional, Tuple


class EmbedMLP(nn.Module):

    def __init__(self, input_size, output_size, bias=True):
        super(EmbedMLP, self).__init__()
        self.input_proc = nn.Linear(input_size, output_size, bias=bias)
    
    def forward(self, x):
        return self.input_proc(x)

class EmbedMLPWithAttention(nn.Module):
    def __init__(self, input_size, output_size):
        super(EmbedMLPWithAttention, self).__init__()
        self.attention = nn.Parameter(torch.zeros(output_size,))
        if isinstance(input_size, tuple):
            assert len(input_size) == 1, f"Can only embed 1d observation, but obs {input_size} has shape {input_size}"
            input_size = input_size[0]
        self.embed = EmbedMLP(input_size, output_size)

    def forward(self, x):
        out = self.embed(x)
        return out * self.attention

class FlattenThenEmbedMLP(nn.Module):
    def __init__(self, input_size, output_size, bias=True):
        super(FlattenThenEmbedMLP, self).__init__()
        self.flatten = nn.Flatten()
        # embed size is product of input size
        input_size_flattened = int(np.prod(input_size))
        self.embed = EmbedMLP(input_size_flattened, output_size, bias=bias)

    def forward(self, x):
        flattened = self.flatten(x)
        return self.embed(flattened)

class FlattenThenEmbedMLPWithAttention(nn.Module):
    def __init__(self, input_size, output_size):
        super(FlattenThenEmbedMLPWithAttention, self).__init__()
        self.attention = nn.Parameter(torch.zeros(output_size,))
        self.embed = FlattenThenEmbedMLP(input_size, output_size)

    def forward(self, x):
        out = self.embed(x)
        return out * self.attention

obs_proc_types = {
    "identity": nn.Identity,
    "flatten": nn.Flatten,
    "embed": EmbedMLP,
    'flatten_then_embed': FlattenThenEmbedMLP,
    'flatten_then_embed_with_attention': FlattenThenEmbedMLPWithAttention,
    'flatten_then_embed_with_attention_to_hidden': FlattenThenEmbedMLPWithAttention,
    'embed_with_attention_to_hidden': EmbedMLPWithAttention,
}


class ForwardProcDict(nn.Module):

    def __init__(self, obs_shapes, obs_proc_spec, add_outputs=True, learn_weights=False, embed_dim=512, first_hidden_dim=256):

        super(ForwardProcDict, self).__init__()

        self.add_outputs = add_outputs
        self.learn_weights = learn_weights
        self.obs_proc_spec = obs_proc_spec
        self.first_hidden_dim = first_hidden_dim

        obs_proc_heads = {}
        extra_proj_heads = {}
        for k, v in obs_proc_spec.items():
            if v["type"] in ["identity", "flatten",]:
                obs_proc_heads[k] = obs_proc_types[v["type"]]()
            # these project to the network input space and then add it on to the input
            elif v["type"] == "embed" or v["type"] == "flatten_then_embed" or v["type"] == "flatten_then_embed_with_attention":
                if v["type"] == "embed":
                    assert len(obs_shapes[k]) == 1, f"Can only embed 1d observation, but obs {k} has shape {obs_shapes[k]}"
                output_dim = v["output_dim"]
                obs_proc_heads[k] = obs_proc_types[v["type"]](obs_shapes[k], output_dim)
            elif v["type"] == "flatten_then_embed_with_attention_to_hidden" or v["type"] == "embed_with_attention_to_hidden":
                try:
                    extra_proj_heads[k] = obs_proc_types[v["type"]](obs_shapes[k], first_hidden_dim)
                except Exception as e:
                    print(f"Error creating extra proj head for {k}: {e}")
                    import pdb; pdb.set_trace()
            else:
                raise NotImplementedError(f"Obs proc type {v['type']} not implemented")

        self.heads = nn.ModuleDict(obs_proc_heads)
        self.extra_proj_heads = nn.ModuleDict(extra_proj_heads)

        output_shape = self.forward({k: torch.zeros(1, *obs_shapes[k]) for k in obs_shapes})[0].shape
        assert len(output_shape) == 2 # expect one batch dim and then latent dim
        self.output_shape = output_shape[1]
    
    def forward(self, input_dict: Dict[str, torch.Tensor]):
        outputs = []
        extra_add_outputs = []
        extra_proj_outputs = []
        for k, head in self.heads.items():
            head_output = head(input_dict[k])
            outputs.append(head_output)

        # extra proj heads, returned separately to be added later to the net
        for k, head in self.extra_proj_heads.items():
            head_output = head(input_dict[k])
            extra_proj_outputs.append(head_output)

        if self.add_outputs:
            ret = torch.stack(outputs, dim=0).sum(dim=0)
            if len(extra_add_outputs) > 0:
                ret = ret + torch.stack(extra_add_outputs, dim=0).sum(dim=0)
        else:
            ret = torch.cat(outputs, dim=-1)
            if len(extra_add_outputs) > 0:
                ret = ret + torch.stack(extra_add_outputs, dim=0).sum(dim=0)
        return ret, extra_proj_outputs


class SequentialWithExtraProj(nn.Sequential):

    def __init__(self, *args, **kwargs):
        super(SequentialWithExtraProj, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, extra_proj_outputs: Optional[List[torch.Tensor]] = None):
        for idx, module in enumerate(self):
            if idx == 0:
                x = module(x)
                if extra_proj_outputs is not None and len(extra_proj_outputs) > 0:
                    for extra_proj_output in extra_proj_outputs:
                        x = x + extra_proj_output
            else:
                x = module(x)
        return x

class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  obs_shapes,
                        num_actions,
                        obs_proc_actor,
                        obs_proc_critic,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        lstm_dim=0,
                        layer_norm=False,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        # For activation monitoring
        self.activation_hooks = []
        self.activation_values = {}

        self.env_obs_shapes = obs_shapes
        self.env_num_actions = num_actions

        self.actor_input_net = ForwardProcDict(obs_shapes, obs_proc_actor, add_outputs=False, embed_dim=256 if lstm_dim == 0 else lstm_dim, first_hidden_dim=actor_hidden_dims[0])
        self.critic_input_net = ForwardProcDict(obs_shapes, obs_proc_critic, add_outputs=False, embed_dim=256 if lstm_dim == 0 else lstm_dim, first_hidden_dim=critic_hidden_dims[0])

        mlp_input_dim_a = self.actor_input_net.output_shape + lstm_dim
        mlp_input_dim_c = self.critic_input_net.output_shape + lstm_dim

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        self.num_actions = num_actions
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                if layer_norm:
                    actor_layers.append(nn.LayerNorm(actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = SequentialWithExtraProj(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                if layer_norm:
                    critic_layers.append(nn.LayerNorm(critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = SequentialWithExtraProj(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)
    
    def re_init_std(self, init_noise_std=1.0):
        self.std.data[:] = init_noise_std 

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def update_distribution(self, observations, call_input_net=True):
        if call_input_net:
            obs_after_proc, extra_proj_outputs = self.actor_input_net(observations)
        logits = self.actor(obs_after_proc, extra_proj_outputs=extra_proj_outputs)
        try:
            std = self.std.expand_as(logits)
            self.distribution = Normal(logits, std)
        except ValueError as e:
            print(f"Error updating distribution: {e}")
            print(f"Logits: {logits}")
            print(f"Std: {std}")

            for k in observations:
                print(f'{k} nan: {torch.isnan(observations[k]).any()}')
                print(f'{k} inf: {torch.isinf(observations[k]).any()}')

            # check if any nan or inf in logits or std
            print(f"Logits nan: {torch.isnan(logits).any()}")
            print(f"Logits inf: {torch.isinf(logits).any()}")
            print(f"Std nan: {torch.isnan(std).any()}")
            print(f"Std inf: {torch.isinf(std).any()}")
            raise e

    def act(self, observations, call_input_net=True, **kwargs):
        self.update_distribution(observations, call_input_net=call_input_net)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, call_input_net=True, monitor_activations=False):
        # Register activation hooks if monitoring is enabled
        if monitor_activations:
            self.register_activation_hooks()
            
        if call_input_net:
            observations, extra_proj_outputs = self.actor_input_net(observations)
        logits = self.actor(observations, extra_proj_outputs=extra_proj_outputs)
        
        # Print activation statistics if monitoring is enabled
        if monitor_activations:
            self.print_activation_stats()
            self.remove_activation_hooks()
            
        return logits

    def evaluate(self, critic_observations, call_input_net=True, monitor_activations=False, **kwargs):
        # Register activation hooks if monitoring is enabled
        if monitor_activations:
            self.register_activation_hooks()
            
        if call_input_net:
            critic_observations, extra_proj_outputs = self.critic_input_net(critic_observations)
        value = self.critic(critic_observations, extra_proj_outputs=extra_proj_outputs)
        
        # Print activation statistics if monitoring is enabled
        if monitor_activations:
            self.print_activation_stats()
            self.remove_activation_hooks()
            
        # if self.normalise_value:
        #     self.value_normalisation.inverse(value)
        return value

    # def learn_value(self, values):
    #     if self.normalise_value:
    #         self.value_normalisation.update(values.view(-1, 1))

    def register_activation_hooks(self):
        """Register forward hooks on each layer to track activations."""
        # Clear any existing hooks
        self.remove_activation_hooks()
        self.activation_values = {}
        
        # Hook function to save activations
        def hook_fn(name):
            def hook(module, input, output):
                self.activation_values[name] = output.detach()
            return hook
        
        # Register hooks for actor layers
        for i, module in enumerate([m for m in self.actor if isinstance(m, nn.Linear)]):
            hook = module.register_forward_hook(hook_fn(f"actor_layer_{i}"))
            self.activation_hooks.append(hook)
            
        # Register hooks for critic layers
        for i, module in enumerate([m for m in self.critic if isinstance(m, nn.Linear)]):
            hook = module.register_forward_hook(hook_fn(f"critic_layer_{i}"))
            self.activation_hooks.append(hook)
    
    def remove_activation_hooks(self):
        """Remove all registered forward hooks."""
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks = []
    
    def print_activation_stats(self):
        """Print statistics of activations in each layer."""
        if not self.activation_values:
            print("No activation values recorded. Call register_activation_hooks() before forward pass.")
            return
            
        print("\n===== Activation Statistics =====")
        for name, activation in sorted(self.activation_values.items()):
            act_mean = activation.mean().item()
            act_min = activation.min().item()
            act_max = activation.max().item()
            act_std = activation.std().item()
            print(f"{name:20s}: mean={act_mean:.6f}, min={act_min:.6f}, max={act_max:.6f}, std={act_std:.6f}")
        print("=================================\n")

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
