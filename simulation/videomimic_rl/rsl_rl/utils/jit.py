import os
import zipfile
import torch
import copy
from typing import Dict
from rsl_rl.modules.actor_critic import FlattenThenEmbedMLPWithAttention

def export_policy_as_jit(actor_critic, path):
    """Export the policy as a JIT file"""
    # get directory of path
    path_dir = os.path.dirname(path)
    # make the directory if it doesn't exist
    os.makedirs(path_dir, exist_ok=True)

    policy_exporter = _TorchPolicyExporter(actor_critic)
    policy_exporter.export(path)

def get_torchscript_model(actor_critic):
    """Return the JITted module"""
    policy_exporter = _TorchPolicyExporter(actor_critic)
    return torch.jit.script(policy_exporter)


def try_load_jit_model(filename):
    """Try to load a JIT model from a file.
    If the file is not a JIT model, return False.
    """
    try:
        model = torch.jit.load(filename)
        return True, model
    except Exception:
        return False, None

class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file.
    Modified from Isaac Lab implementation
    """

    def __init__(self, actor_critic, normalizer=None):
        super(_TorchPolicyExporter, self).__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.critic = copy.deepcopy(actor_critic.critic)

        self.actor_input_net = copy.deepcopy(actor_critic.actor_input_net)
        self.critic_input_net = copy.deepcopy(actor_critic.critic_input_net)
        self.actor_std = copy.deepcopy(actor_critic.std)
        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
            self.rnn.cpu()
            self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.register_buffer("hidden_state_critic", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.register_buffer("cell_state_critic", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.forward = self.forward_lstm
            self.reset = self.reset_memory
            self.evaluate = self.evaluate_lstm
            self.reset_env_ids = self.reset_env_ids_recurrent
        # copy normalizer if exists
        # if normalizer:
        #     self.normalizer = copy.deepcopy(normalizer)
        # else:
        #     self.normalizer = torch.nn.Identity()
        if normalizer:
            raise NotImplementedError("Normalizer not supported for JIT export yet (for dict obs)")

    def forward_lstm(self, x: Dict[str, torch.Tensor]):
        # x = self.normalizer(x)
        x, extra_proj_outputs = self.actor_input_net(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x, extra_proj_outputs)

    def forward(self, x: Dict[str, torch.Tensor]):
        # x = self.normalizer(x)
        x, extra_proj_outputs = self.actor_input_net(x)
        return self.actor(x, extra_proj_outputs)
    
    def evaluate_lstm(self, x: Dict[str, torch.Tensor]):
        # x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state_critic, self.cell_state_critic))
        self.hidden_state_critic[:] = h
        self.cell_state_critic[:] = c
        x = x.squeeze(0)
        return self.critic(x)
    
    @torch.jit.export
    def evaluate(self, x: Dict[str, torch.Tensor]):
        # x = self.normalizer(x)
        x, extra_proj_outputs = self.critic_input_net(x)
        return self.critic(x, extra_proj_outputs)
    
    @torch.jit.export
    def std(self):
        return self.actor_std

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0
        self.hidden_state_critic[:] = 0.0
        self.cell_state_critic[:] = 0.0

    def export(self, path):
        path_dir = os.path.dirname(path)
        # make the directory if it doesn't exist
        os.makedirs(path_dir, exist_ok=True)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
    
    @torch.jit.export
    def reset_env_ids(self, env_ids):
        return
    
    def reset_env_ids_recurrent(self, env_ids):
        self.hidden_state[env_ids] = 0.0
        self.cell_state[env_ids] = 0.0
        self.hidden_state_critic[env_ids] = 0.0
        self.cell_state_critic[env_ids] = 0.0
