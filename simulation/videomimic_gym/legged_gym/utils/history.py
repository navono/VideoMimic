import torch
from torch import Tensor
from typing import Dict, Tuple, Optional

class HistoryHandler:
    
    def __init__(self, num_envs: int, history_config: Dict[str, int], obs_dims: Dict[str, Tuple], device: torch.device,
                 copy_first_obs: bool = True):
        """
        history_config: dict of {obs_key: history_len}
        obs_dims: dict of {obs_key: obs_dims} where obs_dims is a tuple of the shape of the observation
        copy_first_obs: if True, the first observation is copied to the history buffer after reset.
        """
        self.obs_dims = obs_dims
        self.device = device
        self.num_envs = num_envs
        self.copy_first_obs = copy_first_obs

        self.history = {}
        self.just_reset = {}
        for key in history_config.keys():
            self.history[key] = torch.zeros(num_envs, history_config[key], *obs_dims[key], device=self.device)
            self.just_reset[key] = torch.zeros(num_envs, device=self.device, dtype=torch.bool)

        print("History Handler Initialized")
        for key, value in self.history.items():
            print(f"Key: {key}, Shape: {value.shape}")

    def reset(self, reset_ids):
        if len(reset_ids)==0:
            return
        for key in self.history.keys():
            self.history[key][reset_ids] *= 0.
            self.just_reset[key][reset_ids] = True

    def add(self, key: str, value: Tensor):
        assert key in self.history.keys(), f"Key {key} not found in history"
        val = self.history[key].clone()
        self.history[key][:, 1:] = val[:, :-1]
        self.history[key][:, 0] = value.clone()

        if self.copy_first_obs:
            just_reset_ids = torch.nonzero(self.just_reset[key], as_tuple=False).flatten()
            if len(just_reset_ids) > 0:
                # Ensure tensor dtype matches by explicitly converting to the same dtype as the history tensor
                source_tensor = value.clone()[just_reset_ids].unsqueeze(1).to(self.history[key].dtype)
                self.history[key][just_reset_ids, 1:] = source_tensor
        
        self.just_reset[key][:] = False
        
    def query(self, key: str):
        assert key in self.history.keys(), f"Key {key} not found in history"
        return self.history[key].clone()
    
    def get_latest(self, key: str):
        assert key in self.history.keys(), f"Key {key} not found in history"
        return self.history[key][:, 0].clone()
    
    def query_at_history(self, indices: Tensor, key: Optional[str] = None):
        """
        Query the history at specific indices for each environment.
        
        Args:
            indices (Tensor): A tensor of shape (num_envs,) containing the history indices to query for each environment.
                             0 = most recent observation, 1 = one step ago, etc.
            key (str, optional): If provided, query only this specific key. If None, return all keys.
            
        Returns:
            If key is provided: Tensor of shape (num_envs, *obs_dims[key])
            If key is None: Dict of {key: Tensor of shape (num_envs, *obs_dims[key])}
        """
        assert indices.shape[0] == self.num_envs, f"Expected indices of shape ({self.num_envs},), got {indices.shape}"
        
        # Create a batch index for advanced indexing
        batch_idx = torch.arange(self.num_envs, device=self.device)
        
        if key is not None:
            assert key in self.history.keys(), f"Key {key} not found in history"
            return self.history[key][batch_idx, indices]
        else:
            result = {}
            for k in self.history.keys():
                result[k] = self.history[k][batch_idx, indices]
            return result