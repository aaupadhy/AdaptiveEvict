import torch
import numpy as np
import wandb

class KVCacheStorage:
    def __init__(self, max_primary_size, max_secondary_size):
        self.max_primary_size = max_primary_size
        self.max_secondary_size = max_secondary_size
        
        self.primary_cache = {
            'keys': [],
            'values': [],
            'positions': [],
            'attention_scores': []
        }
        
        self.secondary_cache = {
            'keys': [],
            'values': [],
            'positions': [],
            'attention_scores': []
        }
        
    def add_to_primary(self, key, value, position, attention_score):
        if len(self.primary_cache['keys']) >= self.max_primary_size:
            return False
            
        self.primary_cache['keys'].append(key)
        self.primary_cache['values'].append(value)
        self.primary_cache['positions'].append(position)
        self.primary_cache['attention_scores'].append(attention_score)
        return True
        
    def evict_to_secondary(self, indices):
        if len(self.secondary_cache['keys']) + len(indices) > self.max_secondary_size:
            return False
            
        for idx in sorted(indices, reverse=True):
            self.secondary_cache['keys'].append(self.primary_cache['keys'].pop(idx))
            self.secondary_cache['values'].append(self.primary_cache['values'].pop(idx))
            self.secondary_cache['positions'].append(self.primary_cache['positions'].pop(idx))
            self.secondary_cache['attention_scores'].append(self.primary_cache['attention_scores'].pop(idx))
        return True
        
    def retrieve_from_secondary(self, indices):
        if len(self.primary_cache['keys']) + len(indices) > self.max_primary_size:
            return False
            
        for idx in sorted(indices, reverse=True):
            self.primary_cache['keys'].append(self.secondary_cache['keys'].pop(idx))
            self.primary_cache['values'].append(self.secondary_cache['values'].pop(idx))
            self.primary_cache['positions'].append(self.secondary_cache['positions'].pop(idx))
            self.primary_cache['attention_scores'].append(self.secondary_cache['attention_scores'].pop(idx))
        return True
        
    def get_state_features(self):
        primary_features = {
            'size': len(self.primary_cache['keys']),
            'mean_attention': np.mean(self.primary_cache['attention_scores']) if self.primary_cache['attention_scores'] else 0,
            'max_attention': np.max(self.primary_cache['attention_scores']) if self.primary_cache['attention_scores'] else 0,
            'position_range': (min(self.primary_cache['positions']), max(self.primary_cache['positions'])) if self.primary_cache['positions'] else (0, 0)
        }
        
        secondary_features = {
            'size': len(self.secondary_cache['keys']),
            'mean_attention': np.mean(self.secondary_cache['attention_scores']) if self.secondary_cache['attention_scores'] else 0,
            'max_attention': np.max(self.secondary_cache['attention_scores']) if self.secondary_cache['attention_scores'] else 0,
            'position_range': (min(self.secondary_cache['positions']), max(self.secondary_cache['positions'])) if self.secondary_cache['positions'] else (0, 0)
        }
        
        return {
            'primary': primary_features,
            'secondary': secondary_features,
            'primary_utilization': primary_features['size'] / self.max_primary_size,
            'secondary_utilization': secondary_features['size'] / self.max_secondary_size
        }
        
    def log_metrics(self):
        state_features = self.get_state_features()
        wandb.log({
            'primary_cache_size': state_features['primary']['size'],
            'secondary_cache_size': state_features['secondary']['size'],
            'primary_mean_attention': state_features['primary']['mean_attention'],
            'secondary_mean_attention': state_features['secondary']['mean_attention'],
            'primary_utilization': state_features['primary_utilization'],
            'secondary_utilization': state_features['secondary_utilization']
        }) 