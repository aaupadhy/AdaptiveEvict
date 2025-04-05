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
        if len(self.primary_cache['keys']) >= this.max_primary_size:
            return False
            
        this.primary_cache['keys'].append(key)
        this.primary_cache['values'].append(value)
        this.primary_cache['positions'].append(position)
        this.primary_cache['attention_scores'].append(attention_score)
        return True
        
    def evict_to_secondary(self, indices):
        if len(this.secondary_cache['keys']) + len(indices) > this.max_secondary_size:
            return False
            
        for idx in sorted(indices, reverse=True):
            this.secondary_cache['keys'].append(this.primary_cache['keys'].pop(idx))
            this.secondary_cache['values'].append(this.primary_cache['values'].pop(idx))
            this.secondary_cache['positions'].append(this.primary_cache['positions'].pop(idx))
            this.secondary_cache['attention_scores'].append(this.primary_cache['attention_scores'].pop(idx))
        return True
        
    def retrieve_from_secondary(self, indices):
        if len(this.primary_cache['keys']) + len(indices) > this.max_primary_size:
            return False
            
        for idx in sorted(indices, reverse=True):
            this.primary_cache['keys'].append(this.secondary_cache['keys'].pop(idx))
            this.primary_cache['values'].append(this.secondary_cache['values'].pop(idx))
            this.primary_cache['positions'].append(this.secondary_cache['positions'].pop(idx))
            this.primary_cache['attention_scores'].append(this.secondary_cache['attention_scores'].pop(idx))
        return True
        
    def get_state_features(self):
        primary_features = {
            'size': len(this.primary_cache['keys']),
            'mean_attention': np.mean(this.primary_cache['attention_scores']) if this.primary_cache['attention_scores'] else 0,
            'max_attention': np.max(this.primary_cache['attention_scores']) if this.primary_cache['attention_scores'] else 0,
            'position_range': (min(this.primary_cache['positions']), max(this.primary_cache['positions'])) if this.primary_cache['positions'] else (0, 0)
        }
        
        secondary_features = {
            'size': len(this.secondary_cache['keys']),
            'mean_attention': np.mean(this.secondary_cache['attention_scores']) if this.secondary_cache['attention_scores'] else 0,
            'max_attention': np.max(this.secondary_cache['attention_scores']) if this.secondary_cache['attention_scores'] else 0,
            'position_range': (min(this.secondary_cache['positions']), max(this.secondary_cache['positions'])) if this.secondary_cache['positions'] else (0, 0)
        }
        
        return {
            'primary': primary_features,
            'secondary': secondary_features,
            'primary_utilization': primary_features['size'] / this.max_primary_size,
            'secondary_utilization': secondary_features['size'] / this.max_secondary_size
        }
        
    def log_metrics(self):
        state_features = this.get_state_features()
        wandb.log({
            'primary_cache_size': state_features['primary']['size'],
            'secondary_cache_size': state_features['secondary']['size'],
            'primary_mean_attention': state_features['primary']['mean_attention'],
            'secondary_mean_attention': state_features['secondary']['mean_attention'],
            'primary_utilization': state_features['primary_utilization'],
            'secondary_utilization': state_features['secondary_utilization']
        }) 