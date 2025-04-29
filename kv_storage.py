import torch
import numpy as np

class KVCacheStorage:
    def __init__(self, max_primary_size, max_secondary_size):
        self.max_primary_size = max_primary_size
        self.max_secondary_size = max_secondary_size

        self.primary_cache = {'keys': [], 'values': [], 'attention_scores': [], 'positions': []}
        self.secondary_cache = {'keys': [], 'values': [], 'attention_scores': [], 'positions': []}

    def evict_to_secondary(self, evict_indices):
        if not self.primary_cache['keys']:
            return False

        new_primary_keys, new_primary_values = [], []
        new_primary_attention, new_primary_positions = [], []
        evict_set = set(evict_indices)

        for i, (key, value, attn, pos) in enumerate(zip(
            self.primary_cache['keys'],
            self.primary_cache['values'],
            self.primary_cache['attention_scores'],
            self.primary_cache['positions']
        )):
            if i in evict_set:
                self.secondary_cache['keys'].append(key)
                self.secondary_cache['values'].append(value)
                self.secondary_cache['attention_scores'].append(attn)
                self.secondary_cache['positions'].append(pos)
            else:
                new_primary_keys.append(key)
                new_primary_values.append(value)
                new_primary_attention.append(attn)
                new_primary_positions.append(pos)

        self.primary_cache['keys'] = new_primary_keys
        self.primary_cache['values'] = new_primary_values
        self.primary_cache['attention_scores'] = new_primary_attention
        self.primary_cache['positions'] = new_primary_positions

        return True

    def retrieve_from_secondary(self, retrieve_indices):
        if not self.secondary_cache['keys']:
            return False

        new_secondary_keys, new_secondary_values = [], []
        new_secondary_attention, new_secondary_positions = [], []
        retrieve_set = set(retrieve_indices)

        for i, (key, value, attn, pos) in enumerate(zip(
            self.secondary_cache['keys'],
            self.secondary_cache['values'],
            self.secondary_cache['attention_scores'],
            self.secondary_cache['positions']
        )):
            if i in retrieve_set:
                self.primary_cache['keys'].append(key)
                self.primary_cache['values'].append(value)
                self.primary_cache['attention_scores'].append(attn)
                self.primary_cache['positions'].append(pos)
            else:
                new_secondary_keys.append(key)
                new_secondary_values.append(value)
                new_secondary_attention.append(attn)
                new_secondary_positions.append(pos)

        self.secondary_cache['keys'] = new_secondary_keys
        self.secondary_cache['values'] = new_secondary_values
        self.secondary_cache['attention_scores'] = new_secondary_attention
        self.secondary_cache['positions'] = new_secondary_positions

        return True

    def get_state_features(self):
        def compute_attention(cache):
            if cache['attention_scores']:
                return sum(cache['attention_scores']) / len(cache['attention_scores']), max(cache['attention_scores'])
            else:
                return 0.0, 0.0

        primary_mean, primary_max = compute_attention(self.primary_cache)
        secondary_mean, secondary_max = compute_attention(self.secondary_cache)

        def position_range(cache):
            if cache['positions']:
                return (min(cache['positions']), max(cache['positions']))
            else:
                return (0, 0)

        return {
            'primary': {
                'size': len(self.primary_cache['keys']),
                'mean_attention': primary_mean,
                'max_attention': primary_max,
                'position_range': position_range(self.primary_cache)
            },
            'secondary': {
                'size': len(self.secondary_cache['keys']),
                'mean_attention': secondary_mean,
                'max_attention': secondary_max,
                'position_range': position_range(self.secondary_cache)
            }
        }
