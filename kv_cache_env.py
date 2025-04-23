import torch
import torch.nn.functional as F
import numpy as np
import wandb
from kv_storage import KVCacheStorage

class KVCacheEnv:
    def __init__(self, llama_model, max_primary_size, max_secondary_size, 
                 lambda_cost=0.1, semantic_weight=0.3, cache_miss_penalty=0.5,
                 perplexity_weight=0.2, attention_weight=0.3, gradient_weight=0.2):
        self.llama_model = llama_model
        self.storage = KVCacheStorage(max_primary_size, max_secondary_size)
        self.lambda_cost = lambda_cost
        self.semantic_weight = semantic_weight
        self.cache_miss_penalty = cache_miss_penalty
        self.perplexity_weight = perplexity_weight
        self.attention_weight = attention_weight
        self.gradient_weight = gradient_weight
        self.current_step = 0
        self.max_steps = 1000
        self.cache_misses = 0
        self.total_requests = 0
        self.semantic_scores = []
        self.current_generated_tokens = []
        self.prompt = None
        self.reward_history = {
            'perplexity': [],
            'attention': [],
            'gradient': [],
            'semantic': [],
            'cache_miss': [],
            'cost': []
        }

    def reset(self, prompt=None):
        self.storage = KVCacheStorage(self.storage.max_primary_size, self.storage.max_secondary_size)
        self.current_step = 0
        self.cache_misses = 0
        self.total_requests = 0
        self.semantic_scores = []
        self.current_generated_tokens = []
        self.prompt = prompt
        self.reward_history = {k: [] for k in self.reward_history}
        self.llama_model.reset_cache()

        if prompt is not None:
            input_ids = self.llama_model.tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids])
            with torch.no_grad():
                outputs = self.llama_model(input_tensor)
            predicted_ids = outputs.argmax(dim=-1).squeeze(0).tolist()
            self.current_generated_tokens = input_ids + predicted_ids[:5]  # simulate 5 tokens generation
            # manually push into cache for now
            self.storage.primary_cache['keys'].extend(self.current_generated_tokens[:10])

        return self._get_state()

    def _get_state(self):
        state_features = self.storage.get_state_features()

        state = np.array([
            state_features['primary']['size'] / self.storage.max_primary_size,
            state_features['secondary']['size'] / self.storage.max_secondary_size,
            state_features['primary']['mean_attention'],
            state_features['secondary']['mean_attention'],
            state_features['primary']['max_attention'],
            state_features['secondary']['max_attention'],
            state_features['primary']['position_range'][0] / self.max_steps,
            state_features['primary']['position_range'][1] / self.max_steps,
            state_features['secondary']['position_range'][0] / self.max_steps,
            state_features['secondary']['position_range'][1] / self.max_steps,
            self.current_step / self.max_steps,
            self.cache_misses / max(1, self.total_requests),
            np.mean(self.semantic_scores) if self.semantic_scores else 0
        ])

        return state

    def _calculate_semantic_relevance(self, token_id, context_ids):
        return np.random.rand()  # fake a random semantic relevance for now

    def _check_cache_miss(self, token_id):
        self.total_requests += 1
        if token_id not in self.storage.primary_cache['keys'] and token_id not in self.storage.secondary_cache['keys']:
            self.cache_misses += 1
            return True
        return False

    def _calculate_perplexity_reward(self):
        # Use the model's perplexity on the current context as a reward signal
        perplexity = self.llama_model.get_perplexity(self.current_context)
        return -perplexity  # Lower perplexity is better

    def _calculate_attention_reward(self):
        # Use the average attention score of retained tokens as a reward signal
        attention_scores = self.llama_model.get_last_layer_attention()
        avg_attention = attention_scores.mean().item()
        return avg_attention  # Higher attention is better

    def _calculate_gradient_reward(self):
        # Use the gradient norm of token embeddings as a reward signal
        token_gradients = self.llama_model.get_token_gradients()
        gradient_norm = token_gradients.norm().item() if token_gradients is not None else 0
        return -gradient_norm  # Lower gradient norm is better

    def calculate_reward(self):
        # Calculate individual reward components
        perplexity_reward = self._calculate_perplexity_reward() * self.perplexity_weight
        attention_reward = self._calculate_attention_reward() * self.attention_weight
        gradient_reward = self._calculate_gradient_reward() * self.gradient_weight

        # Calculate cost and cache miss penalties
        cost_reward = -self.current_cost * self.lambda_cost
        cache_miss_reward = -self.cache_miss_rate * self.cache_miss_penalty

        # Combine all components into a single reward
        total_reward = (
            perplexity_reward +
            attention_reward +
            gradient_reward +
            cost_reward +
            cache_miss_reward
        )

        # Log reward components for debugging
        self.reward_history['perplexity'].append(perplexity_reward)
        self.reward_history['attention'].append(attention_reward)
        self.reward_history['gradient'].append(gradient_reward)
        self.reward_history['cost'].append(cost_reward)
        self.reward_history['cache_miss'].append(cache_miss_reward)

        return total_reward

    def step(self, action):
        self.current_step += 1

        evict_indices = []
        retrieve_indices = []

        for i, a in enumerate(action):
            if a < -0.5 and i < len(self.storage.primary_cache['keys']):
                evict_indices.append(i)
            elif a > 0.5 and i < len(self.storage.secondary_cache['keys']):
                retrieve_indices.append(i)

        eviction_success = self.storage.evict_to_secondary(evict_indices)
        retrieval_success = self.storage.retrieve_from_secondary(retrieve_indices)

        primary_size = len(self.storage.primary_cache['keys'])
        secondary_size = len(self.storage.secondary_cache['keys'])

        cost = (primary_size / self.storage.max_primary_size) * self.lambda_cost
        self.current_cost = cost
        cache_miss_rate = self.cache_misses / max(1, self.total_requests)
        self.cache_miss_rate = cache_miss_rate

        reward = self.calculate_reward()

        semantic_relevance = 0.0
        if self.storage.primary_cache['keys']:
            current_token = self.storage.primary_cache['keys'][-1]
            context_tokens = self.storage.primary_cache['keys'][:-1]
            semantic_relevance = self._calculate_semantic_relevance(current_token, context_tokens)
            self.semantic_scores.append(semantic_relevance)
            self.reward_history['semantic'].append(semantic_relevance)

        if not eviction_success or not retrieval_success:
            reward -= 1.0

        done = self.current_step >= self.max_steps
        next_state = self._get_state()

        self.storage.log_metrics()
        wandb.log({
            'step': self.current_step,
            'total_reward': reward,
            'cost_reward': self.reward_history['cost'][-1],
            'semantic_reward': semantic_relevance * self.semantic_weight,
            'cache_miss_reward': self.reward_history['cache_miss'][-1],
            'perplexity_reward': self.reward_history['perplexity'][-1],
            'attention_reward': self.reward_history['attention'][-1],
            'gradient_reward': self.reward_history['gradient'][-1],
            'eviction_count': len(evict_indices),
            'retrieval_count': len(retrieve_indices),
            'primary_cache_utilization': primary_size / self.storage.max_primary_size,
            'secondary_cache_utilization': secondary_size / self.storage.max_secondary_size
        })

        return next_state, reward, done, {}

    def get_action_space(self):
        return self.storage.max_primary_size + self.storage.max_secondary_size

    def get_state_space(self):
        return 13
