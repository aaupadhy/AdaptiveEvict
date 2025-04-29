import torch
import torch.nn.functional as F
from kv_storage import KVCacheStorage
from llama import LLAMA

class KVCacheEnv:
    def __init__(self, llama_model, max_primary_size, max_secondary_size, 
                 lambda_cost=0.1, semantic_weight=0.3, cache_miss_penalty=0.5,
                 perplexity_weight=0.2, attention_weight=0.3, gradient_weight=0.2, turn_length=20, sampling_temperature=1.0,
                 num_envs=8):
        self.llama_model = llama_model
        self.device = next(llama_model.parameters()).device
        self.num_envs = num_envs
        self.storage = [KVCacheStorage(max_primary_size, max_secondary_size) for _ in range(num_envs)]
        self.lambda_cost = lambda_cost
        self.semantic_weight = semantic_weight
        self.cache_miss_penalty = cache_miss_penalty
        self.perplexity_weight = perplexity_weight
        self.attention_weight = attention_weight
        self.gradient_weight = gradient_weight
        self.turn_length = turn_length
        self.sampling_temperature = sampling_temperature
        self.current_step = torch.zeros(num_envs, device=self.device)
        self.max_steps = 1000
        self.cache_misses = torch.zeros(num_envs, device=self.device)
        self.total_requests = torch.zeros(num_envs, device=self.device)
        self.semantic_scores = [[] for _ in range(num_envs)]
        self.current_generated_tokens = [[] for _ in range(num_envs)]
        self.prompt = [None] * num_envs
        self.reward_history = {
            'perplexity': [[] for _ in range(num_envs)],
            'attention': [[] for _ in range(num_envs)],
            'gradient': [[] for _ in range(num_envs)],
            'semantic': [[] for _ in range(num_envs)],
            'cache_miss': [[] for _ in range(num_envs)],
            'cost': [[] for _ in range(num_envs)]
        }
        self.token_log_probs_sum = torch.zeros(num_envs, device=self.device)
        self.token_count = torch.zeros(num_envs, device=self.device)

    def reset(self, prompt=None):
        for i in range(self.num_envs):
            self.storage[i] = KVCacheStorage(self.storage[i].max_primary_size, self.storage[i].max_secondary_size)
            self.current_step[i] = 0
            self.cache_misses[i] = 0
            self.total_requests[i] = 0
            self.semantic_scores[i] = []
            self.current_generated_tokens[i] = []
            self.prompt[i] = prompt[i] if prompt is not None else None
            self.reward_history = {k: [[] for _ in range(self.num_envs)] for k in self.reward_history}
            self.token_log_probs_sum[i] = 0.0
            self.token_count[i] = 0

            if self.prompt[i] is not None:
                input_ids = self.llama_model.tokenizer.encode(self.prompt[i])
                self.current_generated_tokens[i] = input_ids.copy()
                
                input_tensor = torch.tensor([input_ids], device=self.device)
                self.llama_model(input_tensor, kv_cache=False)
                
                for idx, token_id in enumerate(input_ids):
                    self.storage[i].primary_cache['keys'].append(token_id)
                    self.storage[i].primary_cache['values'].append(
                        self.llama_model.get_token_embedding(token_id).to(dtype=torch.float16)
                    )
                    attn_score = self.llama_model.get_last_layer_attention().mean().item()
                    self.storage[i].primary_cache['attention_scores'].append(attn_score)
                    self.storage[i].primary_cache['positions'].append(idx)
                self._sync_model_cache(i)

        return self._get_state()

    def _get_state(self):
        states = []
        for i in range(self.num_envs):
            state_features = self.storage[i].get_state_features()
            state = torch.tensor([
                state_features['primary']['size'] / self.storage[i].max_primary_size,
                state_features['secondary']['size'] / self.storage[i].max_secondary_size,
                state_features['primary']['mean_attention'],
                state_features['secondary']['mean_attention'],
                state_features['primary']['max_attention'],
                state_features['secondary']['max_attention'],
                state_features['primary']['position_range'][0] / self.max_steps,
                state_features['primary']['position_range'][1] / self.max_steps,
                state_features['secondary']['position_range'][0] / self.max_steps,
                state_features['secondary']['position_range'][1] / self.max_steps,
                self.current_step[i].item() / self.max_steps,
                self.cache_misses[i].item() / max(1, self.total_requests[i].item()),
                (sum(self.semantic_scores[i]) / len(self.semantic_scores[i]) if self.semantic_scores[i] else 0.0)
            ], device=self.device, dtype=torch.float32)
            states.append(state)
        return torch.stack(states).float()

    def _calculate_semantic_relevance(self, token_id, context_ids):
        token_embedding = self.llama_model.get_token_embedding(token_id)
        if not context_ids:
            return 0.0
        context_embeddings = torch.stack([self.llama_model.get_token_embedding(cid) for cid in context_ids], dim=0)
        avg_context_embedding = context_embeddings.mean(dim=0)
        similarity = F.cosine_similarity(token_embedding.unsqueeze(0), avg_context_embedding.unsqueeze(0), dim=1).item()
        return similarity

    def _check_cache_miss(self, env_index, token_id):
        self.total_requests[env_index] += 1
        storage = self.storage[env_index]
        if token_id not in storage.primary_cache['keys'] and token_id not in storage.secondary_cache['keys']:
            self.cache_misses[env_index] += 1
            return True
        return False

    def _calculate_perplexity_reward(self):
        if self.token_count.sum().item() > 0:
            avg_logp = self.token_log_probs_sum / self.token_count.sum()
            return -avg_logp
        else:
            return 0.0

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

    def calculate_reward(self, env_index, cost):
        if self.token_count[env_index].item() > 0:
            avg_logp = self.token_log_probs_sum[env_index] / self.token_count[env_index]
            perplexity_reward = -avg_logp.item() * self.perplexity_weight
        else:
            perplexity_reward = 0.0

        attention_reward = self._calculate_attention_reward() * self.attention_weight
        gradient_reward = self._calculate_gradient_reward() * self.gradient_weight
        semantic_reward = (sum(self.semantic_scores[env_index]) / len(self.semantic_scores[env_index]) if self.semantic_scores[env_index] else 0.0) * self.semantic_weight

        cost_reward = -cost * self.lambda_cost
        miss_rate = self.cache_misses[env_index].item() / max(1, self.total_requests[env_index].item())
        cache_miss_reward = -miss_rate * self.cache_miss_penalty

        self.reward_history['perplexity'][env_index].append(perplexity_reward)
        self.reward_history['attention'][env_index].append(attention_reward)
        self.reward_history['gradient'][env_index].append(gradient_reward)
        self.reward_history['semantic'][env_index].append(semantic_reward)
        self.reward_history['cost'][env_index].append(cost_reward)
        self.reward_history['cache_miss'][env_index].append(cache_miss_reward)

        return perplexity_reward + attention_reward + gradient_reward + semantic_reward + cost_reward + cache_miss_reward

    def step(self, actions):
        self.current_step += 1
        next_states = []
        rewards = []
        dones = []
        infos = []

        self.reward_history = {k: [[] for _ in range(self.num_envs)] for k in self.reward_history}
        for i in range(self.num_envs):
            evict_indices, retrieve_indices = [], []
            for j, a in enumerate(actions[i]):
                if a < -0.5 and j < len(self.storage[i].primary_cache['keys']):
                    evict_indices.append(j)
                elif a > 0.5 and j < len(self.storage[i].secondary_cache['keys']):
                    retrieve_indices.append(j)

            eviction_success = self.storage[i].evict_to_secondary(evict_indices)
            retrieval_success = self.storage[i].retrieve_from_secondary(retrieve_indices)
            self._sync_model_cache(i)
            self.token_log_probs_sum[i] = 0.0
            self.token_count[i] = 0
            self.semantic_scores[i] = []

            for _ in range(self.turn_length):
                last_token = self.current_generated_tokens[i][-1]
                input_tensor = torch.tensor([[last_token]], device=self.device)
                with torch.no_grad():
                    logits = self.llama_model(input_tensor, kv_cache=True)
                probs = F.softmax(logits[0,0] / self.sampling_temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
                log_prob_selected = torch.log(probs[next_id])
                self.token_log_probs_sum[i] += log_prob_selected.item()
                self.token_count[i] += 1
                self.current_generated_tokens[i].append(next_id)
                missed = self._check_cache_miss(i, next_id)
                self.storage[i].primary_cache['keys'].append(next_id)
                self.storage[i].primary_cache['values'].append(
                    self.llama_model.get_token_embedding(next_id).to(dtype=torch.float16)
                )
                attn_score = self.llama_model.get_last_layer_attention().mean().item()
                self.storage[i].primary_cache['attention_scores'].append(attn_score)
                self.storage[i].primary_cache['positions'].append(len(self.current_generated_tokens[i])-1)
                sem_score = self._calculate_semantic_relevance(next_id, self.current_generated_tokens[i][:-1])
                self.semantic_scores[i].append(sem_score)

            primary_size = len(self.storage[i].primary_cache['keys'])
            secondary_size = len(self.storage[i].secondary_cache['keys'])
            cost = primary_size / self.storage[i].max_primary_size
            cache_miss_rate = self.cache_misses[i].item() / max(1, self.total_requests[i].item())

            reward = float(self.calculate_reward(i, cost))
            sem_reward = self.reward_history['semantic'][i][-1]
            next_state = self._get_state()[i]
            done = self.current_step[i].item() >= self.max_steps

            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            infos.append({
                'reward': reward,
                'semantic_relevance': sem_reward,
                'primary_size': primary_size,
                'secondary_size': secondary_size,
                'cache_misses': self.cache_misses[i].item(),
                'total_requests': self.total_requests[i].item(),
                'cost': cost,
                'cache_miss_rate': cache_miss_rate
            })

        next_states_tensor = torch.stack(next_states)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.bool, device=self.device)
        return next_states_tensor, rewards_tensor, dones_tensor, infos

    def get_action_space(self):
        return self.storage[0].max_primary_size + self.storage[0].max_secondary_size

    def get_state_space(self):
        return 13

    def _sync_model_cache(self, env_index):
        self.llama_model.reset_cache()
        tokens = self.storage[env_index].primary_cache['keys']
        if not tokens:
            return
        input_tensor = torch.tensor([tokens], device=self.device)
        with torch.no_grad():
            self.llama_model(input_tensor, kv_cache=False)
