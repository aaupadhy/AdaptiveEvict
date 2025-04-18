import wandb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict, Any

class KVCacheVisualizer:
    def __init__(self, run_name: str):
        self.run_name = run_name
        self.metrics_history = {
            'reward': [],
            'cost': [],
            'semantic_relevance': [],
            'cache_miss_rate': [],
            'primary_utilization': [],
            'secondary_utilization': [],
            'eviction_count': [],
            'retrieval_count': []
        }
        
    def update_metrics(self, metrics: Dict[str, float]):
        for key in self.metrics_history:
            if key in metrics:
                self.metrics_history[key].append(metrics[key])
                
    def plot_training_progress(self, episode: int):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'KV Cache RL Training Progress - Episode {episode}', fontsize=16)
        
        # Plot 1: Reward and Cost
        ax1 = axes[0, 0]
        ax1.plot(self.metrics_history['reward'], label='Reward', color='blue')
        ax1.plot(self.metrics_history['cost'], label='Cost', color='red')
        ax1.set_title('Reward and Cost Over Time')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Cache Performance
        ax2 = axes[0, 1]
        ax2.plot(self.metrics_history['primary_utilization'], label='Primary Cache', color='green')
        ax2.plot(self.metrics_history['secondary_utilization'], label='Secondary Cache', color='orange')
        ax2.set_title('Cache Utilization')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Utilization Rate')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Semantic Relevance and Cache Miss Rate
        ax3 = axes[1, 0]
        ax3.plot(self.metrics_history['semantic_relevance'], label='Semantic Relevance', color='purple')
        ax3.plot(self.metrics_history['cache_miss_rate'], label='Cache Miss Rate', color='brown')
        ax3.set_title('Semantic Performance')
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Score')
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: Eviction and Retrieval Operations
        ax4 = axes[1, 1]
        ax4.plot(self.metrics_history['eviction_count'], label='Evictions', color='red')
        ax4.plot(self.metrics_history['retrieval_count'], label='Retrievals', color='blue')
        ax4.set_title('Cache Operations')
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('Count')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        wandb.log({
            "training_progress": wandb.Image(fig),
            "episode": episode
        })
        plt.close()
        
    def plot_attention_heatmap(self, attention_scores: List[float], positions: List[int]):
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            np.array(attention_scores).reshape(-1, 1),
            cmap='YlOrRd',
            xticklabels=['Attention'],
            yticklabels=positions,
            cbar_kws={'label': 'Attention Score'}
        )
        plt.title('Token Attention Scores')
        plt.xlabel('Attention Dimension')
        plt.ylabel('Token Position')
        wandb.log({"attention_heatmap": wandb.Image(plt)})
        plt.close()
        
    def plot_cache_state(self, primary_cache: Dict[str, Any], secondary_cache: Dict[str, Any]):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Primary Cache State
        primary_positions = primary_cache['positions']
        primary_attention = primary_cache['attention_scores']
        ax1.scatter(primary_positions, primary_attention, c='blue', alpha=0.6)
        ax1.set_title('Primary Cache State')
        ax1.set_xlabel('Token Position')
        ax1.set_ylabel('Attention Score')
        ax1.grid(True)
        
        # Secondary Cache State
        secondary_positions = secondary_cache['positions']
        secondary_attention = secondary_cache['attention_scores']
        ax2.scatter(secondary_positions, secondary_attention, c='red', alpha=0.6)
        ax2.set_title('Secondary Cache State')
        ax2.set_xlabel('Token Position')
        ax2.set_ylabel('Attention Score')
        ax2.grid(True)
        
        plt.tight_layout()
        wandb.log({"cache_state": wandb.Image(fig)})
        plt.close()
        
    def plot_reward_components(self, episode: int):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        semantic_reward = np.array(self.metrics_history['semantic_relevance']) * 0.3
        cache_miss_penalty = -np.array(self.metrics_history['cache_miss_rate']) * 0.5
        cost_penalty = -np.array(self.metrics_history['cost'])
        
        x = np.arange(len(semantic_reward))
        width = 0.25
        
        ax.bar(x - width, semantic_reward, width, label='Semantic Reward', color='green')
        ax.bar(x, cache_miss_penalty, width, label='Cache Miss Penalty', color='red')
        ax.bar(x + width, cost_penalty, width, label='Cost Penalty', color='blue')
        
        ax.set_title('Reward Components Breakdown')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        wandb.log({
            "reward_components": wandb.Image(fig),
            "episode": episode
        })
        plt.close()
        
    def log_summary_statistics(self, episode: int):
        summary_stats = {
            "mean_reward": np.mean(self.metrics_history['reward'][-100:]),
            "mean_semantic_relevance": np.mean(self.metrics_history['semantic_relevance'][-100:]),
            "mean_cache_miss_rate": np.mean(self.metrics_history['cache_miss_rate'][-100:]),
            "mean_primary_utilization": np.mean(self.metrics_history['primary_utilization'][-100:]),
            "mean_secondary_utilization": np.mean(self.metrics_history['secondary_utilization'][-100:]),
            "total_evictions": sum(self.metrics_history['eviction_count']),
            "total_retrievals": sum(self.metrics_history['retrieval_count'])
        }
        
        wandb.log({
            "summary_statistics": summary_stats,
            "episode": episode
        }) 