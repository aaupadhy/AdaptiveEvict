import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from datetime import datetime

class KVCacheVisualizer:
    def __init__(self, run_name):
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
        self.plot_dir = f"plots/{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.plot_dir, exist_ok=True)

    def update_metrics(self, metrics):
        for key in self.metrics_history:
            if key in metrics:
                self.metrics_history[key].append(metrics[key])

    def plot_training_progress(self, episode):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Training Progress - Episode {episode}')

        axes[0, 0].plot(self.metrics_history['reward'], label='Reward')
        axes[0, 0].plot(self.metrics_history['cost'], label='Cost')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(self.metrics_history['primary_utilization'], label='Primary Cache')
        axes[0, 1].plot(self.metrics_history['secondary_utilization'], label='Secondary Cache')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[1, 0].plot(self.metrics_history['semantic_relevance'], label='Semantic Relevance')
        axes[1, 0].plot(self.metrics_history['cache_miss_rate'], label='Cache Miss Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        axes[1, 1].plot(self.metrics_history['eviction_count'], label='Evictions')
        axes[1, 1].plot(self.metrics_history['retrieval_count'], label='Retrievals')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/training_progress_episode_{episode}.png")
        plt.close()

    def plot_cache_state(self, primary_cache, secondary_cache):
        fig, ax = plt.subplots(figsize=(10, 6))
        primary_utilization = len(primary_cache['keys']) / max(1, len(primary_cache['keys']))
        secondary_utilization = len(secondary_cache['keys']) / max(1, len(secondary_cache['keys']))
        ax.bar(['Primary Cache', 'Secondary Cache'], [primary_utilization, secondary_utilization])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Utilization')
        plt.title('Cache Utilization')
        plt.savefig(f"{self.plot_dir}/cache_state.png")
        plt.close()

    def plot_reward_components(self, episode):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.metrics_history['reward'], label='Total Reward')
        ax.plot(self.metrics_history['cost'], label='Cost')
        ax.plot(self.metrics_history['semantic_relevance'], label='Semantic Relevance')
        ax.plot(self.metrics_history['cache_miss_rate'], label='Cache Miss Rate')
        ax.legend()
        ax.grid(True)
        plt.title(f'Reward Components - Episode {episode}')
        plt.savefig(f"{self.plot_dir}/reward_components_episode_{episode}.png")
        plt.close()

    def log_summary_statistics(self, episode):
        stats = {
            'episode': episode,
            'avg_reward': np.mean(self.metrics_history['reward'][-100:]) if self.metrics_history['reward'] else 0,
            'avg_cost': np.mean(self.metrics_history['cost'][-100:]) if self.metrics_history['cost'] else 0,
            'avg_semantic_relevance': np.mean(self.metrics_history['semantic_relevance'][-100:]) if self.metrics_history['semantic_relevance'] else 0,
            'avg_cache_miss_rate': np.mean(self.metrics_history['cache_miss_rate'][-100:]) if self.metrics_history['cache_miss_rate'] else 0
        }
        
        with open(f"{self.plot_dir}/summary_stats.txt", 'a') as f:
            f.write(f"Episode {episode} Summary:\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
