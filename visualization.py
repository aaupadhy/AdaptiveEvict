import wandb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
        wandb.log({"training_progress": wandb.Image(fig)})
        plt.close()

    def plot_cache_state(self, primary_cache, secondary_cache):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        if primary_cache['positions']:
            ax1.scatter(primary_cache['positions'], primary_cache['attention_scores'], color='blue')
        ax1.set_title('Primary Cache')
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Attention')

        if secondary_cache['positions']:
            ax2.scatter(secondary_cache['positions'], secondary_cache['attention_scores'], color='red')
        ax2.set_title('Secondary Cache')
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Attention')

        plt.tight_layout()
        wandb.log({"cache_state": wandb.Image(fig)})
        plt.close()

    def plot_reward_components(self, episode):
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(self.metrics_history['reward']))

        if len(x) == 0:
            return

        semantic = np.array(self.metrics_history['semantic_relevance']) * 0.3
        miss = -np.array(self.metrics_history['cache_miss_rate']) * 0.5
        cost = -np.array(self.metrics_history['cost'])

        ax.bar(x - 0.2, semantic, 0.2, label='Semantic')
        ax.bar(x, miss, 0.2, label='Cache Miss Penalty')
        ax.bar(x + 0.2, cost, 0.2, label='Cost Penalty')

        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        wandb.log({"reward_components": wandb.Image(fig)})
        plt.close()

    def log_summary_statistics(self, episode):
        summary = {
            "mean_reward": np.nanmean(self.metrics_history['reward'][-100:]),
            "mean_semantic_relevance": np.nanmean(self.metrics_history['semantic_relevance'][-100:]),
            "mean_cache_miss_rate": np.nanmean(self.metrics_history['cache_miss_rate'][-100:]),
            "mean_primary_utilization": np.nanmean(self.metrics_history['primary_utilization'][-100:]),
            "mean_secondary_utilization": np.nanmean(self.metrics_history['secondary_utilization'][-100:]),
            "total_evictions": sum(self.metrics_history['eviction_count']),
            "total_retrievals": sum(self.metrics_history['retrieval_count']),
        }
        wandb.log({"summary_statistics": summary})
