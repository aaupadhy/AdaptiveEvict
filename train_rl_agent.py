import torch
import numpy as np
from sac_agent import SACAgent
from kv_cache_env import KVCacheEnv
from llama import LLAMA
from visualization import KVCacheVisualizer
import argparse
import json
import pickle
import random
import os
import logging
from tqdm import tqdm
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_training_data(data_path):
    logging.info(f"Loading training data from {data_path}")
    with open(data_path, 'r') as f:
        return json.load(f)

def preprocess_training_data(training_data, max_seq_len):
    preprocessed_data = []
    for sample in training_data:
        # Truncate prompts that exceed max_seq_len
        truncated_prompt = sample['prompt'][:max_seq_len]
        preprocessed_data.append({
            'prompt': truncated_prompt,
            'length': len(truncated_prompt.split())
        })
    return preprocessed_data

def generate_prompts(training_data, num_envs):
    return [random.choice(training_data)['prompt'] for _ in range(num_envs)]

def train(args):
    logging.info("Starting training process")
    wandb.init(project="adaptive_evict_llm", config=vars(args))
    logging.info("Initialized Weights & Biases run")
    torch.cuda.empty_cache()
    visualizer = KVCacheVisualizer("training_run")

    logging.info("Preprocessing training data")
    training_data = preprocess_training_data(load_training_data(args.training_data_path), args.max_seq_len)

    logging.info("Loading tokenizer")
    with open("saved_models/tokenizer.pt", "rb") as f:
        tokenizer = pickle.load(f)

    if args.vocab_size is None:
        args.vocab_size = tokenizer.n_tokens

    logging.info("Initializing LLAMA model")
    llama_model = LLAMA(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        max_seq_len=args.max_seq_len,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        forward_mul=args.forward_mul,
        kv_cache=True,
        dropout=args.dropout
    ).to(device).to(dtype=torch.float16)

    if os.path.exists("saved_models/llama.pt"):
        logging.info("Loading saved LLAMA model state")
        state_dict = torch.load("saved_models/llama.pt", map_location=device)
        # Filter out mismatched keys
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in llama_model.state_dict() and llama_model.state_dict()[k].shape == v.shape}
        llama_model.load_state_dict(filtered_state_dict, strict=False)
        logging.info(f"Loaded llama.pt with filtered keys: {len(filtered_state_dict)} keys loaded.")
    else:
        logging.warning("llama.pt not found! Model initialized randomly.")

    llama_model.eval()
    llama_model.tokenizer = tokenizer

    logging.info("Initializing KVCache environment")
    env = KVCacheEnv(
        llama_model=llama_model,
        max_primary_size=args.max_primary_size,
        max_secondary_size=args.max_secondary_size,
        lambda_cost=args.lambda_cost,
        semantic_weight=args.semantic_weight,
        cache_miss_penalty=args.cache_miss_penalty,
        perplexity_weight=args.perplexity_weight,
        attention_weight=args.attention_weight,
        gradient_weight=args.gradient_weight,
        sampling_temperature=args.sampling_temperature,
        turn_length=args.turn_length,
        num_envs=args.num_envs
    )

    logging.info("Initializing SAC agent")
    agent = SACAgent(
        state_dim=env.get_state_space(),
        action_dim=env.get_action_space(),
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        device=device
    )

    wandb.watch(agent.actor, log="all")
    wandb.watch(agent.critic1, log="all")
    wandb.watch(agent.critic2, log="all")

    state_buffer, action_buffer, reward_buffer = [], [], []
    next_state_buffer, done_buffer = [], []

    for episode in tqdm(range(args.num_episodes), desc="Training Episodes"):
        logging.info(f"Starting episode {episode+1}")
        prompts = generate_prompts(training_data, args.num_envs)
        logging.debug(f"Prompts: {prompts}")
        states = env.reset(prompts)
        episode_rewards = torch.zeros(args.num_envs, device=device)
        episode_metrics = []

        for step in range(args.max_steps):
            actions = agent.select_action(states)
            next_states, rewards, dones, infos = env.step(actions)
            logging.debug(f"Step {step}: actions {actions.detach().cpu().numpy()}, rewards {rewards.detach().cpu().numpy()}, dones {dones.detach().cpu().numpy()}")

            state_buffer.extend(states.detach().cpu().numpy())
            action_buffer.extend(actions.detach().cpu().numpy())
            reward_buffer.extend(rewards.detach().cpu().numpy())
            next_state_buffer.extend(next_states.detach().cpu().numpy())
            done_buffer.extend(dones.detach().cpu().numpy())

            episode_rewards += rewards
            episode_metrics.extend(infos)

            if len(state_buffer) >= args.batch_size:
                agent.update(
                    torch.from_numpy(np.array(state_buffer, dtype=np.float32)).to(device),
                    torch.from_numpy(np.array(action_buffer, dtype=np.float32)).to(device),
                    torch.from_numpy(np.array(reward_buffer, dtype=np.float32)).to(device),
                    torch.from_numpy(np.array(next_state_buffer, dtype=np.float32)).to(device),
                    torch.from_numpy(np.array(done_buffer, dtype=np.float32)).to(device)
                )
                state_buffer, action_buffer = [], []
                reward_buffer, next_state_buffer, done_buffer = [], [], []

            if dones.any():
                logging.info(f"Episode {episode+1} finished after {step+1} steps with average reward {episode_rewards.mean().item():.2f}")
                break

            states = next_states

        for metrics in episode_metrics:
            visualizer.update_metrics(metrics)
        visualizer.plot_training_progress(episode)
        visualizer.plot_reward_components(episode)
        visualizer.plot_cache_state(env.storage[0].primary_cache, env.storage[0].secondary_cache)
        visualizer.log_summary_statistics(episode)
        
        print(f"Episode {episode+1} | Average Reward: {episode_rewards.mean().item():.2f} | Steps: {step+1}", flush=True)
        
        if (episode + 1) % args.save_interval == 0:
            logging.info(f"Saving SAC agent model at episode {episode+1}")
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic1_state_dict': agent.critic1.state_dict(),
                'critic2_state_dict': agent.critic2.state_dict(),
                'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                'critic1_optimizer_state_dict': agent.critic1_optimizer.state_dict(),
                'critic2_optimizer_state_dict': agent.critic2_optimizer.state_dict()
            }, f'saved_models/sac_agent_episode_{episode+1}.pt')

        mean_semantic = np.mean([m['semantic_relevance'] for m in episode_metrics])
        mean_primary_size = np.mean([m['primary_size'] for m in episode_metrics])
        mean_secondary_size = np.mean([m['secondary_size'] for m in episode_metrics])
        mean_cache_misses = np.mean([m['cache_misses'] for m in episode_metrics])
        mean_total_requests = np.mean([m['total_requests'] for m in episode_metrics])
        mean_cost = np.mean([m['cost'] for m in episode_metrics])
        mean_cache_miss_rate = np.mean([m['cache_miss_rate'] for m in episode_metrics])
        wandb.log({
            'episode': episode+1,
            'avg_reward': episode_rewards.mean().item(),
            'steps': step+1,
            'avg_semantic_relevance': mean_semantic,
            'avg_primary_size': mean_primary_size,
            'avg_secondary_size': mean_secondary_size,
            'avg_cache_misses': mean_cache_misses,
            'avg_total_requests': mean_total_requests,
            'avg_cost': mean_cost,
            'avg_cache_miss_rate': mean_cache_miss_rate
        })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data_path', type=str, required=True)
    parser.add_argument('--vocab_size', type=int, default=None)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--max_seq_len', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--forward_mul', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_primary_size', type=int, default=64)
    parser.add_argument('--max_secondary_size', type=int, default=128)
    parser.add_argument('--lambda_cost', type=float, default=0.1)
    parser.add_argument('--semantic_weight', type=float, default=0.3)
    parser.add_argument('--cache_miss_penalty', type=float, default=0.5)
    parser.add_argument('--perplexity_weight', type=float, default=0.2)
    parser.add_argument('--attention_weight', type=float, default=0.3)
    parser.add_argument('--gradient_weight', type=float, default=0.2)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--turn_length', type=int, default=10)
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--sampling_temperature', type=float, default=1.0)

    args = parser.parse_args()
    train(args)
