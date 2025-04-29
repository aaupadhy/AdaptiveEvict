# AdaptiveEvict

## Introduction

AdaptiveEvict is a reinforcement learning (RL) framework designed to manage the key-value (KV) cache in transformer-based language models. By learning when to evict and retrieve tokens, our approach dynamically balances inference efficiency with model accuracy under strict memory constraints. This project demonstrates how an RL agent can outperform static heuristics (e.g., FIFO or attention-based methods) by explicitly optimizing a reward that captures both context retention and task performance.

## Problem Statement

Modern large language models (LLMs) such as GPT, LLaMA, and others rely on a KV cache to store past activations for efficient autoregressive generation. However, the cache has a fixed capacity and can quickly overflow as context length grows:

- **Fixed-size limits** force eviction of tokens when capacity is reached.  
- **Static eviction rules** (e.g., FIFO or simple attention heuristics) do not account for future token importance, leading to premature removal of critical context.  
- **Performance degradation** occurs in tasks requiring long-range dependencies (e.g., multi-turn dialogue, open-domain question answering) when key tokens are discarded too early.  

Existing LLM cache management strategies are often proprietary or heuristic-driven, offering no explicit trade-off between inference cost and final task accuracy. Our work addresses this gap by formulating cache management as a Markov Decision Process (MDP), training an RL agent to make eviction/retrieval decisions that maximize a composite reward:

> **Reward = log P(next token) – λ × (tokens retrieved)**

where the first term encourages predictive accuracy and the second penalizes costly retrievals from secondary storage.

## Our Methods

1. **MDP Formulation**  
   - **State Representation:** At each decision point, we extract for each token: last-layer hidden embeddings, cumulative attention scores, recency/position, and global context features, combining into a fixed-length state vector.  
   - **Action Space:** A continuous action vector specifying (a) which tokens to evict from the primary cache, (b) which tokens to retrieve from secondary storage.

2. **Environment Setup**  
   - **Model Backbone:** LLaMA (locally hosted) with KV cache enabled.  
   - **Corpus:** Cosmopedia-100k dataset (~25B tokens) for pre-training and synthetic dialogues.  
   - **Cache Tiers:** Two-tier KV cache (primary fast memory + secondary slower storage) integrated into a custom `KVCacheEnv`.  

3. **RL Agent**  
   - **Algorithm:** Soft Actor-Critic (SAC) with actor, dual critics, and target networks, chosen for sample efficiency and stability in continuous spaces.  
   - **Reward Signal:** Combines log-probability of the correct next token with a penalty for retrieval operations.  
   - **Training Loop:** At each generation step, the agent observes the cache state, issues eviction/retrieval actions, the model generates the next token, and the environment computes the reward.

4. **Evaluation Metrics**  
   - **Perplexity:** Compare RL-driven eviction vs. baselines.  
   - **Task Accuracy:** F1 on QA.  
   - **Context Efficiency:** Average # tokens retained, cache miss rates, and inference speedups.  
   - **Ablations:** Turn off retrieval, vary λ, compare different state feature sets.

## Installation (including `requirements.txt`)

1. **Clone the repository**

   ```bash
   git clone https://github.com/aaupadhy/AdaptiveEvict.git
   cd AdaptiveEvict
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure and run**  
   - Update `config.py` with your environment settings (model path, cache thresholds, λ parameter).
   - Update Slurm files with correct local paths
   - LLM Training
     ```bash
     python main.py --network_type llama --max_merged_tokens 20000 --embed_dim 256 --n_heads 8 --forward_mul 4 --n_layers 8 --batch_size 64 --train_tokens_len 256 --epochs 10 --warmup_epochs 5
     ```
   - Launch Data Generation:
     ```bash
     python generate_rl_data.py \
       --num_prompts 10000 \
       --min_length 50 \
       --max_length 200 \
       --output_file data/rl_training_data.json \
       --model_path ./saved_models \
       --data_path data \
       --data_file train.txt \
       --network_type llama \
       --load_model \
       --load_tokenizer \
       --embed_dim 256 \
       --n_layers 8 \
       --n_heads 8 \
       --forward_mul 4 \
       --train_tokens_len 256 \
       --batch_size 64 \
       --dropout 0.1 \
       --n_workers 2 \
       --max_merged_tokens 20000
     ```
   - RL Training:
     ```bash
     python train_rl_agent.py \
       --training_data_path data/rl_training_data.json \
       --vocab_size 22369 \
       --embed_dim 256 \
       --max_seq_len 128 \
       --n_layers 8 \
       --n_heads 8 \
       --forward_mul 4 \
       --max_primary_size 1024 \
       --max_secondary_size 2048 \
       --lambda_cost 0.1 \
       --semantic_weight 0.3 \
       --cache_miss_penalty 0.5 \
       --perplexity_weight 0.2 \
       --attention_weight 0.3 \
       --gradient_weight 0.2 \
       --hidden_dim 256 \
       --lr 3e-4 \
       --gamma 0.99 \
       --tau 0.005 \
       --alpha 0.2 \
       --num_episodes 50 \
       --max_steps 50 \
       --turn_length 10 \
       --batch_size 32 \
       --save_interval 10
     ```
## **Contributing**

Contributions are welcome! Here's how you can get involved:

1. **Report Issues**:
   - Found a bug? Have a feature request? Open an issue in the GitHub repository.

2. **Suggest Enhancements**:
   - Propose ideas to improve the algorithm or its implementation.

3. **Submit Pull Requests**:
   - Fork the repository.
   - Make your changes in a new branch.
   - Open a pull request for review.

Please ensure all contributions adhere to the repository's coding standards and include sufficient documentation.


## **License**

This project is licensed under the **MIT License**. Feel free to use, modify, and distribute the code, provided you include the original license. See the `LICENSE` file for details.


## **Acknowledgments**

Special thanks to:
- **Prof. James Caverlee**: For guidance, feedback, and inspiration during the project.
