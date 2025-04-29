# AdaptiveEvict

## Introduction

AdaptiveEvict is a reinforcement learning (RL) framework designed to manage the key-value (KV) cache in transformer-based language models. By learning when to evict and retrieve tokens, our approach dynamically balances inference efficiency with model accuracy under strict memory constraints. This project demonstrates how an RL agent can outperform static heuristics (e.g., FIFO or attention-based methods) by explicitly optimizing a reward that captures both context retention and task performance.

## Problem Statement (detailed)

Modern large language models (LLMs) such as GPT, LLaMA, and others rely on a KV cache to store past activations for efficient autoregressive generation. However, the cache has a fixed capacity and can quickly overflow as context length grows:

- **Fixed-size limits** force eviction of tokens when capacity is reached.  
- **Static eviction rules** (e.g., FIFO or simple attention heuristics) do not account for future token importance, leading to premature removal of critical context.  
- **Performance degradation** occurs in tasks requiring long-range dependencies (e.g., multi-turn dialogue, open-domain question answering) when key tokens are discarded too early.  

Existing LLM cache management strategies are often proprietary or heuristic-driven, offering no explicit trade-off between inference cost and final task accuracy. Our work addresses this gap by formulating cache management as a Markov Decision Process (MDP), training an RL agent to make eviction/retrieval decisions that maximize a composite reward:

> **Reward = log P(next token) – λ × (tokens retrieved)**

where the first term encourages predictive accuracy and the second penalizes costly retrievals from secondary storage.

## Our Methods (detailed)

1. **MDP Formulation**  
   - **State Representation:** At each decision point, we extract for each token: last-layer hidden embeddings, cumulative attention scores, recency/position, and global context features, combining into a fixed-length state vector.  
   - **Action Space:** A continuous action vector specifying (a) which tokens to evict from the primary cache, (b) which tokens to retrieve from secondary storage, and (c) no-op decisions.  

2. **Environment Setup**  
   - **Model Backbone:** LLaMA (locally hosted, MoE variant) with KV cache enabled.  
   - **Corpus:** Cosmopedia-100k dataset (~25B tokens) for pre-training and synthetic dialogues.  
   - **Cache Tiers:** Two-tier KV cache (primary fast memory + secondary slower storage) integrated into a custom `KVCacheEnv`.  

3. **RL Agent**  
   - **Algorithm:** Soft Actor-Critic (SAC) with actor, dual critics, and target networks, chosen for sample efficiency and stability in continuous spaces.  
   - **Reward Signal:** Combines log-probability of the correct next token with a penalty for retrieval operations.  
   - **Training Loop:** At each generation step, the agent observes the cache state, issues eviction/retrieval actions, the model generates the next token, and the environment computes the reward.

4. **Evaluation Metrics**  
   - **Perplexity:** Compare RL-driven eviction vs. baselines.  
   - **Task Accuracy:** F1/EM on QA and ROUGE scores for summarization.  
   - **Context Efficiency:** Average # tokens retained, cache miss rates, and inference speedups.  
   - **Ablations:** Turn off retrieval, vary λ, compare different state feature sets.

## Installation (including `requirements.txt`)

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-org>/AdaptiveEvict.git
   cd AdaptiveEvict
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **`requirements.txt`**

   ```text
   torch>=1.12.0
   transformers>=4.0.0
   gym>=0.21.0
   numpy>=1.21.0
   stable-baselines3>=1.4.0
   ```

4. **Configure and run**  
   - Update `config.py` with your environment settings (model path, cache thresholds, λ parameter).  
   - Launch training:
     ```bash
     python run_rl_agent.py --config config.yaml
     ```
   - Evaluate with:
     ```bash
     python evaluate.py --checkpoint path/to/model.pt
     ```
