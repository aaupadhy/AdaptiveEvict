AdaptiveEvict: An RL Framework for Token Window
Management in Large Language Models
Team Members:
Aayush Upadhyay (436000751)
Ajay Jagannath (236002822)
Sarvesh Gurumurthy Sainathan (735007588)
1. Research question
Research Question: Can we employ a reinforcement learning (RL) policy to dynamically
evict (and possibly retrieve) tokens from a Large Language Model’s (LLM) context window in order
to balance task accuracy and efficiency under strict token-memory constraints?
Motivation: LLMs like LLaMA, GPT-2, or Mistral have context-length limits (e.g., 2K–8K to-
kens). Na¨ıve eviction (e.g., FIFO) often discards crucial earlier tokens, degrading performance
on tasks requiring long-range context. Existing approaches rely on heuristics (sliding windows,
attention-based metrics) or supervised gating but do not explicitly optimize a reward function that
captures the trade-off between final performance and context size. We want to fill that gap by
formulating token management as a Markov Decision Process (MDP) and training an RL agent to
decide which tokens to keep and when to evict, possibly with the option to retrieve evicted tokens
later.
2. Why is this interesting?
• Efficiency vs. Accuracy: LLM inference cost grows (often quadratically) with the number
of tokens. If we can drastically reduce context size (e.g., by 50%) without sacrificing accuracy,
we achieve faster inference and lower memory usage.
• Adaptive Long-Context Reasoning: Heuristics cannot anticipate future importance. RL,
trained with an end-task reward, can learn from experience which tokens might be needed
later, achieving better context retention for tasks like QA, summarization, or multi-turn
dialogue.
• Practical Deployment: Long conversations and documents exceed typical LLM context
windows. An intelligent eviction policy extends practical usability of these models on resource-
constrained devices or real-time chat systems.
1
3. Existing research
Existing Research on Token Management in LLMs
• Static & Sliding Window Methods: StreamingLLM [1] (Xiao et al., 2024) uses a fixed
sliding window to retain some initial prompt tokens plus the most recent tokens, discarding
the middle. This is simple but not context-adaptive: critical earlier tokens might be dropped
if they fall outside the retained window.
• Attention Score Heuristics (H2O): Proposed by Zhang et al. (2023), H2O[2] tracks
cumulative attention to identify “low-attention” tokens to evict. Although it gives fair results,
it is biased (it can over-prioritize first or last tokens) and does not look ahead to future utility.
• Learned Token Pruning: Dynamic Context Pruning[3] (Anagnostidis et al., NeurIPS 2024)
shows a trainable pruning mechanism that can remove up to 80% of tokens during generation
with minimal performance loss. However, it is trained via supervised or language-modeling
losses, not an explicit reward that balances performance and context cost.
• KV-Cache Compression/Eviction: FastGen[4] (Ge et al., 2023) and NACL[5] (Chen et
al., 2024) compress or selectively evict key-value pairs. These are typically heuristic-driven,
not RL-based.
• Retrieval & Two-Tier Memory: ArkVale[6] (Chen et al., NeurIPS 2024) partitions con-
text into pages, evicting them to slower storage and later retrieving if needed, based on a
fixed scoring. An RL approach could dynamically learn better eviction/retrieval strategies
for final reward.
• Attention-Gating:[7] (Zeng et al., 2024) trains small binary gates (“keep or drop”) in a
supervised manner, boosting some tasks while pruning >50% of tokens. Again, no explicit
RL reward is used.
• RL in Context Selection (NLP):[8] Kang et al. (2020) used an RL-based selector for
document-level translation to choose relevant sentences. This approach is encouraging but
has not been extended to token-level eviction within the LLM context.
Unanswered Gap: No work to date explicitly applies RL to token window eviction inside
LLMs, with a custom reward balancing memory usage and final accuracy. Our project directly
addresses this gap.
4. Sketch of Method
4.1 MDP Formulation
State Representation: At each generation step (or every N tokens), we extract features for each
token:
• Last-layer hidden embedding (captures semantic content)
2
• Cumulative attention score over recent decoding steps
• Position/recency
• Global info: e.g. partial outputs, step count
We feed these features (possibly through a small encoder) into the RL agent, yielding a state vector.
Action Space:
• Evict certain tokens from the KV-cache (or move them to a secondary memory)
• Retrieve previously evicted tokens (if using a tiered memory like ArkVale)
• No-op (retain current context)
Reward Function:
R = TaskPerformance − λ × (ContextSize)
where TaskPerformance might be QA accuracy, final answer correctness, or negative perplexity;
ContextSize is the count of tokens retained; and λ is a scalar balancing accuracy vs. cost.
4.2 Implementation Setup
LLM Environment: We adapt an open-source decoder-only model like LLaMA-2 (7B or 13B) or
Mistral 7B. When the KV-cache grows beyond a threshold, the RL agent decides which tokens to
drop. We will modify the attention mask so dropped tokens are excluded from future attention.
Datasets:
• Synthetic multi-turn dialogues that embed crucial info early to be used later.
• Real long-context tasks: e.g. HotpotQA or LongBench (for open-domain QA requiring
multiple documents).
RL Algorithm: An actor-critic approach (e.g. PPO) will be used, as it handles high-dimensional
states and sequential decisions well. We treat each conversation or QA scenario as an episode. After
generating the final answer, we compute the reward and update the policy.
5. Evaluation
• Perplexity: Compare perplexity on held-out corpora with RL-driven eviction vs. baselines
(FIFO, H2O).
• Task Accuracy: On QA or summarization tasks, measure correctness (F1, EM, ROUGE).
Track whether crucial tokens remain in context.
• Context Efficiency: Report average # tokens retained, memory usage, and speedups. E.g.,
if the RL agent cuts 50% of tokens but maintains near-baseline accuracy, it’s a win.
3
• Baselines:
– FIFO/Sliding Window
– Attention Heuristics (H2O)
– Learned Gating (Supervised)
– No Eviction (upper bound on performance)
• Ablations:
– Turn off retrieval (only eviction)
– Vary λ in the reward
– Compare different state representations
6. Timeline & Milestones
Weeks 1–2: Setup and Baselines
• Implement baseline token eviction (FIFO, sliding window, attention-based).
• Integrate a smaller LLM (GPT-2 or LLaMA-2 7B) in a custom Python environment that can
manipulate the KV-cache.
Weeks 3–4: RL Policy Implementation
• Introduce PPO-based actor-critic for eviction decisions.
• Train on synthetic dialogues with shorter contexts to confirm the agent learns basic strategies.
Weeks 5–6: Real-Task Experiments
• Use HotpotQA, LongBench, or similar.
• Compare RL eviction vs. baseline heuristics on QA accuracy, perplexity, token usage.
• Begin ablation studies.
Weeks 7–8: Final Tuning & Analysis
• Refine reward weighting and retrieval policy if using tiered memory.
• Gather final metrics, do additional ablations.
• Prepare final presentation/report.
4
References
[1] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming
language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023.
[2] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song,
Yuandong Tian, Christopher R´e, Clark Barrett, et al. H2o: Heavy-hitter oracle for efficient
generative inference of large language models. Advances in Neural Information Processing
Systems, 36:34661–34710, 2023.
[3] Sotiris Anagnostidis, Dario Pavllo, Luca Biggio, Lorenzo Noci, Aurelien Lucchi, and Thomas
Hofmann. Dynamic context pruning for efficient and interpretable autoregressive transformers.
Advances in Neural Information Processing Systems, 36:65202–65223, 2023.
[4] Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. Model tells
you what to discard: Adaptive kv cache compression for llms. arXiv preprint arXiv:2310.01801,
2023.
[5] Yilong Chen, Guoxia Wang, Junyuan Shang, Shiyao Cui, Zhenyu Zhang, Tingwen Liu, Shuo-
huan Wang, Yu Sun, Dianhai Yu, and Hua Wu. Nacl: A general and effective kv cache eviction
framework for llms at inference time. arXiv preprint arXiv:2408.03675, 2024.
[6] Renze Chen, Zhuofeng Wang, Beiquan Cao, Tong Wu, Size Zheng, Xiuhong Li, Xuechao Wei,
Shengen Yan, Meng Li, and Yun Liang. Arkvale: Efficient generative llm inference with re-
callable key-value eviction. Advances in Neural Information Processing Systems, 37:113134–
113155, 2025.
[7] Shuang Zeng, Runxin Xu, Baobao Chang, and Lei Li. Double graph based reasoning for
document-level relation extraction. arXiv preprint arXiv:2009.13752, 2020.
[8] Xiaomian Kang, Yang Zhao, Jiajun Zhang, and Chengqing Zong. Dynamic context selec-
tion for document-level neural machine translation via reinforcement learning. arXiv preprint
arXiv:2010.04314, 2020.