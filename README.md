# Large Language Models (LLM) from Scratch in PyTorch
### Simplified Scratch Pytorch Implementation of Large Language Models (LLM) with Detailed Steps (Refer to <a href="gpt.py">gpt.py</a> and <a href="llama.py">llama.py</a>)

Key Points:
<ul>
  <li>Contains two models: GPT and LLAMA.</li>
  <li> GPT model here is a base simple decoder-only Transformer.</li>
  <li> LLAMA contains advanced concepts like Rotaional Positional Encoding (RoPe), Mixture of Experts, etc. (Refer below.) </li>
  <li> These models are scaled-down versions of their Original architecture. </li>
</ul>  

Status of functionalities added to LLAMA:
:white_check_mark: ByTe-Pair Tokenization
:white_check_mark: Temperature, Top-p and Top-k 
white_check_mark Mixture of Experts
- [] KV Cache
- [ ] Grouped Query Attention
- [ ] Infini Attention

