# Large Language Models (LLM) from Scratch in PyTorch
### Simplified Scratch Pytorch Implementation of Large Language Models (LLM) with Detailed Steps (Refer to <a href="gpt.py">gpt.py</a> and <a href="llama.py">llama.py</a>)

### Key Points
<ul>
  <li> Contains two models: GPT and LLAMA.</li>
  <li> GPT model serves as the base simple decoder-only transformer.</li>
  <li> LLAMA contains advanced concepts like Rotaional Positional Encoding (RoPe), Mixture of Experts, etc. (Refer below.) </li>
  <li> These models are scaled-down versions of their original architectures. </li>
</ul>  

## Status of functionalities (Added to LLAMA):
:white_check_mark: ByTe-Pair Tokenization <br>
:white_check_mark: Temperature, Top-p and Top-k <br>
:white_check_mark: RMSNorm <br>
:white_check_mark: Rotational Positional Encoding (RoPe) <br>
:white_check_mark: Mixture of Experts <br>
:white_square_button: KV Cache <br>
:white_square_button: Grouped Query Attention <br>
:white_square_button: Infini Attention

Feel free to comment if you want anything integrated here.


## Run command: <br>

```
python main.py --network_type llama
```
 
- The network can be selected between llama and gpt.
- The program will download the poem dataset to a text file (default name: "data.txt"). To use a custom dataset, replace the content or provide a different text file.


## Arguments
