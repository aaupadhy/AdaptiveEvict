import torch
import json
import argparse
from solver import Solver

def generate_conversation_prompts(solver, num_prompts=1000, min_length=50, max_length=200):
    prompts = []
    base_prompts = [
        "Let's discuss", "I think that", "The main idea is", "In my opinion",
        "The key point is", "What I understand is", "The important thing is",
        "I believe that", "The fundamental concept is", "The core idea is"
    ]
    
    for _ in range(num_prompts):
        base_prompt = base_prompts[torch.randint(0, len(base_prompts), (1,)).item()]
        generated_text = solver.generate_text(
            input_text=base_prompt,
            n_tokens_to_generate=torch.randint(min_length, max_length, (1,)).item(),
            kv_cache=True
        )
        prompts.append({
            "prompt": generated_text,
            "length": len(generated_text.split())
        })
        
    return prompts

def main(args):
    solver = Solver(args)
    prompts = generate_conversation_prompts(
        solver,
        num_prompts=args.num_prompts,
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    # Open in write ("w") mode, not append ("a") mode to avoid redundant accumulation
    with open(args.output_file, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    print(f"Generated {len(prompts)} prompts and saved to {args.output_file}")
    print(f"Average prompt length: {sum(p['length'] for p in prompts) / len(prompts):.2f} words")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Prompt generation config
    parser.add_argument('--num_prompts', type=int, default=1000)
    parser.add_argument('--min_length', type=int, default=50)
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--output_file', type=str, default='data/rl_training_data.json')

    # Token sampling parameters
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--top_p', type=float, default=0.6)

    # Model architecture required by Solver
    parser.add_argument('--network_type', type=str, default='llama')
    parser.add_argument('--train_tokens_len', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_workers', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--forward_mul', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Tokenizer and data handling
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--data_file', type=str, default='train.txt')
    parser.add_argument('--max_merged_tokens', type=int, default=5000)

    # Load options
    parser.add_argument('--model_path', type=str, default='saved_models')
    parser.add_argument('--load_tokenizer', action='store_true')
    parser.add_argument('--load_model', action='store_true')

    args = parser.parse_args()
    main(args)
