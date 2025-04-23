import torch
import json
import argparse
from solver import Solver
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_conversation_prompts(solver, num_prompts=1000, min_length=50, max_length=200):
    prompts = []
    base_prompts = [
    "Let's discuss",
    "I think that",
    "The main idea is",
    "In my opinion",
    "The key point is",
    "What I understand is",
    "The important thing is",
    "I believe that",
    "The fundamental concept is",
    "The core idea is",
    "From my point of view",
    "It seems to me that",
    "One could argue that",
    "Evidence suggests",
    "To illustrate this point",
    "An intriguing observation is",
    "A critical factor is",
    "It is noteworthy that",
    "Observations show that",
    "Research indicates",
    "A common misconception is",
    "Let's explore",
    "Let's examine",
    "Let's analyze",
    "Focusing on",
    "In this context",
    "Turning to",
    "On the one hand",
    "On the other hand",
    "At its core",
    "At first glance",
    "Ultimately",
    "Interestingly",
    "Surprisingly",
    "Notably",
    "In contrast",
    "Alternatively",
    "For example",
    "For instance",
    "Specifically",
    "Generally speaking",
    "Overall",
    "In conclusion",
    "To summarize",
    "Briefly put",
    "To put it briefly",
    "To elaborate",
    "To clarify",
    "With respect to",
    "Regarding",
    "Concerning",
    "It follows that",
    "Hence",
    "Therefore",
    "Consequently",
    "Moreover",
    "Furthermore",
    "Additionally",
    "Significantly",
    "Crucially",
    "A key takeaway is",
    "The bottom line is",
    "It’s important to note",
    "An important distinction is",
    "One notable aspect is",
    "Let me point out that",
    "Consider that",
    "Bear in mind that",
    "It’s clear that",
    "It’s evident that",
    "One perspective is",
    "A useful analogy is",
    "To put it another way",
    "To frame this differently",
    "This raises the question",
    "This implies that",
    "This suggests",
    "This highlights",
    "This underscores",
    "This demonstrates",
    "This indicates",
    "This proves that",
    "In practical terms",
    "In real-world scenarios",
    "In theory",
    "In practice",
    "In summary",
    "As an example",
    "As a result",
    "Considering that",
    "Given that",
    "While it may seem",
    "Although",
    "Despite",
    "Even though",
    "Regardless",
    "Taking into account",
    "Reflecting on",
    "Looking back",
    "Looking forward",
    "In this discussion, let us consider multiple perspectives",
    "Let us delve into the nuances of this topic",
    "How might we approach this challenge effectively",
    "One aspect worth exploring further in this context",
    "Let us critically evaluate the underlying assumptions here",
    "It is valuable to examine both theoretical and practical implications",
    "How does this phenomenon manifest in real-world scenarios",
    "What lessons can we draw from historical precedents here",
    "Let us break down each component step by step",
    "It’s important to balance rigor with practical usability",
    "Considering both advantages and limitations of this approach",
    "How might alternative frameworks shed new light on this",
    "Let us compare and contrast different methodologies here",
    "To understand the implications, we must dig deeper",
    "Let us frame this problem from a systems perspective",
    "What are the key drivers influencing this outcome",
    "Let us reflect on both quantitative and qualitative data",
    "It’s crucial to highlight the central trade-offs involved",
    "How can we reconcile conflicting objectives in this scenario",
    "This raises important questions about model interpretability"
    ]
    
    logging.info(f"Starting to generate {num_prompts} conversation prompts.")

    for _ in tqdm(range(num_prompts), desc="Generating prompts"):
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
        
    logging.info("Finished generating conversation prompts.")
    return prompts

def main(args):
    logging.info("Initializing Solver with provided arguments.")
    solver = Solver(args)

    logging.info("Starting prompt generation.")
    prompts = generate_conversation_prompts(
        solver,
        num_prompts=args.num_prompts,
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    logging.info(f"Saving generated prompts to {args.output_file}.")
    with open(args.output_file, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    logging.info(f"Generated {len(prompts)} prompts and saved to {args.output_file}.")
    logging.info(f"Average prompt length: {sum(p['length'] for p in prompts) / len(prompts):.2f} words.")

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
