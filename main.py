import os
import torch
import argparse
import datetime
from solver import Solver


def main(args):
    # Create required directories if they don't exist
    os.makedirs(args.model_path,  exist_ok=True)

    solver = Solver(args)
    if not args.test_only:
        solver.train()                                                                  # Training function
    else:
        solver.generate_text(n_tokens_to_generate=args.gen_tokens_len,                  # Custom generating function
                             input_text=args.input_text)  

# Print arguments
def print_args(args):
    for k in dict(sorted(vars(args).items())).items():
        print(k)
    print()


# Update arguments
def update_args(args):
    if args.test_only:
        args.load_model = True
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scratch implementation of LLMs.')

    # Data arguments
    parser.add_argument('--data_path', type=str, default='./data/', help='path where the dataset is to be downloaded.')
    parser.add_argument('--data_file', type=str, default='data.txt', help='name of the final csv to be created.')
    parser.add_argument("--test_only", type=bool, default=False, help='False for train and Test. True for just testing the trained model.')
    parser.add_argument("--input_text", type=str, default='You', help='Input prompt for generating text.')

    # Training Arguments
    parser.add_argument('--epochs', type=int, default=25, help='number of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='number of epochs to warmup learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--n_workers', type=int, default=4, help='number of workers for data loaders')
    parser.add_argument('--lr', type=float, default=1e-3, help='peak learning rate')    
    
    # LLM arguments
    parser.add_argument("--network_type", type=str, default='llama', help='Type of model to use.')
    parser.add_argument("--max_merged_tokens", type=int, default=200, help='Maximum number of tokens to be created.')
    parser.add_argument("--train_tokens_len", type=int, default=64, help='number of tokens to use for training.')
    parser.add_argument("--gen_tokens_len", type=int, default=256, help='number of tokens to generate while testing.')
    parser.add_argument("--temperature", type=float, default=1.0, help='Temperature to sharpen/flatten the output probabilities')
    parser.add_argument("--top_p", type=float, default=0.6, help='Tokens with top-p probabilities to consider while generating text')
    parser.add_argument("--top_k", type=int, default=10, help='Top-k tokens to consider while generating text')

    # LLM Network arguments
    parser.add_argument("--embed_dim", type=int, default=48, help='dimensionality of the latent space')
    parser.add_argument("--n_heads", type=int, default=4, help='number of heads to use in Multi-head attention')
    parser.add_argument("--forward_mul", type=int, default=2, help='forward multiplier')
    parser.add_argument("--n_layers", type=int, default=6, help='number of encoder layers')
    parser.add_argument("--dropout", type=float, default=0.0, help='dropout value')

    parser.add_argument('--model_path', type=str, default='./saved_models', help='path to store trained model')
    parser.add_argument("--load_model", type=bool, default=False, help="load saved model")
    parser.add_argument("--load_tokenizer", type=bool, default=False, help="load saved model")

    start_time = datetime.datetime.now()
    print("Started at " + str(start_time.strftime('%Y-%m-%d %H:%M:%S')))

    args = parser.parse_args()
    args = update_args(args)
    print_args(args)
    
    main(args)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print("Ended at " + str(end_time.strftime('%Y-%m-%d %H:%M:%S')))
    print("Duration: " + str(duration))

