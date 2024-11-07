import os
import torch
import pickle
from gpt import GPT
import torch.nn as nn
from torch import optim
from llama import LLAMA
import torch.nn.functional as F
from get_data import prepare_data
from dataloader import get_dataloader
from tokenizer import BytePairTokenizer


class Solver(object):
    def __init__(self, args):
        self.args = args

        # Download and preprocess dataset if no data file is provided.
        prepare_data(data_path=args.data_path, data_file=self.args.data_file)

        # Tokenizer to convert text to token indices
        if self.args.load_tokenizer:                                                            # Load saved tokenizer
            with open(os.path.join(self.args.model_path, 'tokenizer.pt'), 'rb') as f:
                self.tokenizer = pickle.load(f)
                self.tokenizer.display_info()
        else:
            self.tokenizer = BytePairTokenizer(data_file=os.path.join(self.args.data_path, self.args.data_file), max_merged_tokens=self.args.max_merged_tokens)
            with open(os.path.join(self.args.model_path, 'tokenizer.pt'), 'wb') as f:
                pickle.dump(self.tokenizer, f)

        # Define data loader
        self.train_loader = get_dataloader(data_file  = os.path.join(self.args.data_path, self.args.data_file), 
                                           batch_size = self.args.batch_size, 
                                           seq_len    = self.args.train_tokens_len, 
                                           n_workers  = self.args.n_workers, 
                                           tokenizer  = self.tokenizer)

        # Define Training model
        if self.args.network_type == 'llama':
            training_model = LLAMA
        elif self.args.network_type == 'gpt':
            training_model = GPT

        self.model = training_model(vocab_size  = self.tokenizer.n_tokens, 
                                    embed_dim   = self.args.embed_dim, 
                                    max_seq_len = self.args.train_tokens_len, 
                                    n_layers    = self.args.n_layers, 
                                    n_heads     = self.args.n_heads, 
                                    forward_mul = self.args.forward_mul).cuda()

        # Option to load the saved model.
        if self.args.load_model:
            self.model.load_state_dict(torch.load(os.path.join(self.args.model_path, f"{self.args.network_type}.pt")))

        # Training loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Training parameters stats
        self.n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters in the model: {self.n_parameters}")
        print(f"Number of tokens per parameters: {self.train_loader.dataset.total_tokens/self.n_parameters:.4f}.")

    def train(self):
        iters_per_epoch = len(self.train_loader)

        # Define optimizer for training the model
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=1e-3)

        # schedulers for linear warmup of lr and then cosine decay to 1e-5. Learning rate is adjusted per step to accomodate large/small batch sizes.
        current_iter  = 0
        warmup_iters  = (iters_per_epoch * self.args.warmup_epochs) - 1
        total_iters   = (iters_per_epoch * self.args.epochs)
        linear_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_iters, last_epoch=-1)
        cos_decay     = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=total_iters-warmup_iters, eta_min=1e-5)

        # Training loop
        for epoch in range(self.args.epochs):
            self.model.train()
            for i, data in enumerate(self.train_loader):

                x, y = data
                x, y = x.cuda(), y.cuda()

                logits = self.model(x)

                logits = logits.flatten(0, 1)
                y = y.flatten()

                loss  = self.loss_fn(logits, y)
                b_acc =  (logits.max(1)[1]==y).float().mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update learning rate using schedulers
                if current_iter < warmup_iters:
                    linear_warmup.step()
                else:
                    cos_decay.step()   
                current_iter += 1             

                # Log training progress
                if i % 50 == 0 or i == (iters_per_epoch - 1):
                    current_lr = optimizer.param_groups[-1]['lr']
                    print(f'Ep: {epoch+1}/{self.args.epochs}\tIt: {i+1}/{iters_per_epoch}\tbatch_loss: {loss:.4f}\tbatch_accuracy: {b_acc:.2%}\tlr:{current_lr:.6f}')

            # Save model and token_indexer
            torch.save(self.model.state_dict(), os.path.join(self.args.model_path, f"{self.args.network_type}.pt"))

            # Generate sample output at the end of the epoch
            self.generate_text(n_tokens_to_generate=self.args.gen_tokens_len, input_text=self.args.input_text)

    def generate_text(self, input_text='The', n_tokens_to_generate=256, kv_cache=True):
        self.model.eval()
        if self.args.network_type == 'llama':
            self.model.reset_cache()

        # Encode input text
        tokens_idx = self.tokenizer.encode(input_text)

        # Convert to tensor
        x = torch.LongTensor(tokens_idx).unsqueeze(0).cuda()

        # Generate till we reach generated token length
        while len(x[0]) <= n_tokens_to_generate:
            with torch.no_grad():
                logits = self.model(x[:, -self.args.train_tokens_len:], kv_cache=kv_cache)  # Use last {train_tokens_len} tokens only (as network is trained with this length) i.e. Context length: train_tokens_len

            next_token_logits = logits[0, -1]                                               # Use the output of the last token
            
            # Temperature logic to sharpen/flatten the output probabilities
            probs = F.softmax(next_token_logits/(self.args.temperature + 1e-8), -1)

            # Top-p logic
            p_sorted, p_idx = torch.sort(probs, descending=True)
            p_cumsum = torch.cumsum(p_sorted, dim=0)
            p_not_selected = p_cumsum > self.args.top_p                                     # Select top token which sum upto top-p probability
            p_not_selected[torch.nonzero(p_not_selected)[0]] = False                        # Unselect first element from the not selected list to ensure we always output tokens
            p_sorted[p_not_selected] = 0                                                    # Set probability of not selected tokens to 0
            probs = p_sorted.gather(0, p_idx.argsort(0))                                    # Selected tokens

            # Top-k logic                                                                   # Can be combined with top-k for optimization
            p_sorted, p_idx = torch.sort(probs, descending=True)
            p_sorted[self.args.top_k:] = 0                                                  # Selected Top-k tokens with highest probabilty
            probs = p_sorted.gather(0, p_idx.argsort(0))                                    # Selected tokens

            # Sample based on probability using multinomial sampling
            probs = probs/probs.sum()                                                       # Convert to probability by normalization
            next_token_idx = torch.multinomial(probs, 1)                                    # Sample

            # Append to input and generate next token
            x = torch.cat((x, next_token_idx.unsqueeze(0)), dim=1)

        # Display final output
        generated_text = self.tokenizer.decode(x[0].tolist())            
        print(f'\n\nGenerated text for input text "{input_text}" is:\n{generated_text}\n\n')

