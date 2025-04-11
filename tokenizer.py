import pandas as pd
from tqdm import tqdm
from collections import Counter, defaultdict
import time


class BytePairTokenizer:
    def __init__(self, data_file, max_merged_tokens=0):
        """
        BytePair Tokenizer.
        Creates tokens using unique characters present in the data file.
        Frequent pair of tokens are merged to create merged tokens.

        Parameters:
            data_file         (str) : full path to the data file
            max_merged tokens (int) : Number of maximum merged tokens to create (hyper-parameter)
        """   

        with open(data_file, encoding='utf-8') as f:
            data = f.read()

        self.tokenized_data, self.base_tokens, self.merged_tokens = self.define_tokens(data, max_merged_tokens)
        self.n_tokens = len(self.base_tokens) + len(self.merged_tokens)
        
        # Token to Idx and Idx to Tokens dicts for fast lookups
        self.token_to_idx_map = {token:i for i, token in enumerate(self.base_tokens + self.merged_tokens)}
        self.idx_to_token_map = {i:token for i, token in enumerate(self.base_tokens + self.merged_tokens)}

    def display_info(self):
        print(f"Number of base tokens: {len(self.base_tokens)}")        
        print(f"Number of merged tokens: {len(self.merged_tokens)}")

    def define_tokens(self, data, max_merged_tokens):
        """
        Function to generate base and merged tokens.
        Data is first converted to base tokens.
        Then most common appearing token pairs are merged to create new tokens.
        Added a few rules for merging. Lot more can be added for optimization.

        Input:
            data           (string) : data in string format
            max_merged_tokens (int) : maximum number of merged tokens to be created
        
        Returns:
            tuple of lists: base tokens (as list) and merged tokens (as list)
        """    

        data_tokens = list(data)
        base_tokens = list(set(data_tokens))
        base_tokens_len = len(base_tokens)
        print(f"{base_tokens_len} base tokens identified: {base_tokens}")        
        
        merged_tokens = []
        original_data_tokens = data_tokens.copy()
        
        if max_merged_tokens > 0:
            print("Merging most common base token pairs to create new tokens.")
            
            for _ in tqdm(range(max_merged_tokens)):
                # Build a more efficient representation for pair counting
                pairs = defaultdict(int)
                for i in range(len(data_tokens) - 1):
                    pair = (data_tokens[i], data_tokens[i+1])
                    pairs[pair] += 1
                
                if not pairs:
                    print("No more pairs to merge")
                    break
                
                # Find the most common pair that meets our criteria
                best_pair = None
                best_count = 0
                
                for pair, count in pairs.items():
                    # Skip if frequency is too low
                    if count <= 1:
                        continue
                    
                    # Skip mixed alphanumeric/non-alphanumeric pairs
                    if pair[0].isalnum() != pair[1].isalnum():
                        continue
                    
                    if count > best_count:
                        best_count = count
                        best_pair = pair
                
                # If no valid pair was found, stop merging
                if best_pair is None:
                    print(f"No valid token pairs found after {len(merged_tokens)} merges. Stopping.")
                    break
                
                # Create the new merged token
                new_token = best_pair[0] + best_pair[1]
                merged_tokens.append(new_token)
                
                # Replace all occurrences of the pair with the new token using a more efficient approach
                new_data_tokens = []
                i = 0
                while i < len(data_tokens):
                    if i < len(data_tokens) - 1 and data_tokens[i] == best_pair[0] and data_tokens[i+1] == best_pair[1]:
                        new_data_tokens.append(new_token)
                        i += 2
                    else:
                        new_data_tokens.append(data_tokens[i])
                        i += 1
                
                data_tokens = new_data_tokens
                
                # Break early if we're not making progress
                if len(data_tokens) >= len(original_data_tokens) - 10:
                    if len(merged_tokens) > 10:  # Allow a few initial merges
                        print(f"Minimal compression achieved. Stopping after {len(merged_tokens)} merges.")
                        break
            
            print(f"{len(merged_tokens)} merged tokens created: {merged_tokens}")
            
        return data_tokens, base_tokens, merged_tokens

    '''
    Functions to encode and decode data:
    Encode:   data  -->  tokens  -->  token indices
    Decode:   data  <--  tokens  <--  token indices
    '''

    def data_to_tokens(self, sentence, tqdm=False):
        """
        Function to convert data to tokens.
        Data is first converted to base tokens.
        Then tokens are merged in the same order they were created.

        Input:
            sentence: string 
        
        Returns:
            list of tokens
        """    

        tokens = list(sentence)
        
        # Process in batches for efficiency
        for merged_token in self.merged_tokens:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] + tokens[i+1] == merged_token:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return tokens

    # Function to convert tokens to token indices
    def tokens_to_indices(self, tokens):                 
        indices = [self.token_to_idx_map[token] for token in tokens]
        return indices

    # Function to convert data to token indices
    def encode(self, data, tqdm=False):
        return self.tokens_to_indices(self.data_to_tokens(data, tqdm=tqdm))

    # Function to convert indices to tokens
    def indices_to_tokens(self, indices):
        tokens = [self.idx_to_token_map[i] for i in indices]
        return tokens

    # Function to convert tokens to data
    def tokens_to_data(self, tokens):
        return ''.join(tokens)

    # Function to convert indices to data
    def decode(self, indices):
        return self.tokens_to_data(self.indices_to_tokens(indices))

