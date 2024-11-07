import pandas as pd
from tqdm import tqdm
from collections import Counter


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

        # Generate characters as base tokens
        data_tokens = list(data)
        base_tokens = list(set(data_tokens))
        base_tokens_len = len(base_tokens)
        print(f"{base_tokens_len} base tokens identified: {base_tokens}")        
        
        # Merge most common appearing token pairs to create new tokens
        merged_tokens  = []
        if max_merged_tokens > 0:
            print("Merging most common base token pairs to create new tokens.")

            for _ in tqdm(range(max_merged_tokens)):

                # Create pairs
                data1 = data_tokens[:-1]
                data2 = data_tokens[1:]
                token_pairs = [(t1, t2) for t1, t2 in zip(data1, data2)]

                # Get the most common appearing pair for merging
                token_pairs_frequency = Counter(token_pairs)

                merged_token = None
                while len(token_pairs_frequency) > 0:                                                       # Must have token pairs that have not been tested yet.
                    tokens_to_merge, max_frequency = token_pairs_frequency.most_common(1)[0]     

                    # Added a rule to pair token only if the pair appears more than once.
                    if max_frequency == 1:                                                                  
                        break       

                    # Added a rule to merge alphanumber with alphanumber and non-alphanumber with non-alphanumber only.
                    if tokens_to_merge[0].isalnum() != tokens_to_merge[1].isalnum():                    
                        del token_pairs_frequency[tokens_to_merge]
                        continue

                    # Merge tokens
                    merged_token = tokens_to_merge[0] + tokens_to_merge[1]
                    merged_tokens.append(merged_token)
                    break

                if merged_token is None:
                    break                                                                                   # Stop if no token pair is found.
                else:
                    # Replace token pairs with merged token in the data
                    i = 0
                    while i < len(data_tokens) - 1:
                        if data_tokens[i] == tokens_to_merge[0] and data_tokens[i+1] == tokens_to_merge[1]:
                            data_tokens[i] = merged_token
                            del data_tokens[i+1]
                        i = i + 1

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

        # Tokens are merged in the same order they were created.
        for merged_token in self.merged_tokens:
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i] + tokens[i+1]) == merged_token:
                    tokens[i] = merged_token
                    del tokens[i+1]
                i = i + 1

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

