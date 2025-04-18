import logging
from tokenizers import ByteLevelBPETokenizer

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

        data = open(data_file, encoding='utf-8').read()
        base_tokens = list(set(data))
        vocab_size = len(base_tokens) + max_merged_tokens
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(files=[data_file], vocab_size=vocab_size, min_frequency=2, show_progress=True)
        encoding = tokenizer.encode(data)
        self.tokenizer = tokenizer
        self.tokenized_data = encoding.tokens
        self.base_tokens = base_tokens
        vocab = tokenizer.get_vocab()
        self.n_tokens = len(vocab)
        self.merged_tokens = self.n_tokens - len(self.base_tokens)
        self.token_to_idx_map = vocab
        self.idx_to_token_map = {idx:token for token, idx in vocab.items()}

    def display_info(self):
        logger = logging.getLogger(__name__)
        logger.info("Number of base tokens: %d", len(self.base_tokens))
        logger.info("Number of merged tokens: %d", self.merged_tokens)

    def data_to_tokens(self, sentence):
        return self.tokenizer.encode(sentence).tokens

    def tokens_to_indices(self, tokens):
        return [self.token_to_idx_map[token] for token in tokens]

    def encode(self, data):
        return self.tokenizer.encode(data).ids

    def decode(self, indices):
        return self.tokenizer.decode(indices)

