from tokenizers import ByteLevelBPETokenizer


class BytePairTokenizer:
    def __init__(self, data_file, max_merged_tokens=0):
        """
        BytePair Tokenizer.
        Trains a tokenizer on characters from the dataset and merges frequent token pairs.

        Parameters:
            data_file         (str) : Full path to the training data
            max_merged_tokens (int) : Number of BPE merges to allow (determines vocab size)
        """
        # Read raw data
        with open(data_file, encoding='utf-8') as f:
            raw_text = f.read()

        # Train tokenizer
        base_tokens = list(set(raw_text))
        vocab_size = len(base_tokens) + max_merged_tokens
        self.tokenizer = ByteLevelBPETokenizer()
        self.tokenizer.train(files=[data_file], vocab_size=vocab_size, min_frequency=2, show_progress=True)

        # Store vocab and metadata
        vocab = self.tokenizer.get_vocab()
        self.n_tokens = len(vocab)
        self.token_to_idx_map = vocab
        self.idx_to_token_map = {idx: token for token, idx in vocab.items()}

    def data_to_tokens(self, sentence):
        return self.tokenizer.encode(sentence).tokens

    def tokens_to_indices(self, tokens):
        return [self.token_to_idx_map[token] for token in tokens]

    def encode(self, data):
        return self.tokenizer.encode(data).ids

    def decode(self, indices):
        return self.tokenizer.decode(indices)

    def display_info(self):
        print(f"Vocab size: {self.n_tokens}")

    def display_vocab_size(self):
        print(f"Vocabulary size: {self.n_tokens}")
