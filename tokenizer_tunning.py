import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizer, PreTrainedModel
from datasets import Dataset
from tqdm import tqdm
from copy import deepcopy
from typing import Iterator, List, Dict, Tuple


class TokenizerTuner:
    """
    Class for finetuning existing tokenizer on a new dataset
    and adapting the model's embedding weights to a new vocab.
    
    1. Train a temporary BPE tokenizer on the target dataset to identify new domain-specific tokens.
    2. Extend the original tokenizer's vocabulary by adding the newly discovered tokens to a copy of the base tokenizer.
    3. Resize the model's embedding matrix to accommodate the expanded vocabulary.
    4. Optimize weight initialization for new tokens: 
    Instead of random initialization, compute the initial embeddings for new tokens as the weighted average of their constituent sub-tokens from the original vocabulary. 
    This approach accelerates domain adaptation and improves training stability compared to random initialization.
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, dataset: Dataset, vocab_size: int = 30_000):
        """
        Tunner initialization.

        Args:
            model (PreTrainedModel): Transformer model (Hugging Face).
            tokenizer (PreTrainedTokenizer): Tokenizer from model.
            dataset (Dataset): Dataset with 'text' columns. For manupulation with dataset function _batch_iterator may be redefined.
            vocab_size (int): Added tokens count.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.vocab_size = vocab_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"{self.device =}")
        self.model.to(self.device)

    def _batch_iterator(self) -> Iterator[str]:
        """Generator for tokenizer trainer."""
        for example in tqdm(self.dataset, desc="Load dataset"):
            yield example["text"]

    def _smart_initialize(self, token: str, embedding_layer: torch.nn.Embedding, original_vocab: Dict[str, int]) -> torch.Tensor:
        """
        Init new token embedding.

        Splits a new token into subtokens using the original tokenizer.
        If the subtokens are found in the dictionary, the average of their embeddings is returned.
        Otherwise the new token is a random vector created from mean and standard deviation as the whole embedding matrix.

        Args:
            token (str): New token.
            embedding_layer (torch.nn.Embedding): Original embedding layer.
            original_vocab (Dict[str, int]): Original tokenizer vocab.

        Returns:
            torch.Tensor: New token embedding.
        """
        sub_tokens = self.tokenizer.tokenize(token)
        sub_token_ids = [original_vocab.get(st) for st in sub_tokens if st in original_vocab]

        if sub_token_ids:
            sub_token_embeddings = embedding_layer.weight.data[sub_token_ids]
            mean_embedding = torch.mean(sub_token_embeddings, dim=0)
            return mean_embedding
        else:
            mean = embedding_layer.weight.data.mean()
            std = embedding_layer.weight.data.std()
            return torch.normal(mean, std, size=(embedding_layer.weight.size(1),), device=self.device)

    def _train_new_tokenizer(self) -> Tuple[PreTrainedTokenizer, List[str]]:
        """
        Training new tokenizator
        """
        temp_tokenizer = Tokenizer(models.BPE())
        temp_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(vocab_size=self.vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

        temp_tokenizer.train_from_iterator(self._batch_iterator(), trainer=trainer)

        new_vocab = temp_tokenizer.get_vocab()
        original_vocab = self.tokenizer.get_vocab()
        unique_new_tokens = [token for token in new_vocab.keys() if token not in original_vocab]

        new_tokenizer = deepcopy(self.tokenizer)
        added_count = new_tokenizer.add_tokens(unique_new_tokens)
        print(f"Added {added_count} new tokens in the original tokenizator.")

        return new_tokenizer, unique_new_tokens

    def _reinit_model_embeddings(self, new_tokenizer: PreTrainedTokenizer):
        """
        Change embeddings size in the original model and reinitialize embeddings.
        """
        original_embeddings = self.model.get_input_embeddings()
        original_vocab_size = original_embeddings.weight.size(0)
        
        self.model.resize_token_embeddings(len(new_tokenizer))
        
        new_embeddings = self.model.get_input_embeddings()
        new_vocab = new_tokenizer.get_vocab()
        original_vocab = self.tokenizer.get_vocab()

        added_tokens = {token: token_id for token, token_id in new_vocab.items() if token_id >= original_vocab_size}
        
        print(f"New tokens count {len(added_tokens)}")
        for token, token_id in tqdm(added_tokens.items(), desc="Init weight"):
            new_embedding_vector = self._smart_initialize(token, original_embeddings, original_vocab)
            new_embeddings.weight.data[token_id] = new_embedding_vector
        
        print("Init embeddings is done.")

    def tune(self) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
        """
        Start training process.

        Returns:
            Tuple[PreTrainedTokenizer, PreTrainedModel]: New tokenizator and model with smart reinit embeddings.
        """
        print(f"---{"Start training":^40}---")
        new_tokenizer, _ = self._train_new_tokenizer()
        self._reinit_model_embeddings(new_tokenizer)
        self.model.config.vocab_size = len(new_tokenizer)
        print(f"---{"Training is end":^40}---")
        return new_tokenizer, self.model
