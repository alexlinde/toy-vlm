"""
Text processing module for the Toy VLM.
Handles tokenization and text-related functionality.
"""

import re
import json
import os
from typing import List, Set
import torch

# Token constants
MAX_SEQ_LEN = 40  # Increased for reasoning scratchpad

class SimpleTokenizer:
    """Simple word-based tokenizer for the toy VLM."""
    
    def __init__(self, vocab_file: str = None):
        if vocab_file and os.path.exists(vocab_file):
            # Load pretrained vocabulary
            self.load_vocab(vocab_file)
        else:
            # Initialize with base vocabulary - will be extended during training
            # Added special tokens for chain-of-thought reasoning
            self.vocab = {
                '<PAD>': 0,
                '<START>': 1,
                '<END>': 2,
                '<UNK>': 3,
                '<Q>': 4,        # Question marker
                '<REASON>': 5,   # Start of reasoning section
                '<SEP>': 6,      # Separator between reasoning and answer
                '<FINAL>': 7,    # Final answer marker
            }
            self._update_mappings()
    
    def _update_mappings(self):
        """Update reverse mapping and special token IDs."""
        # Create reverse mapping
        self.idx_to_word = {idx: word for word, idx in self.vocab.items()}

        # Special token IDs
        self.pad_token_id = self.vocab['<PAD>']
        self.bos_token_id = self.vocab['<START>']
        self.eos_token_id = self.vocab['<END>']
        self.unk_token_id = self.vocab['<UNK>']

        # Reasoning special tokens
        self.q_token_id = self.vocab.get('<Q>', self.unk_token_id)
        self.reason_token_id = self.vocab.get('<REASON>', self.unk_token_id)
        self.sep_token_id = self.vocab.get('<SEP>', self.unk_token_id)
        self.final_token_id = self.vocab.get('<FINAL>', self.unk_token_id)
    
    def build_vocab_from_rationales(self, rationale_generator, num_samples=200):
        """Build vocabulary from RationaleGenerator for CoT reasoning."""
        from shapes import ShapeGenerator

        shape_gen = ShapeGenerator()
        word_set = set()

        # Generate samples for each difficulty level
        difficulties = ['easy', 'medium', 'hard']
        for difficulty in difficulties:
            for _ in range(num_samples // len(difficulties)):
                try:
                    # Generate a multi-shape image
                    image, metadata = shape_gen.generate_multi_shape_image()

                    # Generate QA with rationale
                    question, answer, rationale = rationale_generator.generate_qa_with_rationale(
                        metadata, difficulty=difficulty
                    )

                    if question and answer and rationale:
                        # Extract words from all three
                        q_words = self._extract_words(question)
                        a_words = self._extract_words(answer)
                        r_words = self._extract_words(rationale)

                        word_set.update(q_words)
                        word_set.update(a_words)
                        word_set.update(r_words)
                except Exception as e:
                    print(f"Warning: Could not generate QA with rationale: {e}")
                    continue

        # Add common words that might be missing
        common_words = {
            'shape', 'shapes', 'round', 'flat', 'big', 'small', 'object', 'figure',
            'red', 'blue', 'green', 'white', 'black', 'color', 'size',
            'at', 'and', 'or', 'the', 'a', 'is', 'are', 'there', 'they',
            'count', 'found', 'compare', 'vs', 'greater', 'less', 'equal',
            'look', 'identify', 'zero', 'one', 'two', 'three', 'four',
            '0', '1', '2', '3', '4', 'yes', 'no'
        }
        word_set.update(common_words)

        # Build vocabulary
        next_idx = len(self.vocab)
        for word in sorted(word_set):
            if word not in self.vocab:
                self.vocab[word] = next_idx
                next_idx += 1

        self._update_mappings()
        print(f"Built vocabulary with {len(self.vocab)} tokens from RationaleGenerator")

    def _extract_words(self, text: str) -> Set[str]:
        """Extract normalized words from text."""
        if not text:
            return set()
        
        normalized = self._preprocess_text(text)
        return set(normalized.split()) if normalized else set()
    
    def save_vocab(self, vocab_file: str):
        """Save vocabulary to file."""
        with open(vocab_file, 'w') as f:
            json.dump(self.vocab, f, indent=2, sort_keys=True)
        print(f"Saved vocabulary to {vocab_file}")
    
    def load_vocab(self, vocab_file: str):
        """Load vocabulary from file."""
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
        self._update_mappings()
        print(f"Loaded vocabulary with {len(self.vocab)} tokens from {vocab_file}")
    
    @classmethod
    def load_pretrained(cls, vocab_file: str):
        """Create tokenizer instance from saved vocabulary."""
        tokenizer = cls()
        tokenizer.load_vocab(vocab_file)
        return tokenizer
    
    def _preprocess_text(self, text: str) -> str:
        """Remove non-alphanumeric characters and normalize."""
        # Convert to lowercase and keep only alphanumeric characters and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        # Normalize whitespace
        text = ' '.join(text.split())
        return text
        
    def tokenize(self, text: str, allow_unk: bool = False) -> List[int]:
        """Convert text to token IDs.

        Args:
            text: Text to tokenize
            allow_unk: If False, raise error on unknown words (for training).
                      If True, use <UNK> token (for inference).
        """
        text = self._preprocess_text(text)
        words = text.split()

        tokens = []
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                if not allow_unk:
                    raise ValueError(f"Unknown word '{word}' in text '{text}'. "
                                   f"Vocabulary may not be properly built. "
                                   f"Available words: {sorted(self.vocab.keys())[:20]}...")
                tokens.append(self.unk_token_id)

        return tokens
    
    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to text."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        words = []
        for token_id in tokens:
            if token_id in self.idx_to_word:
                word = self.idx_to_word[token_id]
                if skip_special_tokens and word.startswith('<') and word.endswith('>'):
                    continue
                words.append(word)
        
        return ' '.join(words)
    
    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)

class TextProcessor:
    """Handles text processing tasks for the VLM."""
    
    def __init__(self):
        self.tokenizer = SimpleTokenizer()
    
    def prepare_input_sequence(
        self,
        question: str,
        answer: str = None,
        rationale: str = None,
        *,
        mask_controls: bool = True  # mask <Q> and <REASON> if they are injected at inference
    ) -> tuple:
        """
        Returns:
            input_ids:  List[int]  = [<START>, q..., <Q>, <REASON>, r..., <SEP>, <FINAL>, a...]
            target_ids: List[int]  = input_ids[1:] + [<END>]
            loss_mask:  List[int]  = 0/1 per target position (same length as target_ids)
        """
        tok = self.tokenizer

        # --- Build input sequence ---
        q_tokens = tok.tokenize(question)

        if answer is not None:
            a_tokens = tok.tokenize(answer)
            if rationale is not None:
                r_tokens = tok.tokenize(rationale)

                # We inject <Q> and <REASON> into the prompt
                input_ids = (
                    [tok.bos_token_id] +
                    q_tokens +
                    [tok.q_token_id, tok.reason_token_id] +
                    r_tokens +
                    [tok.sep_token_id, tok.final_token_id] +
                    a_tokens
                )

                # Shifted targets
                target_ids = input_ids[1:] + [tok.eos_token_id]

                # ----- Loss mask -----
                # Positions (in targets) to ignore:
                #   - predicting each question token  => len(q_tokens)
                #   - optionally predicting <Q> and <REASON> if you inject them
                ignore = len(q_tokens)
                if mask_controls:
                    ignore += 2  # <Q>, <REASON>

                loss_mask = [0] * ignore + [1] * (len(target_ids) - ignore)

            else:
                # Legacy: no rationale section
                input_ids = [tok.bos_token_id] + q_tokens + a_tokens
                target_ids = input_ids[1:] + [tok.eos_token_id]

                # Mask only the question part in legacy mode
                ignore = len(q_tokens)
                loss_mask = [0] * ignore + [1] * (len(target_ids) - ignore)
        else:
            # Inference-only: no targets/mask
            input_ids = [tok.bos_token_id] + q_tokens
            target_ids = None
            loss_mask = None

        return input_ids, target_ids, loss_mask

    def pad_sequence(self, tokens, max_length=MAX_SEQ_LEN):
        pad_id = self.tokenizer.pad_token_id
        if len(tokens) > max_length:
            return tokens[:max_length]
        return tokens + [pad_id] * (max_length - len(tokens))
            
    def clean_response(self, response: str) -> str:
        """Clean up generated response text."""
        # Remove extra spaces and normalize
        response = ' '.join(response.split())
        
        # Ensure proper punctuation spacing
        response = response.replace(' ?', '?').replace(' .', '.')
        
        return response.strip()