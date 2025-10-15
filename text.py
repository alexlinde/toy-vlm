"""
Text processing module for the Toy VLM.
Handles tokenization and text-related functionality.
"""

import re
import json
import os
from typing import List, Set

# Token constants
MAX_SEQ_LEN = 20

class SimpleTokenizer:
    """Simple word-based tokenizer for the toy VLM."""
    
    def __init__(self, vocab_file: str = None):
        if vocab_file and os.path.exists(vocab_file):
            # Load pretrained vocabulary
            self.load_vocab(vocab_file)
        else:
            # Initialize with base vocabulary - will be extended during training
            self.vocab = {
                '<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3
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
    
    def build_vocab_from_questions(self, question_generator):
        """Build vocabulary by analyzing all possible questions and answers."""
        from shapes import ShapeGenerator
        
        # Get all shape types
        shape_gen = ShapeGenerator()
        shapes = shape_gen.get_available_shapes()
        
        # Collect all unique words from questions and answers
        word_set = set()
        
        # Generate samples for each shape and template
        for shape in shapes:
            for _ in range(10):  # Multiple samples per shape to get variety
                question, answer = question_generator.generate_qa_pair(shape)
                
                # Extract words from question and answer
                word_set.update(self._extract_words(question))
                word_set.update(self._extract_words(answer))
        
        # Add shape names directly to ensure they're included
        word_set.update(shapes)
        
        # Add common shape-related words that might be missing
        common_words = {
            'shape', 'round', 'flat', 'big', 'small', 'object', 'figure',
            'red', 'blue', 'green', 'white', 'black', 'color', 'size'
        }
        word_set.update(common_words)
        
        # Build vocabulary starting after special tokens
        next_idx = len(self.vocab)
        for word in sorted(word_set):  # Sort for consistent ordering
            if word not in self.vocab:
                self.vocab[word] = next_idx
                next_idx += 1
        
        self._update_mappings()
        print(f"Built vocabulary with {len(self.vocab)} tokens from QuestionGenerator")
        
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
        """Remove non-alpha characters and normalize."""
        # Convert to lowercase and keep only alphabetic characters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        # Normalize whitespace
        text = ' '.join(text.split())
        return text
        
    def tokenize(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        text = self._preprocess_text(text)
        words = text.split()
        
        tokens = []
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(self.unk_token_id)
        
        return tokens
    
    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to text."""        
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
    
    def prepare_input_sequence(self, question: str, answer: str = None) -> tuple:
        """Prepare input and target sequences for training or inference."""
        q_tokens = self.tokenizer.tokenize(question)
        
        if answer is not None:
            # Training mode - include answer
            a_tokens = self.tokenizer.tokenize(answer)
            input_tokens = self.pad_sequence([self.tokenizer.bos_token_id] + q_tokens + a_tokens)
            target_tokens = self.pad_sequence(q_tokens + a_tokens + [self.tokenizer.eos_token_id])
            loss_mask = [0] * MAX_SEQ_LEN
            loss_mask[len(q_tokens):len(q_tokens) + len(a_tokens) + 1] = [1] * (len(a_tokens) + 1)
        else:
            # Inference mode - only question
            input_tokens = self.pad_sequence([self.tokenizer.bos_token_id] + q_tokens)
            target_tokens = None
            loss_mask = None
        
        return input_tokens, target_tokens, loss_mask
    
    def pad_sequence(self, tokens: List[int]) -> List[int]:
        """Pad or truncate sequence to max_length."""
        if len(tokens) > MAX_SEQ_LEN:
            tokens = tokens[:MAX_SEQ_LEN]
        else:
            tokens = tokens + [self.tokenizer.pad_token_id] * (MAX_SEQ_LEN - len(tokens))
        return tokens
    
    def clean_response(self, response: str) -> str:
        """Clean up generated response text."""
        # Remove extra spaces and normalize
        response = ' '.join(response.split())
        
        # Ensure proper punctuation spacing
        response = response.replace(' ?', '?').replace(' .', '.')
        
        return response.strip()