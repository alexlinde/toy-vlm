"""
Text processing module for the Toy VLM.
Handles tokenization and text-related functionality.
"""

import re
from typing import List, Set

# Token constants
MAX_SEQ_LEN = 20

class SimpleTokenizer:
    """Simple word-based tokenizer for the toy VLM."""
    
    def __init__(self):
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
        """Build vocabulary by deterministically enumerating every template.

        Renders every template in questions.txt for every shape, with the
        `random_other_shape` slot also ranging over every shape, so the
        vocabulary covers every word that could ever be rendered and is
        independent of RNG state (unlike random sampling, which could miss a
        rarely-drawn template's unique word).
        """
        from shapes import ShapeGenerator

        # Get all shape types
        shape_gen = ShapeGenerator()
        shapes = shape_gen.get_available_shapes()

        # Collect all unique words from questions and answers
        word_set = set()

        # Render every template for every (shape, random_other_shape) pair
        for template_name in question_generator.template_names:
            template = question_generator.env.get_template(template_name)
            for shape in shapes:
                for other_shape in shapes:
                    result = template.render(shape=shape, random_other_shape=other_shape)
                    question, answer = result.split('|', 1)
                    word_set.update(self._extract_words(question.strip()))
                    word_set.update(self._extract_words(answer.strip()))

        # Add shape names directly to ensure they're included
        word_set.update(shapes)

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
    
    @classmethod
    def from_vocab(cls, vocab: dict):
        """Create tokenizer instance from an in-memory vocabulary dict."""
        tokenizer = cls()
        tokenizer.vocab = dict(vocab)
        tokenizer._update_mappings()
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
                if skip_special_tokens and word.startswith('<') and word.endswith('>') and word != '<UNK>':
                    continue
                words.append(word)
        
        return ' '.join(words)
    
    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)

    def oov_words(self, text: str) -> List[str]:
        """Return the unique out-of-vocabulary words in text, in first-seen order."""
        words = self._preprocess_text(text).split()
        seen = set()
        result = []
        for word in words:
            if word not in self.vocab and word not in seen:
                seen.add(word)
                result.append(word)
        return result

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
            # BOS + question + answer + EOS must fit in MAX_SEQ_LEN, or the loss_mask
            # slice assignment below would silently extend past it (collate crash).
            needed = len(q_tokens) + len(a_tokens) + 2
            assert needed <= MAX_SEQ_LEN, (
                f"Sequence too long for MAX_SEQ_LEN={MAX_SEQ_LEN}: "
                f"question={question!r} ({len(q_tokens)} tokens), "
                f"answer={answer!r} ({len(a_tokens)} tokens), needed={needed}"
            )
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