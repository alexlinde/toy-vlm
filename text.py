"""
Text processing module for the Toy VLM.
Updated to support prefixing image tokens in the sequence and
multi-turn-friendly special tokens for user/assistant and reasoning spans.
"""

import re
import json
import os
from typing import List, Set
import torch
from shapes import ObjType

# Token/sequence constants
# Increased to support prefixed image tokens
MAX_SEQ_LEN = 128
NUM_IMG_TOKENS = 64

class SimpleTokenizer:
    """Simple word-based tokenizer for the toy VLM (word-level)."""
    
    def __init__(self, vocab_file: str = None):
        if vocab_file and os.path.exists(vocab_file):
            # Load pretrained vocabulary
            self.load_vocab(vocab_file)
        else:
            # Initialize with base vocabulary for dialog + image-token format
            self.vocab = {
                '<PAD>': 0,
                '<BOS>': 1,
                '<EOS>': 2,
                '<UNK>': 3,
                '<|user|>': 4,
                '<|assistant|>': 5,
                '<THINK>': 6,
                '</THINK>': 7,
                '<FINAL>': 8,
                '</FINAL>': 9,
                '<IMG_START>': 10,
                '<IMG_END>': 11,
                '<IMG>': 12,
            }
            self._update_mappings()
    
    def _update_mappings(self):
        """Update reverse mapping and special token IDs."""
        # Create reverse mapping
        self.idx_to_word = {idx: word for word, idx in self.vocab.items()}

        # Special token IDs
        self.pad_token_id = self.vocab['<PAD>']
        self.bos_token_id = self.vocab['<BOS>']
        self.eos_token_id = self.vocab['<EOS>']
        self.unk_token_id = self.vocab['<UNK>']

        # Conversation and reasoning markers
        self.user_token_id = self.vocab['<|user|>']
        self.assistant_token_id = self.vocab['<|assistant|>']
        self.think_start_id = self.vocab['<THINK>']
        self.think_end_id = self.vocab['</THINK>']
        self.final_start_id = self.vocab['<FINAL>']
        self.final_end_id = self.vocab['</FINAL>']

        # Image markers
        self.img_start_id = self.vocab['<IMG_START>']
        self.img_end_id = self.vocab['<IMG_END>']
        self.img_token_id = self.vocab['<IMG>']
    
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
                    image, metadata = shape_gen.generate_multi_shape_image(1, False)

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
        # add shapes and digits 0-9
        word_set.update({e.value for e in list(ObjType)})
        word_set.update({str(i) for i in range(10)})
        
        # add common words that might be missing
        word_set.update({'yes', 'no', 'is', 'are', 'there', 'count', 'compare', 'equal', 'greater', 'less'})

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
        rationale: str = None
    ) -> tuple:
        """
        Compose sequence per new format:

        input_ids  = [BOS] + [USER] + q_ids + img_block + [ASSIST]
                     + [THINK] + r_ids + [THINK_END]
                     + [FINAL] + a_ids + [FINAL_END] + [EOS]
        target_ids = input_ids[1:] + [EOS]
        """
        tok = self.tokenizer

        q_tokens = tok.tokenize(question)
        a_tokens = tok.tokenize(answer) if answer is not None else []
        r_tokens = tok.tokenize(rationale) if rationale is not None else []

        img_block = [tok.img_start_id] + [tok.img_token_id] * NUM_IMG_TOKENS + [tok.img_end_id]

        input_ids = (
            [tok.bos_token_id] +
            [tok.user_token_id] + q_tokens +
            img_block +
            [tok.assistant_token_id] +
            [tok.think_start_id] + r_tokens + [tok.think_end_id] +
            [tok.final_start_id] + a_tokens + [tok.final_end_id] +
            [tok.eos_token_id]
        )

        # Standard next-token targets: predict input_ids shifted by 1
        target_ids = input_ids[1:]

        # Build span masks per policy (masks align to target_ids = input_ids[1:]):
        # We supervise predictions of tokens strictly inside each span.
        # - rat_mask: supervise predictions of rationale content tokens only
        # - ans_mask: supervise predictions of answer content tokens only
        rat_mask = [0] * len(target_ids)
        ans_mask = [0] * len(target_ids)

        # Supervision windows defined on source positions (k indexes input_ids[k] -> predicts input_ids[k+1])
        def find_pos(tid):
            try:
                return input_ids.index(tid)
            except ValueError:
                return -1

        th_s = find_pos(tok.think_start_id)
        th_e = find_pos(tok.think_end_id)
        fn_s = find_pos(tok.final_start_id)
        fn_e = find_pos(tok.final_end_id)

        # Map to target positions: target[k] == input_ids[k+1]
        # Supervise content tokens AND the closing tag for each span.
        for k in range(len(target_ids)):
            pred_idx = k + 1  # index into input_ids of the token being predicted
            if th_s != -1 and th_e != -1 and th_s < pred_idx <= th_e:
                rat_mask[k] = 1
            if fn_s != -1 and fn_e != -1 and fn_s < pred_idx <= fn_e:
                ans_mask[k] = 1

        # Diagnostics and structural assertions
        # Ensure marker order and presence
        def find_pos(tid):
            try:
                return input_ids.index(tid)
            except ValueError:
                return -1

        bos_p = 0 if len(input_ids) > 0 and input_ids[0] == tok.bos_token_id else -1
        usr_p = find_pos(tok.user_token_id)
        asst_p = find_pos(tok.assistant_token_id)
        th_s = find_pos(tok.think_start_id)
        th_e = find_pos(tok.think_end_id)
        fn_s = find_pos(tok.final_start_id)
        fn_e = find_pos(tok.final_end_id)
        eos_p = len(input_ids)-1 if len(input_ids) and input_ids[-1] == tok.eos_token_id else -1

        assert bos_p == 0, "[text] <BOS> must be at position 0"
        assert usr_p > 0, "[text] Missing <|user|>"
        img_s = find_pos(tok.img_start_id)
        img_e = find_pos(tok.img_end_id)
        assert img_s > usr_p and img_e > img_s, "[text] Invalid image block order"
        # Check placeholders count
        num_placeholders = sum(1 for t in input_ids if t == tok.img_token_id)
        assert num_placeholders == NUM_IMG_TOKENS, f"[text] Expected {NUM_IMG_TOKENS} <IMG> tokens, found {num_placeholders}"
        assert asst_p > usr_p, "[text] <|assistant|> must come after <|user|>"
        # THINK and FINAL sections exist even if empty strings were passed (they may be contiguous)
        assert th_s > asst_p and th_e > th_s, "[text] Invalid THINK span"
        assert fn_s > th_e and fn_e > fn_s, "[text] Invalid FINAL span"
        assert eos_p == len(input_ids)-1, "[text] <EOS> must be final token"

        # Mask policy sanity: special markers must be unsupervised
        special_set = {
            tok.bos_token_id, tok.user_token_id, tok.assistant_token_id,
            tok.think_start_id, tok.think_end_id, tok.final_start_id, tok.final_end_id,
            tok.eos_token_id, tok.img_start_id, tok.img_end_id, tok.img_token_id
        }
        allowed_special_to_supervise = {tok.think_end_id, tok.final_end_id}
        for k, tid in enumerate(target_ids):
            if tid in special_set and tid not in allowed_special_to_supervise:
                assert rat_mask[k] == 0 and ans_mask[k] == 0, f"[text] Mask error: supervising special token id={tid} at k={k}"

        return input_ids, target_ids, rat_mask, ans_mask

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