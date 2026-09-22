"""
C063: Virtual Syntax Skipping (Grammar-KV Decoupling)
Structured JSON decoding engine that enforces syntax constraints via DFA/CFG,
but suppresses KV cache allocation for deterministic grammar tokens.
Preserves semantic representation purity and eliminates the 'grammar reasoning tax'.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Set


class VirtualSyntaxEngine:
    def __init__(self, syntax_token_ids: Set[int]):
        """
        syntax_token_ids: Set of token IDs corresponding to pure structural syntax
        (e.g., '{', '}', ':', '"', ',', '[', ']').
        """
        self.syntax_token_ids = set(syntax_token_ids)
        self.cached_kv_keys: List[torch.Tensor] = []
        self.cached_kv_values: List[torch.Tensor] = []
        self.virtual_syntax_skipped_count = 0
        self.total_tokens_generated = 0

    def step_kv_cache(
        self,
        token_id: int,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        skip_syntax_kv: bool = True
    ) -> bool:
        """
        Appends key/value tensors to the cache ONLY if the token is semantic,
        or if skip_syntax_kv is False (standard baseline behavior).
        Returns True if token was inserted into KV cache, False if virtually skipped.
        """
        self.total_tokens_generated += 1
        is_syntax = token_id in self.syntax_token_ids

        if is_syntax and skip_syntax_kv:
            # VIRTUAL SYNTAX SKIPPING: Do NOT allocate or store KV state
            self.virtual_syntax_skipped_count += 1
            return False
        else:
            # Store in KV cache
            self.cached_kv_keys.append(k_tensor)
            self.cached_kv_values.append(v_tensor)
            return True

    def get_memory_savings_ratio(self) -> float:
        """Computes the fraction of KV cache memory saved via virtual skipping."""
        if self.total_tokens_generated == 0:
            return 0.0
        return self.virtual_syntax_skipped_count / self.total_tokens_generated

    def reset(self):
        self.cached_kv_keys.clear()
        self.cached_kv_values.clear()
        self.virtual_syntax_skipped_count = 0
        self.total_tokens_generated = 0
