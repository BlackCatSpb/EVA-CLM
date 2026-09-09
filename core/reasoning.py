"""
Explicit Reasoning Module for EVA.
Adds chain-of-thought reasoning with thinking tokens.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ReasoningTokens:
    """Special tokens for chain-of-thought reasoning."""
    THINK: int = 65536   # <think>
    STEP: int = 65537    # <step>
    ANSWER: int = 65538  # <answer>
    END: int = 65539     # </think>


class ReasoningMemory(nn.Module):
    """Explicit reasoning memory for chain-of-thought.

    Stores intermediate reasoning steps in a dedicated fixed-size buffer
    and attends to them via sigmoid attention (independent per-step gates).

    Args:
        D: Hidden dimension.
        max_steps: Maximum number of reasoning steps to store.
    """

    def __init__(self, D: int, max_steps: int = 8) -> None:
        super().__init__()
        self.D: int = D
        self.max_steps: int = max_steps

        self.step_encoder: nn.Linear = nn.Linear(D, D)
        self.step_query: nn.Linear = nn.Linear(D, D)
        self.step_key: nn.Linear = nn.Linear(D, D)
        self.step_value: nn.Linear = nn.Linear(D, D)
        self.output_proj: nn.Linear = nn.Linear(D, D)

    def forward(
        self,
        h: torch.Tensor,
        reasoning_buffer: Optional[torch.Tensor] = None,
        reasoning_count: Optional[torch.Tensor] = None,
        record: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for reasoning memory.

        Args:
            h: (B, L, D) current hidden state.
            reasoning_buffer: (B, max_steps, D) previous reasoning steps or None.
            reasoning_count: scalar long tensor = number of valid rows.
            record: Whether to write this step to the buffer.

        Returns:
            output: (B, D) reasoning contribution.
            new_buffer: (B, max_steps, D) updated buffer.
            new_count: scalar tensor updated count.
        """
        B: int
        L: int
        D: int
        B, L, D = h.shape

        current_step: torch.Tensor = self.step_encoder(h[:, -1:, :])  # (B, 1, D)

        if reasoning_buffer is None:
            reasoning_buffer = torch.zeros(B, self.max_steps, D, device=h.device, dtype=h.dtype)
        if reasoning_count is None:
            reasoning_count = torch.zeros((), dtype=torch.long, device=h.device)

        q: torch.Tensor = self.step_query(current_step)
        k: torch.Tensor = self.step_key(reasoning_buffer)
        v: torch.Tensor = self.step_value(reasoning_buffer)

        attn: torch.Tensor = torch.sigmoid(q @ k.transpose(-2, -1) / math.sqrt(D))
        mask: torch.Tensor = (torch.arange(self.max_steps, device=h.device, dtype=h.dtype)
                < reasoning_count.to(h.dtype)).view(1, 1, self.max_steps)
        attn = attn * mask
        context: torch.Tensor = attn @ v

        combined: torch.Tensor = current_step + context
        output: torch.Tensor = self.output_proj(combined)
        empty: torch.Tensor = (reasoning_count <= 0).to(h.dtype)
        output = current_step * empty + output * (1.0 - empty)

        if record:
            new_count: torch.Tensor = (reasoning_count + 1).clamp(max=self.max_steps)
            row_idx: torch.Tensor = reasoning_count.clamp(max=self.max_steps - 1)
            one_hot: torch.Tensor = F.one_hot(row_idx, self.max_steps).float().view(1, self.max_steps, 1)
            full: torch.Tensor = (reasoning_count >= self.max_steps).float()
            shifted: torch.Tensor = torch.cat(
                [reasoning_buffer[:, 1:], torch.zeros_like(reasoning_buffer[:, :1])], dim=1)
            buf_pre: torch.Tensor = shifted * full + reasoning_buffer * (1.0 - full)
            new_buffer: torch.Tensor = (buf_pre * (1.0 - one_hot) + current_step.detach() * one_hot).detach()
        else:
            new_buffer = reasoning_buffer
            new_count = reasoning_count

        return output.squeeze(1), new_buffer, new_count


class ReasoningGate(nn.Module):
    """Per-step decision gates for adaptive reasoning depth.

    The reasoning loop runs up to `max_steps` iterations; each iteration i
    produces a reasoning vector r_i. A gate α_i = σ(Linear(h)) ∈ (0,1) decides
    how much of r_i is added to the hidden state. The loop stops early when
    the gate falls below `stop_threshold` (adaptive depth per token).

    Critical property: the gates are initialized so that the FIRST step is
    fully on (bias[0]=+10 → tanh≈1.0) and the REST are off (bias=0 with
    zero-weight projections → tanh(0)=0). This makes an adaptive model resume
    from a checkpoint trained with the old single-step reasoning EXACTLY as
    before; the +10 saturation intentionally freezes gate-0 at ON early on
    (its gradient through tanh is ~0; the stop decision is carried by the
    later gates and by the STE path in EVAStack._adaptive_reasoning).

    Args:
        D: Hidden dimension.
        max_steps: Maximum reasoning steps.
        know_dim: Dimension of the knowledge signal.
    """

    def __init__(self, D: int, max_steps: int = 8, know_dim: int = 8) -> None:
        super().__init__()
        self.max_steps: int = max_steps
        self.proj: nn.Linear = nn.Linear(D, max_steps)
        nn.init.zeros_(self.proj.weight)
        self.know_proj: nn.Linear = nn.Linear(know_dim, max_steps)
        nn.init.zeros_(self.know_proj.weight)
        nn.init.zeros_(self.know_proj.bias)
        self.r_proj: nn.Linear = nn.Linear(D, max_steps)
        nn.init.zeros_(self.r_proj.weight)
        nn.init.zeros_(self.r_proj.bias)
        with torch.no_grad():
            self.proj.bias[0] = 10.0
            self.proj.bias[1:] = 0.0

    def forward(
        self,
        h: torch.Tensor,
        know: Optional[torch.Tensor] = None,
        r: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute reasoning gates.

        Args:
            h: (B, L, D) hidden state.
            know: (B, know_dim) knowledge signal.
            r: (B, 1, D) reasoning candidate.

        Returns:
            gates: (B, L, max_steps) in (-1, 1) via tanh.
        """
        logit: torch.Tensor = self.proj(h)
        if know is not None:
            logit = logit + self.know_proj(know).unsqueeze(1)
        if r is not None:
            logit = logit + self.r_proj(r)
        return torch.tanh(logit)

    def logits(
        self,
        h: torch.Tensor,
        know: Optional[torch.Tensor] = None,
        r: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Raw logits (B, L, max_steps) for straight-through gradient."""
        logit: torch.Tensor = self.proj(h)
        if know is not None:
            logit = logit + self.know_proj(know).unsqueeze(1)
        if r is not None:
            logit = logit + self.r_proj(r)
        return logit


class ThinkingTokenHead(nn.Module):
    """Head that predicts thinking tokens for explicit reasoning.

    Args:
        D: Hidden dimension.
        num_reasoning_tokens: Number of special reasoning tokens.
    """

    def __init__(self, D: int, num_reasoning_tokens: int = 4) -> None:
        super().__init__()
        self.reasoning_proj: nn.Linear = nn.Linear(D, num_reasoning_tokens)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Predict reasoning token logits.

        Args:
            h: (B, L, D) hidden state.

        Returns:
            logits: (B, L, num_reasoning_tokens).
        """
        return self.reasoning_proj(h)
