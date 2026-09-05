"""Curriculum learning via loss-based chunk sampling (BlackCatSpb §2.8).

p(chunk_i) ∝ exp(L_i / τ_curr),  τ_curr(t) = τ_0 · exp(-t / T_decay) + τ_min

Using POSITIVE sign (exp(+L/τ)), unlike the audit's eq which had a sign error.
At τ→0: concentrates on hardest chunks (highest L).
At τ→∞: uniform sampling.
"""
import math
import torch


class CurriculumTracker:
    """Tracks per-stream/chunk losses and computes temperature-decayed sampling probs.

    Usage:
        tracker = CurriculumTracker(n_streams=10, tau_0=2.0, tau_min=0.1, decay_steps=50000)
        for step in range(max_steps):
            probs = tracker.sample_probs(step)
            stream_idx = torch.multinomial(probs, 1).item()
            x, y = stream.get_batch(...)
            loss = model.compute_loss(...)
            tracker.update(stream_idx, loss.item())
    """
    def __init__(self, n_streams, tau_0=2.0, tau_min=0.1, decay_steps=50000, momentum=0.95):
        self.n = n_streams
        self.tau_0 = tau_0
        self.tau_min = tau_min
        self.decay_steps = decay_steps
        self.momentum = momentum
        self.ema_loss = torch.zeros(n_streams)
        self._steps = 0

    def to(self, device):
        self.ema_loss = self.ema_loss.to(device)
        return self

    @property
    def tau(self):
        """Temperature: τ_0 · exp(-t / T_decay) + τ_min"""
        frac = min(1.0, self._steps / max(self.decay_steps, 1))
        return self.tau_0 * math.exp(-self._steps / max(self.decay_steps, 1)) + self.tau_min

    def update(self, stream_idx, loss_val):
        """Update EMA loss for a stream."""
        if isinstance(stream_idx, torch.Tensor):
            stream_idx = stream_idx.item()
        self.ema_loss[stream_idx] = (
            self.momentum * self.ema_loss[stream_idx].item()
            + (1.0 - self.momentum) * loss_val
        )

    def sample_probs(self, cur_step):
        """(n_streams,) multinomial probabilities for stream selection.
        p_i ∝ exp(L_i / τ) — hard examples get higher prob as τ → 0.
        """
        self._steps = cur_step
        t = max(self.tau, 1e-8)
        # Center logits to prevent overflow
        L_centered = self.ema_loss - self.ema_loss.mean()
        logits = L_centered / t
        logits = logits - logits.max()  # numerical stability
        probs = torch.softmax(logits, dim=0)
        return probs
