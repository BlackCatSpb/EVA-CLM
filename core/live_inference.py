"""
LiveInference + MirrorMonitor — continuous stateful inference
with internal state tracing for the WideBind model.

LiveInference:  Wraps WideBindStack, preserves global_state + layer states
                between calls. Supports think() (self-dialogue without input)
                and respond() (process actual input).

MirrorMonitor:  Non-invasive tracer. After each forward, reads per-expert gates,
                mirror magnitudes, log_scale stats from all layers. Stores
                rolling traces for analysis / visualization.
"""

import torch
from .stack import WideBindStack, AdaptiveController


class MirrorMonitor:
    """Rolling tracer of internal model states.

    Call capture() after each model.forward() to log per-layer and per-expert
    metrics.  History is stored as lists of tensors for later analysis.
    """

    def __init__(self, model: WideBindStack, max_history: int = 5000):
        self.model = model
        self.max_history = max_history
        self.clear()

    def clear(self):
        self.history = {
            'step': [],
            'expert_gates': [],        # (n_layers, G) — per-expert meta-gate
            'mirror_mag': [],          # (n_layers,) — average |mirror|
            'log_scale_mean': [],      # (n_layers,) — per-dim scale avg
            'log_scale_std': [],       # (n_layers,) — per-dim scale spread
            'i_gate': [],              # (n_layers,) — input gate (softplus)
            'tau': [],                 # (n_layers,) — mirror timescale
            'exploration': [],         # scalar — adaptive controller expl
            'global_state_norm': [],   # scalar — ||global_state||
        }

    def capture(self, global_state=None):
        """Read internal metrics from all layers.
        Must be called AFTER model.forward().
        """
        m = self.model
        l = len(m.layers)

        gates = torch.zeros(l, m.layers[0].mirror.G)
        mag = torch.zeros(l)
        ls_mean = torch.zeros(l)
        ls_std = torch.zeros(l)
        i_gates = torch.zeros(l)
        taus = torch.zeros(l)

        for i, layer in enumerate(m.layers):
            mir = layer.mirror
            gates[i] = mir._last_gates
            mag[i] = mir._last_magnitude
            # log_scale is a learned parameter, always accessible
            ls = mir.log_scale.detach()
            ls_mean[i] = ls.mean().item()
            ls_std[i] = ls.std().item()
            # Adaptive gate biases: i_gate = softplus(mean(h)*w_i + b_i)
            # _last_h_pool is (G, d) — pooled h_g over B, L.
            # Reshape to (D,) to match w_i, b_i
            h_mean_d = mir._last_h_pool.reshape(-1).detach()  # (D,)
            gate_logits = h_mean_d * layer.w_i + layer.b_i
            i_gates[i] = torch.nn.functional.softplus(gate_logits).mean().item()
            # Tau from b_d: tau = exp(b_d)
            taus[i] = torch.exp(layer.b_d).mean().item()

        expl_val, _ = AdaptiveController.stats(m.layers)

        self.history['step'].append(None)  # filled by LiveInference
        self.history['expert_gates'].append(gates)
        self.history['mirror_mag'].append(mag)
        self.history['log_scale_mean'].append(ls_mean)
        self.history['log_scale_std'].append(ls_std)
        self.history['i_gate'].append(i_gates)
        self.history['tau'].append(taus)
        self.history['exploration'].append(expl_val)
        self.history['global_state_norm'].append(
            global_state.norm(dim=(-2, -1)).mean().item() if global_state is not None else 0.0
        )

        # Trim if over max_history
        if len(self.history['step']) > self.max_history:
            for k in self.history:
                self.history[k] = self.history[k][-self.max_history:]

    def summary(self, window=100):
        """Return a dict of mean/std over the last `window` steps."""
        n = len(self.history['step'])
        if n == 0:
            return {}
        w = min(window, n)
        s = {}
        for k in ['exploration', 'global_state_norm']:
            arr = [v for v in self.history[k][-w:] if v is not None]
            if arr:
                s[k] = (sum(arr) / len(arr),
                        (sum((x - sum(arr) / len(arr)) ** 2 for x in arr)
                         / len(arr)) ** 0.5)
        # Per-layer averages
        gates = torch.stack(self.history['expert_gates'][-w:])  # (w, L, G)
        s['expert_gates_mean'] = gates.mean(dim=(0, 2))  # (L,)
        s['expert_gates_std'] = gates.std(dim=(0, 2))
        s['expert_gates_spread'] = gates.mean(dim=0).std(dim=-1)  # (L,) — how differentiated are experts
        mag = torch.stack(self.history['mirror_mag'][-w:])  # (w, L)
        s['mirror_mag'] = mag.mean(dim=0)
        s['i_gate'] = torch.stack(self.history['i_gate'][-w:]).mean(dim=0)
        s['tau'] = torch.stack(self.history['tau'][-w:]).mean(dim=0)
        return s


class LiveInference:
    """Continuous stateful inference wrapper for WideBindStack.

    Maintains global_state and layer states between forward calls.
    Enables "self-dialogue" via think() — runs mirror exchange even
    without external input.

    Usage:
        model = WideBindStack(cfg)
        live = LiveInference(model, cfg)

        # Model "lives" — internal state evolves
        for _ in range(100):
            live.think()  # self-dialogue step

        # Process actual input
        h = model.embed_tokens(input_ids)
        out = live.respond(h)

        # Check what happened inside
        summary = live.monitor.summary(window=50)
    """

    def __init__(self, model: WideBindStack, cfg,
                 monitor: bool = True, max_history: int = 5000):
        self.model = model
        self.cfg = cfg
        self.layer_states = None
        self.global_state = None
        self.intent_state = None
        self.step = 0

        if monitor:
            self.monitor = MirrorMonitor(model, max_history=max_history)
        else:
            self.monitor = None

    def think(self, n_steps: int = 1, h: torch.Tensor = None) -> torch.Tensor:
        """Run internal self-dialogue steps.

        If h is None, feeds a zero activation (minimal "think" token).
        Between steps, the last output is fed as next input so the
        internal state evolves continuously.

        Returns the final hidden state after n_steps.
        """
        out = None
        for _ in range(n_steps):
            if h is None:
                h = torch.zeros(1, 1, self.cfg.D,
                                device=next(self.model.parameters()).device)
            out, new_states, self.global_state, self._reasoning = self.model(
                h, self.layer_states, global_state=self.global_state,
                intent_state=self.intent_state
            )
            self.layer_states = new_states
            self.intent_state = getattr(self.model, '_last_intent_state', None)
            self.step += 1

            if self.monitor is not None:
                self.monitor.capture(self.global_state)
                self.monitor.history['step'][-1] = self.step

            # Feed last output as next input for continuous evolution
            h = out[:, -1:, :].detach()

        return out

    def respond(self, h: torch.Tensor) -> torch.Tensor:
        """Process an actual input through the live model.

        Unlike think(), this does NOT feed the output back as input —
        the sequence length of h determines the context.
        """
        out, new_states, self.global_state, self._reasoning = self.model(
            h, self.layer_states, global_state=self.global_state,
            intent_state=self.intent_state
        )
        self.layer_states = new_states
        self.intent_state = getattr(self.model, '_last_intent_state', None)
        self.step += 1

        if self.monitor is not None:
            self.monitor.capture(self.global_state)
            self.monitor.history['step'][-1] = self.step

        return out

    def reset_state(self):
        """Reset all internal states (layer states + global_state)."""
        self.layer_states = None
        self.global_state = None
        self.intent_state = None
        self.step = 0
        if self.monitor is not None:
            self.monitor.clear()

    def generate(self, prompt_ids, gen_len=100, think_steps=0):
        """Convenience: think (optional) -> prefill -> generate tokens."""
        self.respond(self.model.embed_tokens(prompt_ids))

        for _ in range(think_steps):
            self.think()

        tokens = []
        h = None  # will use last output from respond
        for _ in range(gen_len):
            out = self.think(n_steps=1, h=h)
            logits = self.model.lm_head(out)
            next_id = logits[:, -1].argmax(dim=-1).item()
            tokens.append(next_id)
            h = None  # think() will use its own loopback

        return tokens
