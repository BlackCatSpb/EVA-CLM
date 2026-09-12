from __future__ import annotations

from typing import Any, Dict, List, Optional


class MirrorLRScheduler:
    """LR scheduler modulated by cognitive mirror state dynamics.

    Live mechanism (do not confuse with the legacy knobs below):
      warmup+blend: linear ramp of pg lr (and the mirror alpha-override/temp
      schedules) for `warmup` steps;
      growth-ratio multipliers (neutral at growth=1): var/alpha/gate fast/slow
      EMA ratios → LR up when specialization grows, down when stalled;
      mag_factor (cap): |mirror| rising above its own EMA → LR reduced
      (counter-cyclical); boost >1 only while val is on a downtrend;
      Loss damping (persistent): val regression >lr_regress_rel → factor ×0.5,
      restored by lr_improve_thresh or a τ-gated warm-restart.

    `target_var`, `mag_threshold`, `lr_min_ratio`, `max_decay_steps`,
    `var_min_for_lr_decay` are the OLD λ-tied decay-knob API, still accepted
    (config parity with LambdaConfig self-check) but INERT since the EMA-
    relative redesign — the live thresholds are all self-referencing EMAs.
    """
    def __init__(self, model, optimizer, base_lr=None, warmup=1000,
                 target_var=0.161, mag_threshold=0.296, lr_min_ratio=0.026,
                 max_decay_steps=2584, var_min_for_lr_decay=0.008,
                 cfg=None):
        # legacy decay knobs: intentionally NOT stored (see docstring)
        del target_var, mag_threshold, lr_min_ratio, max_decay_steps, var_min_for_lr_decay
        self.cfg = cfg
        if cfg is not None:
            base_lr = base_lr or cfg.lr
            warmup = getattr(cfg, 'warmup_steps', warmup)
        self.model = model
        self.optimizer = optimizer
        self.base_lr = base_lr
        self._orig_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.warmup = warmup
        self._step = 0
        self._last_log = 0
        # Adaptive thresholds: EMA of mirror stats
        self._tau_var = None
        self._tau_mag = None
        self._tau_1malpha = None
        self._tau_gate_var = None
        self._tau_ema = 0.99
        # Per-layer var(log_scale) modulation (cfg.per_layer_ls_lr)
        self._ls_enabled = bool(getattr(cfg, 'per_layer_ls_lr', False)) if cfg is not None else False
        self._ls_fast = None
        self._ls_slow = None
        self._ls_mult = None
        # Observability / last-computed multipliers (for diagnostics & tests)
        self.last_mirror_mult = 1.0
        self.last_mult = 1.0
        # Trend signal: is validation actually on a downtrend? (hysteresis between evals)
        self._val_ema = None
        self._val_improving = False

    def _mirror_stats(self):
        var_sum = 0.0
        mag_sum = 0.0
        alpha_sum = 0.0
        gate_var_sum = 0.0
        n = len(self.model.layers)
        for layer in self.model.layers:
            m = layer.mirror
            ls = m.log_scale.data
            var_sum += ls.var().item()
            mag_sum += m._last_magnitude.item()
            alpha = m.alpha_diag.data
            alpha_sum += (1.0 - alpha).abs().mean().item()
            gate_var_sum += m._last_gates.var().item()
        return var_sum / n, mag_sum / n, alpha_sum / n, gate_var_sum / n

    def _update_ls_mult(self):
        """Per-layer multiplier from fast/slow EMA of var(log_scale) per layer.

        ratio_i = fast_i / slow_i  (trend detector: fast EMA tracks current level,
        slow EMA the long-term baseline). Rising variance -> ratio>1 -> mult<1
        (layer throttled), falling (specialization) -> mult>1 (layer boosted).
        """
        if not self._ls_enabled:
            self._ls_mult = None
            return None
        n = len(self.model.layers)
        vals = []
        for layer in self.model.layers:
            ls = layer.mirror.log_scale.data
            vals.append(ls.var().item())
        if self._ls_fast is None or len(self._ls_fast) != n:  # B14 (F4-05 sibling)
            self._ls_fast = list(vals)
            self._ls_slow = list(vals)
            self._ls_mult = [1.0] * n
            return self._ls_mult
        tf = getattr(self.cfg, 'ls_ema_fast', 0.99)
        ts = getattr(self.cfg, 'ls_ema_slow', 0.999)
        lo = getattr(self.cfg, 'ls_mult_min', 0.5)
        hi = getattr(self.cfg, 'ls_mult_max', 2.0)
        mults = []
        for i in range(n):
            self._ls_fast[i] = tf * self._ls_fast[i] + (1 - tf) * vals[i]
            self._ls_slow[i] = ts * self._ls_slow[i] + (1 - ts) * vals[i]
            # B13 (F4-07): a frozen log_scale layer has var==0 EXACTLY:
            # r = 0/1e-10 = 0.0 and 1.0/r raised ZeroDivisionError (Python
            # floats, not tensors). Undefined ratio -> neutral multiplier.
            _fa, _sl = self._ls_fast[i], self._ls_slow[i]
            if _fa < 1e-12 and _sl < 1e-12:
                mults.append(1.0)
                continue
            r = _fa / max(_sl, 1e-12)
            mults.append(max(lo, min(hi, 1.0 / max(r, 1e-12))))
        self._ls_mult = mults
        return mults

    def report_train_loss(self, train_loss, ce_loss=None):
        """Report training loss for LR damping. Uses CE (not total) to avoid pred_loss false dampings."""
        pass

    def report_val_loss(self, val_loss):
        """Adaptive LR damping — anchored to the historical best, no one-way ratchet.

        Replaces the old fixed thresholds (regress if val > EMA*1.0123; restore only
        if val < best*0.889). The old rule was unreachable at chance level (val never
        improves 11% early) so ``_loss_lr_factor`` could only ratchet down to the 0.05
        floor on ordinary eval noise — exactly the Run B LR-collapse trap.

        New rule, anchored to ``_best_val_loss`` (the known-good level, which does NOT
        chase a regression upward):
          - **regression** (damp ×0.5) only if ``val > best*(1+lr_regress_rel)``
            (default 5% — a real divergence, not the ~1% eval noise seen at chance);
          - **improvement** (full restore to 1.0) when ``val < best*lr_improve_thresh``
            (default 0.98, reachable) — learning is clearly working, so no damping.

        Because the baseline is the historical best (not a fast EMA) and the restore
        threshold is reachable, LR is damped only for genuine divergence and recovers
        whenever the model improves — the asymmetric ratchet is gone.
        """
        if not hasattr(self, '_best_val_loss'):
            self._best_val_loss = val_loss
            self._loss_lr_factor = 1.0
            self._val_ema = val_loss
            self._val_improving = False
            self._lr_damp_steps = 0
        # B12 (F3-07): _lr_damp_steps postdates the scheduler's pickled state;
        # a resume from an older best.pt HAS _best_val_loss but NOT the counter
        # -> first plateau damp raised AttributeError (verified both copies).
        if not hasattr(self, '_lr_damp_steps'):
            self._lr_damp_steps = 0
        regress_rel = getattr(self.cfg, 'lr_regress_rel', 0.05)
        improve_thresh = getattr(self.cfg, 'lr_improve_thresh', 0.98)
        if val_loss > self._best_val_loss * (1.0 + regress_rel):
            # genuine divergence relative to best -> damp
            self._loss_lr_factor = max(0.05, self._loss_lr_factor * 0.5)
            self._lr_damp_steps = 0  # reset counter on damp
        elif val_loss < self._best_val_loss * improve_thresh:
            # genuine improvement -> restore base LR
            self._best_val_loss = val_loss
            self._loss_lr_factor = 1.0
            self._lr_damp_steps = 0
        else:
            # Plateau zone: neither regression nor improvement.
            # Warm-restart: gradually restore LR over time (τ-gated, no magic constant).
            if self._loss_lr_factor < 1.0:
                self._lr_damp_steps += 1
                # Restore rate: 1/τ_damp_steps per step → smooth exponential approach
                # τ_damp = 200 steps (default) — reaches 0.95 in ~600 steps
                tau_damp = getattr(self.cfg, 'lr_warm_restart_tau', 200)
                restore_rate = 1.0 / tau_damp
                self._loss_lr_factor = min(1.0, self._loss_lr_factor + restore_rate)
        # Downtrend detector (the "is learning actually happening?" signal that
        # gates the upward LR path). Retained between evals (hysteresis) so a boost
        # window stays open for a while after an improving eval. Small tolerance
        # avoids flicker on eval noise.
        tol = getattr(self.cfg, 'lr_improve_tol', 0.002)
        self._val_improving = bool(val_loss < self._val_ema * (1.0 + tol))
        self._val_ema = 0.9 * self._val_ema + 0.1 * val_loss

    def step(self):
        self._step += 1
        warmup_end = self.warmup
        blend_steps = 50
        if self._step < warmup_end + blend_steps:
            self._ls_mult = None
            if self._step < warmup_end:
                mult = self._step / max(warmup_end, 1)
                override = max(0.0, 1.0 - mult * 0.7)
            else:
                blend = (self._step - warmup_end) / blend_steps
                mult = 1.0 - blend * 0.3
                override = 0.3 * max(0.0, 1.0 - blend)
            temp_max, temp_min = 2.0, 0.5
            if self._step < warmup_end:
                t = self._step / max(warmup_end, 1)
                temp = temp_max - t * (temp_max - temp_min)
            else:
                blend = min(1.0, (self._step - warmup_end) / blend_steps)
                temp = temp_min + (1.0 - blend) * (temp_max - temp_min) * 0.3
            for layer in self.model.layers:
                layer.mirror._alpha_override.fill_(override)
                layer.mirror._usefulness_temp.fill_(max(temp, 0.1))
        else:
            for layer in self.model.layers:
                layer.mirror._alpha_override.fill_(0.0)
            var, mag, mean_1malpha, gate_var = self._mirror_stats()

            if self._tau_var is None:
                self._tau_var = var + 1e-10
                self._tau_mag = mag + 1e-10
                self._tau_1malpha = mean_1malpha + 1e-10
                self._tau_gate_var = gate_var + 1e-10

            te = self._tau_ema
            self._tau_var = te * self._tau_var + (1 - te) * var
            self._tau_mag = te * self._tau_mag + (1 - te) * mag
            self._tau_1malpha = te * self._tau_1malpha + (1 - te) * mean_1malpha
            self._tau_gate_var = te * self._tau_gate_var + (1 - te) * gate_var

            var_ratio = var / self._tau_var
            var_mult = min(2.0, max(0.5, 1.0 / max(var_ratio, 1e-10)))

            alpha_ratio = mean_1malpha / self._tau_1malpha
            alpha_mult = min(2.0, max(0.5, 1.0 / max(alpha_ratio, 1e-10)))

            gate_ratio = gate_var / self._tau_gate_var
            gate_mult = min(2.0, max(0.5, 1.0 / max(gate_ratio, 1e-10)))

            mag_ratio = mag / self._tau_mag
            mag_factor = min(1.0, max(0.2, 1.0 / max(mag_ratio, 1e-10)))

            mirror_mult = (var_mult * alpha_mult * gate_mult) ** (1/3) * mag_factor
            boost_max = getattr(self.cfg, 'lr_boost_max', 2.0)
            m = max(0.2, mirror_mult)
            # Upward path: allow LR to climb ABOVE base. The old code hard-capped
            # at 1.0, so the adaptive multiplier could only ever DAMP. Boost is
            # permitted ONLY when validation is on a genuine downtrend
            # (self._val_improving) — otherwise a stalled/"dead" model would be
            # boosted and destabilized. Self-limiting: a boost raises the loss
            # landscape variance -> var_ratio>1 -> var_mult<1 -> multiplier falls
            # back on its own (negative feedback via the EMA baselines).
            if m > 1.0 and not getattr(self, '_val_improving', False):
                m = 1.0
            m = min(m, boost_max)
            self.last_mirror_mult = mirror_mult
            mult = m
            if hasattr(self, '_loss_lr_factor'):
                mult = mult * self._loss_lr_factor
            self.last_mult = mult
            self._update_ls_mult()

            if self._step - self._last_log >= 500:
                self._last_log = self._step
                tau_var = self._tau_var.item() if hasattr(self._tau_var, 'item') else self._tau_var
                ls_info = ''
                if self._ls_mult is not None:
                    ls_info = (f' ls_mult[min={min(self._ls_mult):.3f} '
                               f'max={max(self._ls_mult):.3f}]')
                print(f'  lr_adapt: var(ls)={var:.6f} |1-a|={mean_1malpha:.6f} '
                      f'gate_var={gate_var:.6f} |mirror|={mag:.4f} '
                      f'tau_var={tau_var:.6f} '
                      f'mult={mult:.4f} lr={self.base_lr*mult:.2e}{ls_info}')

        for i, pg in enumerate(self.optimizer.param_groups):
            if i < len(self._orig_lrs):
                pg['lr'] = self._orig_lrs[i] * mult
            # groups added AFTER scheduler init (e.g. bridge_conn aux head) keep
            # their own lr (set when the group was appended) -> don't index _orig_lrs.

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]

    def state_dict(self):
        sd = {
            'step': self._step,
            'last_log': self._last_log,
            'type': 'MirrorLRScheduler',
            'tau_var': self._tau_var,
            'tau_mag': self._tau_mag,
            'tau_1malpha': self._tau_1malpha,
            'tau_gate_var': self._tau_gate_var,
            'orig_lrs': self._orig_lrs,
        }
        if hasattr(self, '_best_val_loss'):
            sd['best_val_loss'] = self._best_val_loss
            sd['loss_lr_factor'] = self._loss_lr_factor
        if self._val_ema is not None:
            sd['val_ema'] = self._val_ema
            sd['val_improving'] = self._val_improving
        if self._ls_enabled and self._ls_fast is not None:
            sd['ls_fast'] = self._ls_fast
            sd['ls_slow'] = self._ls_slow
        return sd

    def load_state_dict(self, sd):
        self._step = sd.get('step', 0)
        self._last_log = sd.get('last_log', 0)
        self._tau_var = sd.get('tau_var')
        self._tau_mag = sd.get('tau_mag')
        self._tau_1malpha = sd.get('tau_1malpha')
        self._tau_gate_var = sd.get('tau_gate_var')
        if 'orig_lrs' in sd:
            # orig_lrs is positional (step() indexes groups by position); a
            # checkpoint saved before an optimizer-group-shaping change would
            # silently map wrong LRs onto new groups -> only accept when the
            # group count still matches, else keep the fresh snapshot.
            if len(sd['orig_lrs']) == len(self._orig_lrs):
                self._orig_lrs = sd['orig_lrs']
        if 'best_val_loss' in sd:
            self._best_val_loss = sd['best_val_loss']
            self._loss_lr_factor = sd.get('loss_lr_factor', 1.0)
        if 'val_ema' in sd:
            self._val_ema = sd['val_ema']
            self._val_improving = sd.get('val_improving', False)
        if self._ls_enabled:
            self._ls_fast = sd.get('ls_fast')
            self._ls_slow = sd.get('ls_slow')
