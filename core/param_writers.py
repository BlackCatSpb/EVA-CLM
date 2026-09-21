"""P3-3: the registry of parameters with TWO writers (optimizer + a control law).

"Metacognition is inseparable from the parameters" formally means: part of the
parameters has a second writer — the optimizer (the fast inner loop) and a
control law (the slow outer loop, mutating `.data`). This is Borkar's
two-timescale stochastic approximation: separable and stable while
`tau_control >= SEPARATION * TAU_ADAM`.

The registry is enforced by an AST scanner (tests/test_dual_writer_census.py):
a new `.data` mutation of a registered-class leaf without a registry entry
turns the test red — the same discipline as `test_tau_lint`.

Categories:
  optimizer-only   — an ordinary parameter (no registry entry needed);
  dual-writer      — optimizer + a control law (.data mutation); the separation
                     condition applies (tau_ctrl in steps, None = quasi-static);
  quasi-static     — rare event writes (birth / index_copy), not a lerp class:
                     tau=None + a mandatory justification;
  buffer-writers   — `.data` writes into NON-parameters (buffers): not part of
                     the parameter census, listed for the scanner's benefit.
"""
from __future__ import annotations

TAU_ADAM = 10.0        # 1/(1-beta1) = 10 steps
SEPARATION = 10.0      # the two-timescale separation constant

# leaf name -> (writer, tau_ctrl in steps, note/justification)
DUAL_WRITERS = {
    'b_i': ('AdaptiveController -> stack.forward lerp', 1000.0,
            'vsa_b_d_smooth=0.999; the Adam side is slowed by vsa_b_lr_mult=0.1'),
    'b_d': ('AdaptiveController -> stack.forward lerp', 1000.0, 'same'),
    'alpha_diag': ('mirror self-regulation (pend/flush, 1/step)', 100.0,
                   'lerp 0.01; B13 F4-01: exactly one write per step'),
    'phantom_basis': ('EMA-steering by confirmed directions', 770.0,
                      '0.01 per observe, observe every 100 head-forwards (~12 steps)'),
    'concept_keys': ('functional write + birth_from_direction', None,
                     'quasi-static: rare events, not a lerp class'),
    'concept_vals': ('functional write + birth_from_direction', None,
                     'same'),
}

# `.data` writes into non-parameters (buffers) — excluded from the census but
# listed so the scanner can report them knowingly.
BUFFER_WRITERS = {
    'W_out': ('tie value mirror (pre-hook / _sync_W_out)', 'buffer, not a param'),
}
