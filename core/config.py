"""EVA configuration with λ_d hierarchy support."""

from dataclasses import dataclass, field
from .lambda_utils import LambdaConfig
from . import tau_api  # T8: единый язык временных констант (period/horizon/temp)

_LAMBDA_OVERRIDE_DOC = (
    "Set to None to use λ_d-derived value (recommended for Experiment 1)."
)


@dataclass
class WideBindConfig:
    D: int = 4096
    n_layers: int = 32
    bind_K: int = 64
    vocab: int = 65536  # B7: real corpus ids reach 65535 (50000 silently folded 7.8% of FANTASY)
    seq_len: int = 256
    batch_size: int = 2
    # M60 (the corpus arithmetic): the unigram's log-frequencies (~-3..-5) need
    # |Delta| ~ lr*steps, so 3e-4 took ~10-17k steps just for the unigram
    # (measured: token_bias std 0.0028 at 250 steps, ce_raw 10.8 vs the corpus
    # unigram 7.48 / bigram 4.21). 6e-4 halves that; the guards (AGC, M50/M51,
    # plateau damping) carry the spike risk.
    lr: float = 6e-4
    warmup_steps: int = 300   # M60: 1000 -> 300 (the full LR from step 300)
    weight_decay: float = 0.01
    # (grad_clip removed B3: AGC ratio c is the live clipper knob; this field
    # was constructed, logged and never consumed)
    dtype: str = 'float32'

    # False = обучать EOS-токен (границы предложений), True = маскировать (старое поведение)
    mask_eos: bool = False   # Colab: учим EOS (данные *_eos.bin — границы предложений)

    # ─── λ_d hierarchy ─────────────────────────────────────────
    lambda_d: int = 3            # dimension of generalized golden ratio
    lambda_d_enabled: bool = True  # True = apply λ_d derivation in __post_init__

    tie_bind: bool = True  # True = W_out = W_proj^T (autoencoder bind bottleneck)
    tie_mirror_proj: bool = True  # True = mirror W_out = W_proj^T (per-expert K-space AE)
    # P0-2 (F3) A/B-ручка: True = tie IN-GRAPH (градиент выходного пути течёт в
    # W_proj; измерено: h_norm ×15 и u-насыщение 0.66 на резюме 7040 — новая
    # тропа на 11k-шаговом чекпойнте взрывоопасна), False = прежнее чтение
    # буфера/зеркала (значения те же — буфер синхронизирован с W_proj^T — но
    # градиент выходного пути мёртв). Default False: смена градиентной тропы
    # на резюме — только осознанным A/B.
    tie_grad: bool = False

    # Variable Precision Memory
    variable_precision: bool = True   # add exact sequence memory on top of VSA (canonical Colab stack)
    precision_threshold: float = 0.3  # gate threshold to activate exact memory

    # Explicit Reasoning (chain-of-thought)
    explicit_reasoning: bool = True   # canonical Colab stack: reasoning loop enabled
    reasoning_max_steps: int = 8  # max reasoning steps in chain-of-thought
    reasoning_ramp_steps: int = 1000  # exp ramp of block influence: scale = 1 - exp(-t/ramp)
    reasoning_adaptive: bool = True   # per-step gates (adaptive depth) — canonical Colab stack
    reasoning_gate_stop_threshold: float = 0.5  # loop stops when mean gate < threshold

    # ─── Триада: Рассудок как участник (замыкание петли) ───
    # Верификатор (Рассудок) не только читает _knowledge_signal/_last_conf, но
    # при неуверенности ре-циркулирует ствол (повторный осмысленный проход =>
    # бóльшая эффективная глубина). Чистый control-flow: новых параметров нет,
    # переобучение не требуется. Активно только в inference/generation
    # (not self.training и step is not None), тренировка и валидация нетронуты.
    triad_reason: bool = True
    triad_conf_thr: float = 0.5       # порог уверенности lm_head: ниже -> ре-проход
    triad_max_passes: int = 3         # бюджет ре-циркуляций (защита от зацикливания)

    # AMP (Automatic Mixed Precision)
    use_amp: bool = False  # True = mixed precision (requires CUDA, ~2x speed)

    head_mode: str = "sigmoid_coded"
    head_normalize: bool = True
    head_bus_cap: float = 3.0     # M61: scale-invariant cap on the intent stencil
                                  # (|bus_bias| RMS; 0 disables) — the 165-spike:
                                  # zt += bus_bias was the unbounded channel
    head_u_wall: float = 1e-3     # M52a: soft wall on the head's bit log-odds
                                  # (keeps the CE gradient alive at |u|>u0);
                                  # 0 disables
    head_u_wall_u0: float = 6.0   # M52a: the wall's threshold
    head_lacuna: bool = True      # M52b: lacuna residual + phantom channel
    head_pair_rank: int = 0       # P1-2: rank-r pairwise (Ising) channel of the
                                  # head — 0 = off (identity at init, A/B arm 16).
                                  # The probe (scripts/probe_head_ceiling.py)
                                  # measured the bit-independence price at
                                  # ~4.2 nat/bigram and a rank-16 channel closing
                                  # ~67% of it for 2Kr = 2048 params.
    head_phantom_bits: int = 32   # M52b: K_p (the phantom basis rank)
    head_phantom_max: int = 64    # M59c (link C): the phantom channel's CAPACITY —
                                  # the active count grows in place from
                                  # head_phantom_bits up to this as the concepts
                                  # accumulate (no shape changes ever)
    head_phantom_noise: float = 0.05  # M52b: exploration-noise init (eta,
                                      # learnable; EVA-Ai used 0.05)
    head_phantom_l1: float = 1e-4  # M52b: light sparsity on the phantom firing
    head_srl: bool = False        # M53: compute the SRL classification telemetry
                                  # (concept/contradiction/lacuna) in the forward
    head_srl_every: int = 50      # M53e: compute the SRL telemetry every N forwards
                                  # (the live cost is ~25% tok/s + ~2.5GB; 1 = every step)
    head_srl_apply: bool = False  # M53d: ALSO apply the refinement to u (the
                                  # 1045 post-mortem: the hard pull snapped the bits
                                  # to the codes and killed the run - opt-in only)
    head_srl_steps: int = 3       # M53: refinement passes (T)
    head_srl_shortlist: int = 64  # M53: candidate codes per pass
    head_srl_expl_thr: float = 0.7  # M53: explanation cost (nats/bit) above
                                   # which a state is classified a lacuna
    head_srl_after: int = 1045    # M53c: SRL activates from this step (at init
                                  # the commitment to random codes doubled the CE:
                                  # 11.47 -> 22.39, measured); 0 = from the start
    head_phantom_after: int = 1045  # M53c: the bank observes from this step (at
                                    # init every residual is noise)
    mem_lacuna_k: float = 0.5     # M55a: memory-search broadening on the lacuna
                                  # (attn temp *= 1 + k*lacuna; 0 disables)
    head_temper: bool = True      # M55a: contradiction tempering (head<->memory)
    head_temper_k: float = 0.5    # M55a: logits /= (1 + k*chi)
    head_temper_cos: float = 0.3  # M55a: chi = relu(cos_thr - cos(implied, mem))
    head_temper_after: int = 1045  # M55a: warmup (the memory is noise at init)
    # ─── P4-3 (ContradictionField): межшкальный детектор + потребители ───
    contradiction_field: bool = False  # fast-vs-slow VSA детектор (буферы +
                                       # телеметрия; forward не меняет)
    head_temper_rel: bool = False      # A/B: относительная форма χ по _chi_ladder
                                       # (сложение с межшкальным χ + нормировка
                                       # на собственный рабочий уровень)
    phantom_chi_salience: bool = False  # A/B: противоречие как новизна для
                                        # наблюдения фантома (max двух салиентностей)
    head_phantom_slots: int = 16  # M54: the phantom-concept bank slots
    head_phantom_merge: float = 0.7  # M54: cosine >= merge -> the same phantom
    head_phantom_merge_lo: float = 0.2  # M64 (M63-C): the soft route — cosine
    # T8 role=design: порог мягкого слияния направлений (калиброван M63-C)
    # T9 (порт EVA-Ai ConceptMiner): подтверждение требует не только conf>=0.75,
    # но и не менее cycles_before_stable наблюдений (recurrence gate).
    head_phantom_cycles_stable: int = 5
                                         # >= merge_lo takes a similarity-weighted
                                         # EMA + a partial confidence bump. The
                                         # hard merge=0.7 is unreachable on
                                         # D=2560 residuals (measured: zero merges
                                         # in the whole run) -> the confirmation
                                         # path was dead by arithmetic. R1/R3: the
                                         # null best-cos is ~0.06 mean / 0.09 max,
                                         # the recurring-structure mode ~0.3 ->
                                         # 0.2 sits between; stats() reports the
                                         # percentiles for the real calibration.
    head_phantom_thr: float = 1.1  # M55b: RELATIVE lacuna (ell/EMA) above which a
                                   # position is observed (the absolute ell is ~0.97
                                   # for ANY realistic state: the readout spans K of D)
                                   # T9.7b: this is now the STATIC FALLBACK only —
                                   # the default mode 'noise' self-calibrates (1.1
                                   # sat above the whole signal range, p99 ~1.08).
    head_phantom_thr_mode: str = 'noise'  # T9.7b: 'noise' = the observation bar is
                                   # 1 + max(floor, k*MAD) over the BASE rate, with
                                   # MAD from the recent per-call salience maxima
                                   # (clamped). Not a quantile: no guaranteed firing
                                   # rate; on a calm stream the width collapses and
                                   # only a genuine exceedance over the base fires.
                                   # 'static' = head_phantom_thr.
    head_phantom_thr_k: float = 2.0       # robust deviations above the BASE rate
    head_phantom_thr_floor: float = 0.002  # the minimum excess over the base
    head_phantom_thr_lo: float = 1.0005   # clamp floor (degenerate-width guard)
    head_phantom_thr_hi: float = 1.10     # clamp ceiling
    head_phantom_thr_min_samples: int = 16  # ring samples before 'noise' takes over
    # T9.7b: the salience ladder (the head's "contradiction ladder"). Default =
    # the cache ms-ladder (8/32/128/512/2048/8192, ×4) so the salience horizons
    # are ALIGNED with the ms-cache scales; () -> tau_api.VSA_LADDER (4 rungs).
    head_lacuna_ladder: tuple = (8.0, 32.0, 128.0, 512.0, 2048.0, 8192.0)
    head_lacuna_gate_tau: float = 128.0   # rung read by the lacuna/temper gates
                                          # (≈ the old single-EMA ~100 drop-in)
    head_phantom_base_tau: float = 0.0    # rung read by the phantom observation
                                          # (0 = the longest rung = the base rate)
    head_lacuna_ema: float = tau_api.period(100)  # M55b: the self-calibration EMA decay
    # (T8: каденция — период 100 наблюдений, объявлен через tau_api.period)
    head_phantom_every: int = 25  # M54: observe cadence (steps)
    # T9.9 шаг 2: наблюдения фантома по ПУЛАМ предложений (семантические
    # единицы; средняя салиентность сегмента vs порог). False = per-position.
    phantom_sentence_level: bool = True
    # T9.9 шаг 2 (опции, default off — A/B): boundary-aware VSA (мягкий сброс
    # памяти на границах: decay *= 1 − strength·sep) и conv-стена (вход conv
    # обнуляется на SEP). Включать только по A/B: сброс меняет семантику памяти.
    vsa_boundary_reset: float = 0.0
    conv_boundary_wall: bool = False
    head_phantom_decay: float = tau_api.period(100)  # M62/M64: the bank's per-OBSERVE confidence
                                       # decay (0.999 = the old per-forward value).
                                       # R1/R2 calibration: at the observed ~0.13
                                       # observes/step (checkpoint counters) 0.99 gives the
                                       # 0.5 -> 0.25 transition in ~69 observes ~ 526 steps —
                                       # the confirmation window ballpark. M64.9r2: with the
                                       # notebook's 1000-step chunk a slot born in the first
                                       # ~470 steps does NOT survive to the next shift (~47%
                                       # do); the fade and the chunk must be compared
                                       # whenever either changes (see the whiteboard).
    code_dim: int = 32
    code_sparsity: int = 6
    embed_rope: bool = False
    # T9.9 (оператор): границы предложений как первоклассный сигнал для ВСЕЙ
    # модели (SEP id=2 читал только банк; ствол/голова/кэш видели плоский
    # поток). sent_eos_emb/sent_bos_emb/sent_pos_emb — zero-init (на старте
    # forward бит-в-бит прежний), выучиваются через CE. False = откат (A/B).
    sent_boundary_emb: bool = True
    sent_pos_max: int = 64       # B2: legacy rotary tag in embedding (off: see embedding.py)
    logit_cache_mode: str = 'topk'   # M18: 'profile' = store z@C on write, no V-work on read
    vsa_decay_floor_k: float = 2.0   # B18 (audit 02b F2B-02): decay floor exp(-k/tau_s);
                                     # content can shorten memory down to tau_s/k, never below.
    embed_center: bool = False   # B9 (audit 02a): remove the 95% common-mode of the codebook (zero-mean codes)
    codebook: str = 'legacy'   # 'twin_free' (B2): max pairwise overlap ≤ S−2, needs code_dim≥64 for full vocab

    mirror_k: int = 32
    mirror_k_staircase: bool = True  # True = k_l∈{8,16,32} по третям глубины
    mirror_tau_min: float = 2.0     # min τ for per-K-dimension alpha init (predictive mirror)
    mirror_tau_max: float = 200.0   # max τ for per-K-dimension alpha init (predictive mirror)
    w_pred_scale_init: float = 3.0
    log_scale_init_std: float = 0.05
    mlp_groups: int = 32
    mlp_expand: int = 4
    # Force uniform log_skip_alpha=0 on build/resume (SMF L0-depth fix).
    # Default False: only matters when resuming an OLD checkpoint that carries
    # the 17.8x L0 bias — set True to neutralize it without retraining from scratch.
    reset_skip_alpha: bool = False
    private_mem: bool = True  # cross-expert private memory bank (meta-cognitive layer)

    # Private-memory WRITE GATE.
    # При включённой maturation (maturation_enabled=True, по умолчанию) запись
    # управляется ЕДИНЫМ показателем зрелости M_l(t)=max(time/τ-рампа,
    # bridge_readiness), пересекающим matur_write_thr — см. mirror.py. Это
    # интеллектуальный (когнитивный) гейт: память не засевается ранне-случайными
    # состояниями (эхо-камера невозможна), а компетентный bridge подключает её
    # ровно когда полезно. В этом режиме pm_write_delay ИГНОРИРУЕТСЯ (time-floor
    # уже встроен внутрь M_l). Не надо подгонять его вручную.
    # При ВЫКЛЮЧЕННОЙ maturation (legacy) работает старый crutch: запись открывается
    # после pm_write_delay шагов ИЛИ по когерентности (mlp_mod std >= pm_coh_gate_std).
    # pm_write_delay<=0 тогда означает «только по когерентности» (без шагового пола).
    # ВНИМАНИЕ: слепое pm_write_delay=0 при СТАРОМ коде (до этого фикса) открывало
    # запись с шага 0 — именно это засевало эхо в best.pt. Теперь исправлено.
    pm_write_delay: int = 0      # legacy-пол только при maturation_enabled=False; 0 => coherence-only (maturity-only write gate)
    pm_coh_gate_std: float = 0.02  # legacy fallback when maturation is disabled

    # ─── Maturation gate (unified wake-up controller) ───
    # Replaces the ad-hoc pm_write_delay / pm_coh_gate_std / bridge-injection
    # scale=0 crutches with ONE principled per-layer maturity M_l(t) in [0,1]
    # that gates live BridgeGLU modulation, private-memory write, semantic
    # bridge injection and the intent bus. M_l = readiness_l * geometry_l:
    #   readiness_l : expert saturation via pred-error drop vs the random regime
    #   geometry_l  : deeper layers (larger tau in the VSA ladder) engage later
    # The frozen base MLP gate (~0.667) stays OPEN regardless (no deadlock).
    maturation_enabled: bool = True
    matur_alpha: float = 1.0        # geometry delay strength
    matur_T0: float = 8000.0        # gate starts opening at ~T0 steps (smooth ramp-in)
    matur_T_delay: float = 8000.0   # deepest layer opens at ~T0 + alpha*T_delay steps
    matur_delta: float = 4000.0     # geometry ramp width (steps)
    # NOTE: matur_T0 здесь — лишь Safety-Floor ВНУТРИ fused maturity
    # (= max(time_ramp, bridge_readiness)). Снижать его вручную НЕ нужно:
    # bridge_readiness открывает ветви рано, как только in-core bridge
    # научился предсказывать next-token (его косинус-лосс упал), а time_ramp
    # гарантирует открытие даже при «мёртвом» bridge. T0 не блокирует
    # раннее включение — оно управляется компетентностью, а не часами.
    matur_r0: float = 0.3           # readiness sigmoid center (lower = earlier opening)
    matur_rs: float = 0.2           # readiness sigmoid slope (lower = sharper transition)
    matur_ema: float = tau_api.period(1000)        # pred-error EMA decay (smoothness)
    matur_warm: int = 300           # warm steps: capture random-regime pred_err_init
    # matur_warmup_steps: REMOVED — no warmup, clean start from checkpoint
    matur_write_thr: float = 0.3    # maturity needed before private-memory writes
    # T8 role=permission: тот же класс, что mem_min_write_mat (выключатель, не амплитуда)
    # ─── Maturity: компетентностная добавка к time-рампе (E3-уточнение) ───
    # effective maturity = max(time_ramp, readiness_l), где readiness_l
    # считается в maturation.py из НАСЫЩЕНИЯ ЗАМЕРА pred-ошибки зеркала
    # (sat = 1 - EMA[pen]/pen_init -> сигмоида), а НЕ из bridge: код —
    # stack.py torch.maximum(mat_gate, maturation.readiness). bridge.readiness()
    # живёт отдельно и является optimizer-TRUST для группы bridge_glu (eva_optim).
    # Исторический комментарий здесь обещал «bridge_readiness» — документация
    # отставала от кода; исправлено 2026-09-13 по MATHEMATICAL_ANALYSIS E3.
    # T0 — запасной time-floor, не блок: компетентный слой открывается раньше.
    # r0/rs (0.3/0.2) поднимают потолок readiness до ~0.97.
    matur_bridge_readiness: bool = True
    matur_bridge_r0: float = 0.3    # центр сигмоиды готовности (доля падения лосса)
    matur_bridge_rs: float = 0.2    # наклон сигмоиды готовности

    # ─── Spec 1: Asymmetric expert init ───
    expert_asymmetry: bool = True  # break symmetry: different alpha, log_scale, W_proj per expert

    # ─── Spec 3: Recursive meta-trust ───
    meta_trust: bool = True  # track trust dynamics, penalize unstable experts (requires private_mem)

    collective_layer: bool = True
    collective_layer_idx: int = None
    collective_read_out: bool = True
    collective_S: int = 8
    collective_uncert_theta: float = 0.5
    collective_uncert_kappa: float = 3.0
    collective_contra_thresh: float = -0.1
    collective_contra_gain: float = 6.0
    collective_birth_gap: float = 0.55
    collective_maturity_thresh: float = 0.12

    log_scale_l2_weight: float = 0.01  # L2 on exp(log_scale) > 10 to prevent gradient explosion
    div_weight: float = 10.0   # sigmoid-bounded log_scale divergence. NOTE
                               # (M64.10/T13): in the align mode every aux term
                               # (including this one) goes through the spectral
                               # balancer — only `gradalign` is in BYPASS_AUX.
                               # The old 'bypasses spectral alignment' comment
                               # was wrong (the weight is on/off there).
    alpha_novelty_weight: float = 0.05  # push per-expert alpha apart (the LOSS
                                        # term; M64.6: the mirror push was removed —
                                        # it applied the same objective twice,
                                        # bypassing the balancer)
    gate_bias_scale: float = 2.0  # linspace init for gate bias per expert [-scale, scale]
    gate_bias_scale_per_layer: bool = True  # 0.5 (first layer) -> 2.0 (last layer)

    # Scheduler (values below will be overridden by λ_d when lambda_d_enabled=True)
    scheduler: str = 'mirror'
    target_var: float = 0.1
    mag_threshold: float = 0.3
    lr_min_ratio: float = 0.05
    max_decay_steps: int = 50000
    var_min_for_lr_decay: float = 0.005
    lr_improve_thresh: float = 0.98   # restore _loss_lr_factor to 1.0 when val < best*this (reachable)
    lr_regress_rel: float = 0.05      # damp LR only if val > best*(1+this) (5% — real divergence, not eval noise)
    lr_boost_max: float = 2.0         # upward-path ceiling: LR may climb above base up to this (0=disable boost)
    lr_improve_tol: float = 0.002     # val downtrend tolerance for the boost gate (hysteresis vs eval noise)

    # Per-layer LS-based LR modulation (индивидуальная адаптация по var(log_scale))
    per_layer_ls_lr: bool = False  # True = per-layer mult из fast/slow EMA var(ls)
    ls_ema_fast: float = tau_api.period(100)
    ls_ema_slow: float = tau_api.period(1000)
    ls_mult_min: float = 0.5
    ls_mult_max: float = 2.0
    ls_mirror_mult_max: float = 2.0  # кламп итога irm*ls_mult для mirror-градиентов

    # AdaptiveController (values below will be overridden by λ_d when lambda_d_enabled=True)
    exploration_threshold: float = 0.25
    differentiation_threshold: float = 0.08
    w_mem2v_scale_min: float = 0.5
    w_mem2v_scale_max: float = 1.0
    ema_alpha_min: float = 0.90
    ema_alpha_max: float = tau_api.period(100)
    noise_scale_min: float = 0.001
    noise_scale_max: float = 0.05
    delta_var_ema_min: float = 0.80
    delta_var_ema_max: float = tau_api.period(100)

    # Optimizer
    gate_lr_mult: float = 5.0
    lambda_lr_hierarchy: bool = True  # True = LR mult по степеням λ_d^p
    optimizer: str = "adamw"          # 'adamw' | 'eva' (EVAAdamW) | 'eva_proj' (EVAAdamW + AdamP-projection)

    # Optimizer hardening / progressive unfreeze (anti-collapse guard)
    llrd: float = 0.9              # DEPRECATED: index-based LR decay (replaced by tau_llrd_gamma)
    init_active_layers: int = 8    # blocks trainable from step 0 (rest frozen at init)
    stage_steps: int = 15000       # fixed backstop: unlock next block every N steps
    readiness_full: float = 0.6    # meta-maturity (differentiation) that unlocks deepest block
    stage_mode: str = 'readiness'  # 'readiness' (meta-driven) or 'fixed' (schedule only)

    # w_m2v hierarchy by τ (Proposal IV)
    w_m2v_hierarchy_target: float = 1.0  # m — max target for deep layers
    w_m2v_hierarchy_weight: float = 0.01  # λ_weight for w_m2v regularisation (drives _tau_l_dev adaptation)

    # Intent Bridge: τ-ladder regularization (targets are DETACHED to prevent co-adaptation)
    intent_tau_hierarchy_target: float = 0.3  # desired integration rate alpha for intent_state
    intent_tau_hierarchy_weight: float = 0.01  # λ_weight for intent-τ regularisation

    # Init stds
    w_d_init_std: float = 0.1
    conv_init_std: float = 0.01

    # Conv
    conv_kernel: int = 48

    # Spectral
    spec_lo: float = 0.5
    spec_hi: float = 1.5
    lambda_sliding: bool = True

    # Memory
    cov_multi_timescale: bool = True
    cov_tau_lo: int = 3
    cov_tau_hi: int = 200

    # Gate sparsity (auxiliary loss weight for expert specialization)
    gate_l1_weight: float = 0.0001   # L1 penalty on expert gates (0=disabled)
    # Expert reinforcement: align gate with usefulness prediction
    reinforce_weight: float = 0.001  # MSE(gate, usefulness) aux loss weight
    # Load balancing: encourages uniform expert usage across tokens
    balance_weight: float = 0.026  # λ⁻⁶ → HHI-based load balancing (adaptive)
    # Diversity loss: decorrelate per-group MLP outputs
    diversity_weight: float = 0.001  # ||cov - I||² weight (0=disabled)
    # Nuclear norm regularization for bind W_proj
    orth_weight: float = 0.0  # B2: was 1e-4=ON with 24× D² gram products (multi-GB) — the other dataclass had documented 0; unified off
    # Surprisal-weighted loss: focus on informative tokens
    surprisal_weight: float = 0.0  # γ, 0=disabled, 0.5=mild, 1.0=aggressive

    # Branch balance: equalize log-variance of conv/bind/mirror (Proposal V-3)
    branch_balance_weight: float = 0.0  # λ_B, 0=disabled

    # Gradient-reactive governance loss (prototype): open MLP gate where the MLP
    # output actually changes the CE loss. Aligns per-expert mlp_mod to
    # g_target = ||∂CE/∂mlp_out|| (detached). 0 = disabled (default).
    gradalign_weight: float = 0.0
    # M64.7 (M63-A): the readout/embedding LR multiplier override. 0 = the
    # historical λ⁻² damp (0.296 at λ_d=3; measured: the head's readout moved
    # -1.9% in 7315 steps while the head was the LM bottleneck). 1.0 = the
    # unfreeze A/B arm (readout at the base LR). The A/B is pre-registered in
    # docs/WHITEBOARD.md; the notebook does NOT enable it by default.
    readout_lr_mult: float = 0.0
    # M64.12 (M63-E §7): the aux kill-switch. When `aux_kill_switch` is on the
    # LossBalancer measures 2-3 aux terms per log interval (round-robin
    # grad-geometry vs CE) and prints the proj values; `aux_kill_disable` opts
    # into the actual disabling (a term whose CE projection stays below the
    # Schmitt threshold for `dwell` measurements is dropped from the aux dict).
    aux_kill_switch: bool = False
    aux_kill_disable: bool = False
    # M64.4 (M63-F): the LossBalancer align cadence. The align path costs THREE
    # graph traversals (CE/aux/bypass; ~3 recomputes with checkpointing) — the
    # largest structural cost of the run. k>1 = align every k-th step, the
    # cheap one-backward normalized total otherwise; 0 = never align.
    # Default 1 = every step (historical behaviour).
    balancer_align_every: int = 1
    # T9.6: safety-каналы (head_wall) не гейтятся CE-градиентом (предохранитель
    # не должен питаться от того сигнала, который защищает; при насыщении
    # |u|>17 CE-градиент≈0 и батчер глушил стену — замер 2026-09-19).
    # False = откат к гейтингу (A/B-рука) без правок кода.
    balancer_safety_ungated: bool = True
    # Cognitive MLP gate opening (fix for "MLP asleep"): init mlp_gate_b > 0 so the
    # mirror-gated MLP modulation (mlp_mod) actually scales the SwiGLU gate and
    # mod_scale_mlp receives a CE-gradient path (can open/close). 0 disables.
    mlp_gate_b_init: float = 0.25
    # Per-depth MLP gradient boost (fix for vanishing gradient to deep MLP):
    # scales MLP gradient by exp(mlp_depth_lr_exp * layer_idx). 0 = disabled.
    # Calibrated to the real (trained) gradient profile: ~13x collapse L0->mid
    # (not the 10k-20k x init-artifact). 0.10 -> L16 ~x4.5, L23 ~x10.
    mlp_depth_lr_exp: float = 0.10
    # Reopen value for the REAL cognitive gate `mod_scale_mlp` (MirrorMemory
    # param, sigmoid -> MLP scale). Checkpoints freeze it at init log(2)~0.69
    # (sigmoid 0.667 = "asleep"). On resume set it to log(3)~1.10 (sigmoid 0.75)
    # so the gate starts clearly open AND the deep-MLP gradient boost can move it.
    mlp_mod_scale_reopen: float = 1.0986  # math.log(3.0)
    # Hybrid gate tau: sigmoid+softmax temperature for the MLP modulation gate.
    # Replaces frozen mod_scale_mlp baseline. tau < 1 = sharper (winner-take-all),
    # tau > 1 = softer (uniform). 1.0 = balanced default.
    mlp_hybrid_gate_tau: float = 1.0
    # BridgeGLU: relocate SwiGLU gating tooling INTO the mirror/bridge. When True,
    # the per-expert MLP gate (mlp_mod) is produced by a GLU network over the
    # semantic delta instead of the frozen mod_scale_mlp parameter. The gate thus
    # becomes a live function of the bridge (semantic), not a stuck param. MLP
    # keeps its SwiGLU; BridgeGLU is the *outer* semantic gate. Experimental.
    bridge_glu: bool = True

    # bridge_glu_beta: modulation strength of the live BridgeGLU gate AROUND the
    # frozen stable baseline (sigmoid(mod_scale_mlp) ~ 0.667). BridgeGLU MODULATES
    # the baseline (mlp_mod = base * (1 + beta*(2*glu-1))); it does NOT replace it.
    # This keeps the run stable while the gate stays input/semantic-live. 0.25 = sane default.
    bridge_glu_beta: float = 0.25

    # bridge_conn: weight of the per-layer in-pipeline semantic bridge aux loss.
    # When > 0 a SemanticBridge runs inside the core forward (every layer emits a
    # semantic vector, predicts the NEXT token's embedding via cosine loss, and a
    # persistent cross-layer stream is injected back into each layer). This makes
    # the bridge part of the live pipeline in BOTH training and inference. 0.0 =
    # off. 0.1 is a sane default.
    bridge_conn: float = 0.1

    # bridge_dim: semantic vector width for the in-core SemanticBridge probe.
    bridge_dim: int = 256
    # bridge_depth: inject cross-layer (bottom-up + top-down) stream neighbours
    # back into each layer. If False the bridge still predicts next-token
    # embeddings but does not inject a spatial stream signal.
    bridge_depth: bool = True

    # bridge_hard_neg_k: hard-negative mining для контрастива моста (порт FCF,
    # T9). 0 = выключено (текущее поведение: полный пул негативов); >0 = CE по
    # [позитив + top-K самых похожих негативов] — концентрирует градиент на
    # различимых парах (лечит голодание: bridge_conn ≈ шанс весь прогон).
    # A/B: k=0 vs k=32 (pre-registered в docs/WHITEBOARD.md).
    bridge_hard_neg_k: int = 0

    # ─── T9 (порт EVA-Ai/FCP): Covariance Memory ───
    # Опциональная per-layer ветвь памяти второго порядка:
    # M_t = d_t·M_{t−1} + i_t·k_t k_tᵀ, чтение y = W_out(W_read(qᵀ M/√Dh)).
    # Дополняет VSA (первый момент): хранит парные корреляции k-пространства
    # (интерференция ниже, чем у векторной суперпозиции; не нулевая).
    # τ ветви = живой τ_l слоя (M7-refresh). W_out zero-init (rank>0: W_out_b)
    # ⇒ старт бит-в-бит residual (включать безопасно).
    # Дефолты под бюджет A100: Hd = heads×head_dim = 128, low-rank выход 128
    # ⇒ ~1.6M параметров/слой при D=2560 (не D²).
    # A/B: cov_memory False vs True (спека — docs/WHITEBOARD.md, T9).
    cov_memory: bool = False
    cov_memory_heads: int = 4        # головы; Hd = heads × head_dim (не D)
    cov_memory_head_dim: int = 32
    cov_memory_rank: int = 128       # ранг выхода (0 = полный D×D)
    cov_memory_chunk: int = 64       # размер чанка log-space скана

    # bridge_lr_mult: REMOVED — bridge uses base LR (the LBG routing was
    # removed in M64.5 as a dead channel; see docs/WHITEBOARD.md)

    # ─── Streaming Memory Bank (hierarchical L1+L2+L3) ───
    memory_bank: bool = False       # enable streaming memory bank
    mem_l1_slots: int = 3           # L1 rolling buffer slots (immediate)
    mem_l2_slots: int = 32          # L2 learned bank slots (short-term)
    mem_min_write_mat: float = 0.3  # min maturation before writes allowed (like private_mem)
    # T8 role=permission: бинарный выключатель записи (не amplitude-множитель;
    # не делать непрерывной функцией — T3/T7)
    mem_bridge_dim: int = 256       # memory bank bridge dim (matches bridge_dim)
    concept_birth_novelty_threshold: float = 0.15  # birth only if d_min > threshold (best_sim < 1-threshold)

    # ─── Unified Concept Layer (replaces per-block CollectiveConceptLayer + L3Concepts) ───
    unified_concept_layer: bool = True  # enable unified concept layer (global, after embedding)
    unified_concept_S: int = 8          # number of concept prototypes

    # ─── Logit Cache (dual-mode: training=h, inference=logits) ───
    logit_cache_enabled: bool = True     # decision #3: code-space cache, integrated in forward (zero-init gate: identity at start)
    logit_cache_max_entries: int = 64   # cached (B,L,D) windows; old cap counted 102400 full-window ENTRIES as tokens
    logit_cache_n_heads: int = 8         # attention heads for logit cache
    logit_cache_scheduled_sampling: float = 0.05  # R1: probability of inference-mode during training (0.05 = 5%)
    logit_cache_reset_on_resume: bool = True  # R6: clear cache on resume/LR-reset
    # T9.8 (оператор): кэш должен хранить ВЫХОД оператора, а не состояние.
    # Тренировочное чтение уже использует write-time K/V (audit #3), но K/V
    # были полноразмерными (2·D на токен = 504MB при 64 окнах); low-rank kv_dim
    # даёт то же для обеих сторон (трейн/инференс — одно пространство).
    # 0 = D (прежнее поведение, бит-совместимо); A/B-рука: 64 (×40 меньше).
    logit_cache_kv_dim: int = 0
    # T9.9 шаг 2: sentence-ring — второй уровень чтения кэша (пулы предложений
    # по SEP; окна = точный контент, ring = семантика). False = откат/A-B.
    logit_cache_sentence_ring: bool = True
    # T9.15: многоразрешающий кэш — пулы K/V на ФИКСИРОВАННЫХ шкалах τ
    # (сборка снизу вверх от атомов 8: 8 → 32 → 128 → 512 → …; старшая шкала =
    # среднее 4 младших пулов, набирается между шагами). Чтение — тем же
    # attention'ом, записи шкал помечены обучаемыми level-эмбеддингами, чтобы
    # модель сама выбирала разрешение. Пустая строка = выключено (A/B-рука),
    # формат "8,32,128,512". τ-шкалы становятся адресами памяти, а не только
    # временами забывания (см. журнал: «Многоразрешающий кэш на шкалах τ»).
    logit_cache_ms_spans: str = ""
    logit_cache_ms_max: int = 16    # кольцо записей на каждую шкалу
    # T9.5 (оператор 2026-09-19): кэш — двусторонний KV-аналог (тренировка:
    # копит последовательности и внимает; инференс: полное внимание).
    # ДИАГНОЗ УТОЧНЁН замером: σ(bias)=4.5e-5 — ложный индикатор (weight-терм
    # распределён по позициям; ФАКТ mean-гейт=0.043 уже на 250 шагах, |g| веса
    # гейта≈7 — кэш открывается сам). Рампа НЕ нужна; оставлена A/B-рукой
    # (0 = выкл, дефолт). Сила руки: при bias_final=−2 факт-гейт на рабочих
    # входах 0.13–0.5+ (зависит от входа) — сильное вмешательство.
    # Инвариант: расписание действует только в training (generate передаёт
    # step в eval — guard внутри set_gate_schedule).
    logit_cache_gate_ramp: int = 0         # 0 = выкл (A/B-рука: >0 = ramp)
    logit_cache_gate_bias_final: float = -2.0   # финал рампы (свободный параметр руки)
    cache_horizon_tokens: int = 0     # M34: 0 = AUTO = cfg.tau_max (the cache
                                      # spans exactly the slowest VSA scale);
                                      # negative = legacy blind FIFO; >0 = fixed
                                      # token horizon. Entries beyond it are
                                      # released by retention score
                                      # novelty x exp(-age/tau), not FIFO.

    # ─── Режим Б (открытое сознание): отказ от softmax-свёртки ───
    # Все точки комбинации смыслов используют нормированное сигмоид-среднее
    # (выпуклая комбинация, сумма весов = 1) вместо softmax-конкуренции.
    # Сохраняет лакуну (потенциал несовпавшего) и проецирует горизонт
    # событий, порождая новый концепт, без взрыва параметров. False = старый
    # softmax (закрытый режим, для A/B).
    softmax_free: bool = True

    # VSA long-range memory
    vsa_b_d_max: float = 12.0       # max b_d (τ≈160K at 12.0, was 5.0/τ≈150)
    vsa_b_d_smooth: float = tau_api.period(1000)   # per-step lerp rate towards controller target
                                    # 0.999 = 0.1%/step (τ_lerp≈1000 steps)
                                    # 1.0 = instant overwrite (old behavior)
    vsa_b_lr_mult: float = 0.1      # optimizer LR multiplier for b_d/b_i

    # ─── Unified τ-field (TauConfig) ───
    # All τ-dependent quantities derived from ONE set of parameters.
    tau_enabled: bool = True          # use unified tau_config (replaces _vsa_log_param + _tau_l_dev)
    tau_min: float = 8.0              # fastest layer (shallow)
    tau_max: float = 512.0            # slowest layer (deep)
    tau_dev_max: float = 0.3          # max deviation of log-space increments (cumsum → monotonic tau)
    tau_llrd_gamma: float = 0.65      # LLRD exponent: lr_l ∝ (tau_l / tau_ref)^(-gamma)  (γ=0.65 → ~5× spread)
    tau_mem_ref: float = 64.0         # reference τ for memory bank temperatures
    tau_dev_lr_mult: float = 0.2      # LR multiplier for _tau_dev (system-lever: conservative update)
    # (tau_gate_clamp removed B3: the clamp lives once in vsa_utils.DEV_CLAMP)
    gate_tau_min: float = 0.3         # min temperature for SpectrumGate (mature → precision)
    gate_tau_max: float = 5.0         # max temperature for SpectrumGate (immature → diversity)

    # ─── Qwen3-inspired upgrades ───
    bind_qk_norm: bool = True            # RMSNorm on hp before bottleneck cross (≈QK-Norm)
    rope_theta: float = 1000000.0        # RoPE base frequency (Qwen3: 1e6)
    rope_scaling: float = 1.0            # RoPE scaling factor (linear)
    mlp_swiglu: bool = True              # SwiGLU gate_proj parallel to up_proj (Qwen3-style)

    bind_twist_mode: str = "trajectory_spiral"
    bind_twist_S: int = 4
    bind_traj_dims: int = 3
    hybrid_alpha_max: float = 0.7
    hybrid_alpha_min: float = 0.3
    bind_twist_ocular: str = "tied"
    bind_twist_scheme: str = "golden"
    bind_twist_gate: bool = True

    # Trajectory manifold (FCF): beams + Zeckendorf decay on trajectory bind
    traj_manifold: bool = False        # clever wrap: TrajectoryManifoldBind instead of Spiral
    traj_beams: int = 0                # число лучей: 0 = авто ceil(sqrt(buffer))
    traj_buffer_size: int = 1024       # буфер переходов (Mini: 512; 1024→32 луча автоматом)
    traj_cos_threshold: float = 0.5    # cos-порог кластеризации лучей
    traj_rebuild_interval: int = 128   # пересборка лучей каждые N переходов
    traj_gain: float = 0.05            # масштаб вклада манифолда

    # ─── Intent Bridge (нисходяще-восходящая передача «намерения» экспертам) ───
    # Эксперты «подхватывают» восходящий сигнал (то, что идёт наверх и станет
    # логитом). Реализуется как обёртка: IntentProbe + zero-init w_intent/b_intent.
    intent_bridge: bool = True     # добавить мост ( checkpoint-совместимо: ноль-эффект при init)
    intent_topdown: bool = True    # зарезервировано: форма нисходящей трансляции intent_state

    # Gradient accumulation
    accum_steps: int = 1  # effective batch = batch_size * seq_len * accum_steps

    compile: bool = False
    gradient_checkpointing: bool = True  # trade compute for memory, essential on T4

    # Training
    max_steps: int = 500000
    log_interval: int = 100
    eval_interval: int = 440   # оператор: каждые 440 = 8×55 (выравнивание с логом)
    ucl_read_scale_floor: float = 0.0   # M59: floor on sigmoid(read_scale) until
                                        # `ucl_read_scale_floor_until` (0 = off);
    # T8 role=amplitude-floor: пол на АМПЛИТУДУ чтения (не permission; после релиза
    # модель закрывает чтение — лечится валютой/UCB, не ещё одним полом)
                                        # lets the UCL prove itself before the model
                                        # self-closes it (measured: -4.0 in 120 steps)
    ucl_read_scale_floor_until: int = 0  # M59: the step until which the floor holds
    stream_cap: float = 1e3        # M50: per-layer residual-stream magnitude
                                   # cap (scale-invariant fuse); 0 disables
    stream_chunk_steps: int = 0    # M62: rotate the genre stream every N steps
                                   # (0 = legacy: rotate only on stream exhaustion).
                                   # The novelty machinery (lacuna/phantom bank/UCL)
                                   # fires on DISTRIBUTION SHIFTS. M64.9r2: the notebook
                                   # uses 1000 (448k tokens/chunk — the M63-F stability
                                   # compromise; 250 = 112k tokens ~ 30 s was too short
                                   # for the CE) — see the F-vs-C conflict note in the
                                   # whiteboard (a 1000-step chunk shortens the phantom
                                   # slot survival across shifts to ~47%).
    branch_cap: float = 1e4        # M51: per-branch injection cap (conv/bind/
                                   # mirror/VPM/spectral/MLP); 0 disables
    branch_var_anchor: float = 0.5  # M51: absolute-scale anchor in the branch
                                    # loss (needs branch_balance_weight>0);
                                    # 0 disables
    eval_early_until: int = 3000   # M49: early measurement evals (no control
                                   # updates) until this step; 0 disables
    eval_early_every: int = 250    # M49: their cadence
    save_interval: int = 5000
    patience: int = 999999
    resume: str = ''

    # Paths
    data_dir: str = ''
    save_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    # ─── P4: прямая работа с состояниями (по заявке автора; все default off) ───
    inner_eye: bool = False       # P4-1: обучаемая добавка к гейту зеркала
                                  # (общий модуль на слои, zero-init ⇒ identity)
    meta_head: bool = False       # P4-2: зонд читаемости внутренних сигналов из h
    meta_head_grad: bool = False  # False = чистый зонд (h.detach, ствол не трогаем);
                                  # True = aux-канал «learning to introspect» (A/B)

    @classmethod
    def minimal(cls, **kw):
        """P2-1 (proposed patches): the bare trunk for the ablation.

        embed + block (conv + bind + VSA + spectral + MLP + mirror core) +
        coded head; every cognitive add-on off. For A/B science only — the
        attribution of the thesis ("code-state superposition > attention")
        needs the minimal arm and a same-budget baseline on the same stream.
        """
        cfg = cls(**kw)
        cfg.variable_precision = False
        cfg.explicit_reasoning = False
        cfg.triad_reason = False
        cfg.private_mem = False
        cfg.meta_trust = False
        cfg.collective_layer = False
        cfg.unified_concept_layer = False
        cfg.logit_cache_enabled = False
        cfg.memory_bank = False
        cfg.cov_memory = False
        cfg.bridge_conn = 0.0
        cfg.intent_bridge = False
        cfg.bridge_glu = False
        cfg.head_lacuna = False
        cfg.head_srl = False
        cfg.head_temper = False
        cfg.head_temper_rel = False
        cfg.phantom_chi_salience = False
        cfg.contradiction_field = False
        cfg.inner_eye = False
        cfg.meta_head = False
        cfg.head_pair_rank = 0
        cfg.maturation_enabled = False
        cfg.softmax_free = True
        cfg.bind_twist_mode = 'trajectory_spiral'   # the bind core stays
        cfg.lambda_d_enabled = False                # flat schedules, less coupling
        return cfg

    def __post_init__(self):

        # B8 (audit 02a F2A-04): nested-rope readout inverse was never
        # implemented — enabling embed_rope silently drops code roundtrip
        # to 0.20 (measured). Flag retired loudly rather than left as a trap.
        if getattr(self, 'embed_rope', False):
            raise NotImplementedError(
                'embed_rope=True retired (audit 02a F2A-04): readout rotation '
                'inverse is dead code; roundtrip collapses to 0.20.')
        # E10 (math-analysis audit): twin_free packing needs K=64 for the
        # full 65536-vocab with overlap <= S-2 (C(32,6) pools twin at that
        # size); failing late in codebook generation is cryptic.
        if getattr(self, 'codebook', 'legacy') == 'twin_free' and int(self.code_dim) < 64:
            raise ValueError(
                f"codebook='twin_free' requires code_dim >= 64 (got {self.code_dim}); "
                "use codebook='legacy' for smaller K.")
        if self.lambda_d_enabled:
            self._apply_lambda_d()

    def _apply_lambda_d(self):
        lc = LambdaConfig(self.lambda_d)
        self.warmup_steps = lc.warmup_steps
        self.target_var = lc.target_var
        self.mag_threshold = lc.mag_threshold
        self.lr_min_ratio = lc.lr_min_ratio
        self.max_decay_steps = lc.max_decay_steps
        self.var_min_for_lr_decay = lc.var_min_for_lr_decay
        self.exploration_threshold = lc.exploration_threshold
        self.differentiation_threshold = lc.differentiation_threshold
        self.w_mem2v_scale_min = lc.mem2v_scale_min
        self.w_mem2v_scale_max = lc.mem2v_scale_max
        self.ema_alpha_min = lc.ema_alpha_min
        self.ema_alpha_max = lc.ema_alpha_max
        self.noise_scale_min = lc.noise_scale_min
        self.noise_scale_max = lc.noise_scale_max
        self.delta_var_ema_min = lc.delta_var_ema_min
        self.delta_var_ema_max = lc.delta_var_ema_max
        self.gate_lr_mult = lc.gate_lr_mult
        self.log_scale_init_std = lc.log_scale_init_std
        self.conv_init_std = lc.conv_init_std
        self.w_d_init_std = lc.w_d_init_std
        self.log_interval = lc.log_interval
        self.eval_interval = lc.eval_interval
        self.save_interval = lc.save_interval
        self.patience = lc.patience


# EVA branding alias (keeps WideBindConfig as canonical for pickle compat)
EVAConfig = WideBindConfig
