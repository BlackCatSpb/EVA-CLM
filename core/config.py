"""WideBind configuration with λ_d hierarchy support."""

from dataclasses import dataclass, field
from .lambda_utils import LambdaConfig

_LAMBDA_OVERRIDE_DOC = (
    "Set to None to use λ_d-derived value (recommended for Experiment 1)."
)


@dataclass
class WideBindConfig:
    D: int = 4096
    n_layers: int = 32
    bind_K: int = 64
    vocab: int = 50000
    seq_len: int = 128
    batch_size: int = 2
    lr: float = 3e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    grad_clip: float = 0.5
    dtype: str = 'float32'

    # False = обучать EOS-токен (границы предложений), True = маскировать (старое поведение)
    mask_eos: bool = False   # Colab: учим EOS (данные *_eos.bin — границы предложений)

    # ─── λ_d hierarchy ─────────────────────────────────────────
    lambda_d: int = 3            # dimension of generalized golden ratio
    lambda_d_enabled: bool = True  # True = apply λ_d derivation in __post_init__

    tie_bind: bool = True  # True = W_out = W_proj^T (autoencoder bind bottleneck)
    tie_mirror_proj: bool = True  # True = mirror W_out = W_proj^T (per-expert K-space AE)

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
    code_dim: int = 32
    code_sparsity: int = 6

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
    matur_ema: float = 0.999        # pred-error EMA decay (smoothness)
    matur_warm: int = 300           # warm steps: capture random-regime pred_err_init
    # matur_warmup_steps: REMOVED — no warmup, clean start from checkpoint
    matur_write_thr: float = 0.3    # maturity needed before private-memory writes
    # ─── Maturity готовность по компетентности bridge (вместо слепой time-рампы) ───
    # effective maturity = max(time_ramp, bridge_readiness). Ветви (bridge-инъекция,
    # live-модуляция, private-memory write, intent-шина) открываются, как только
    # in-core SemanticBridge научился предсказывать next-token embedding (его
    # косинус-лосс упал), а не по фиксированным часам T0. Bridge учится
    # независимо от LM-лосса ствола => готовность НЕ зацикливается (в отличие от
    # pred_err зеркала). У init bridge случаен => readiness=0 => ствол не
    # возмущается => стабильность обучения сохранена.
    # T0/T_delay снижены (8000/8000) для ускорения открытия рампа после прорыва;
    # r0/rs снижены (0.3/0.2) для поднятия потолка readiness с ~0.76 до ~0.97.
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
    orth_weight: float = 0.0  # ortho-gran loss; 0=off (32x D²=4096² gram graphs cost 2GB+ VRAM)
    div_weight: float = 10.0   # sigmoid-bounded log_scale divergence (bypasses spectral alignment)
    gate_repulse_weight: float = 0.3  # push gate variance up (inverse of balance, bypasses spectral)
    alpha_novelty_weight: float = 0.05  # push per-expert alpha apart (heuristic, no spectral)
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
    ls_ema_fast: float = 0.99
    ls_ema_slow: float = 0.999
    ls_mult_min: float = 0.5
    ls_mult_max: float = 2.0
    ls_mirror_mult_max: float = 2.0  # кламп итога irm*ls_mult для mirror-градиентов

    # AdaptiveController (values below will be overridden by λ_d when lambda_d_enabled=True)
    exploration_threshold: float = 0.25
    differentiation_threshold: float = 0.08
    w_mem2v_scale_min: float = 0.5
    w_mem2v_scale_max: float = 1.0
    ema_alpha_min: float = 0.90
    ema_alpha_max: float = 0.99
    noise_scale_min: float = 0.001
    noise_scale_max: float = 0.05
    delta_var_ema_min: float = 0.80
    delta_var_ema_max: float = 0.99

    # Optimizer
    gate_lr_mult: float = 5.0
    lambda_lr_hierarchy: bool = True  # True = LR mult по степеням λ_d^p

    # Optimizer hardening / progressive unfreeze (anti-collapse guard)
    llrd: float = 0.9              # DEPRECATED: index-based LR decay (replaced by tau_llrd_gamma)
    init_active_layers: int = 8    # blocks trainable from step 0 (rest frozen at init)
    stage_steps: int = 15000       # fixed backstop: unlock next block every N steps
    readiness_full: float = 0.6    # meta-maturity (differentiation) that unlocks deepest block
    stage_mode: str = 'readiness'  # 'readiness' (meta-driven) or 'fixed' (schedule only)
    watchdog_ce: float = 15.0      # CE above this => rollback to best.pt + fresh Adam
    recover_lr_mult: float = 0.5   # LR multiplier applied on each recovery
    recover_max: int = 20          # abort after this many recoveries

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
    nuclear_weight: float = 1e-5  # stochastic ||W||_* weight (0=disabled)
    orth_weight: float = 1e-4  # ||Ŵ^TŴ - I||² weight (0=disabled)
    # Surprisal-weighted loss: focus on informative tokens
    surprisal_weight: float = 0.0  # γ, 0=disabled, 0.5=mild, 1.0=aggressive

    # Branch balance: equalize log-variance of conv/bind/mirror (Proposal V-3)
    branch_balance_weight: float = 0.0  # λ_B, 0=disabled

    # Gradient-reactive governance loss (prototype): open MLP gate where the MLP
    # output actually changes the CE loss. Aligns per-expert mlp_mod to
    # g_target = ||∂CE/∂mlp_out|| (detached). 0 = disabled (default).
    gradalign_weight: float = 0.0
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

    # bridge_lr_mult: REMOVED — bridge uses base LR with LayerBridgeGate routing

    # ─── Streaming Memory Bank (hierarchical L1+L2+L3) ───
    memory_bank: bool = False       # enable streaming memory bank
    mem_l1_slots: int = 3           # L1 rolling buffer slots (immediate)
    mem_l2_slots: int = 32          # L2 learned bank slots (short-term)
    mem_l3_concepts: int = 8        # L3 emergent concept slots (long-range)
    mem_l3_birth_threshold: float = 0.85  # cosine sim threshold for concept birth
    mem_min_write_mat: float = 0.3  # min maturation before writes allowed (like private_mem)
    mem_bridge_dim: int = 256       # memory bank bridge dim (matches bridge_dim)
    concept_birth_novelty_threshold: float = 0.15  # birth only if d_min > threshold (best_sim < 1-threshold)

    # ─── Unified Concept Layer (replaces per-block CollectiveConceptLayer + L3Concepts) ───
    unified_concept_layer: bool = True  # enable unified concept layer (global, after embedding)
    unified_concept_S: int = 8          # number of concept prototypes

    # ─── Режим Б (открытое сознание): отказ от softmax-свёртки ───
    # Все точки комбинации смыслов используют нормированное сигмоид-среднее
    # (выпуклая комбинация, сумма весов = 1) вместо softmax-конкуренции.
    # Сохраняет лакуну (потенциал несовпавшего) и проецирует горизонт
    # событий, порождая новый концепт, без взрыва параметров. False = старый
    # softmax (закрытый режим, для A/B).
    softmax_free: bool = True

    # VSA long-range memory
    vsa_b_d_max: float = 12.0       # max b_d (τ≈160K at 12.0, was 5.0/τ≈150)
    vsa_b_d_smooth: float = 0.999   # per-step lerp rate towards controller target
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
    tau_gate_clamp: float = 2.0       # unified clamp for adaptive tau deviation in gates
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
    eval_interval: int = 1000
    save_interval: int = 5000
    patience: int = 999999
    resume: str = ''

    # Paths
    data_dir: str = ''
    save_dir: str = 'checkpoints'
    log_dir: str = 'logs'

    def __post_init__(self):
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
