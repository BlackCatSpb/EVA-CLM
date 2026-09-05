# WideBind — Architecture Characterization & Comparative Survey

> Audience: external engineers/reviewers who already know modern sequence models
> (Transformers, SSMs, memory-augmented and adaptive-compute networks).
> Goal: give an accurate, mechanism-level picture of WideBind and map each of its
> ideas to the closest known work, so differences are easy to spot.

---

## 1. Design philosophy

WideBind is a **cognitive-inspired, VSA-centric language model** rather than a pure
attention or state-space model. Three commitments shape every component:

1. **Vector Symbolic Architecture (VSA / Holographic Reduced Representations) as the
   representational substrate** — bindings and memories are built by circular
   convolution (HRR) and vector superposition, not by attention weight matrices alone.
2. **"Unbounded context via an Intent Bus"** — instead of attending over an ever-growing
   history, a small *cross-layer intent signal* is streamed through the depth and used
   to bias both expert gates and the output head. Context is *compressed and routed*,
   not *stored and re-scanned*.
3. **Fully adaptive training harness with no magic numbers** — learning rate, active
   depth, and auxiliary-loss weighting are all driven by principled online controllers
   (variance/EMA ratios, 3σ failure detection, spectral gradient alignment), not fixed
   schedules or hand-tuned thresholds.

The model is early-stage research code: it is **not** benchmarked on standard LM eval
suites yet, and its training dynamics are intentionally unusual (see §6).

---

## 2. Component inventory (grounded in `core/`)

| Subsystem | Module | What it does |
|---|---|---|
| Token embed | `ZeckendorfEmbedding` / `PartitionedEmbedding` + `RotaryEmbedding` | Fibonacci/Zeckendorf-coded token ids + rotary position |
| **Block** | `WideBindBlock` | Pre-LN → VSA bind → memory → conv → spectral → MLP |
| VSA bind | `TrajectorySpiralBind` (hybrid HRR) | `D→K` projection, **hybrid circular-convolution + elementwise** bind, frequency-modulated by a "trajectory spiral" of `hp` phases |
| Sequence memory | `ExactSequenceMemory` | **scaled-dot-product softmax attention over the local sequence** (`q,k,v` linear → softmax) — i.e. WideBind *does* contain a local attention per block |
| Conv | depthwise 48-tap | local temporal mixing |
| Spectral | DCT basis scaling | frequency-domain feature shaping |
| MLP | `GroupedMLP` | grouped experts; **gate opened by gradient alignment** with the CE loss (`gradalign`) |
| **Mirror** | `GroupedCognitiveMirror` | per-group K-space factorization; emits *slow metacognitive signals* (temporal, global, predictive, symmetry, help); **learns a per-K-dimension time-constant** (`alpha_diag` via residual-variance EMA); private memory read on uncertainty; contradiction gate |
| **Intent Bridge** | in `stack.py` | `intent_probe` → per-layer `intent_stream` (detached) → `Bus_i = mean(past fresh + future carried)` → modulates mirror gate via `w_intent` + salience; **phase-2 stencil** `bus_head_proj` biases the head readout |
| Collective | `CollectiveConceptLayer` | uncertainty (`resvar`), contrastive disagreement, maturity gating, read-out |
| Head | `SigmoidCodedHead` | **codebook of sparse block-codes**; independent **sigmoid log-odds** (not softmax); temperature + bit-bias + token-bias + bridge `bus_bias` |
| Reasoning | `reasoning.py` | adaptive-depth explicit reasoning loop gated by knowledge signals |
| Controllers | `adaptation.py` | `DepthController` (progressive unfreeze), `LRController` (mirror-adaptive LR **with an upward boost path gated on validation downtrend**), `FailureDetector` (3σ), `GradientClipper` (AGC), `LossBalancer` (spectral aux) |

### 2.1 The Intent Bridge (the most distinctive part)
- A per-layer `intent_probe` projects hidden states into a compact intent space; its
  sequence-mean becomes the *fresh* intent for that layer.
- A persistent, **detached** `intent_stream` carries *carried* intent forward; the
  per-layer **Bus** averages all past-fresh and future-carried intents.
- The bus modulates the mirror's openness gate: `gate = einsum(hp − intent, w_intent) +
  b_intent + salience·w_sal`. `w_intent/b_intent/w_sal` are **zero-init** (checkpoint-safe).
- A **salience** signal (from the head's own logits) multiplicatively gates the probe,
  one step delayed — word-importance feeds back into what gets broadcast.
- **Phase-2 stencil**: `bus_head_proj` (zero-init) projects the bus into the head's
  code space and is added to the head logits — the bridge directly shapes output.

This is *not* a residual stream and *not* cross-layer attention: it is a learnable,
low-rank, salience-weighted **gist bus** that is written by probes and read by both
expert gates and the head.

---

## 3. Comparative survey

Legend: SM = sequence mixer, MEM = long-range memory, HEAD = output layer,
ADAPT = recurrence/adaptive compute.

| Project | SM (core) | MEM | HEAD | ADAPT | Relation to WideBind |
|---|---|---|---|---|---|
| **Transformer** (Vaswani'17) | full softmax attention | none (context = window) | softmax | fixed depth | WideBind reuses local attention (`ExactSequenceMemory`) but adds VSA bind, conv, spectral, mirror, bridge on top |
| **Transformer-XL** (Dai'19) | segmented attention | segment-level recurrence cache | softmax | — | Similar "carry context across depth" goal, but WB uses a compressed *intent* bus, not raw hidden caches |
| **Compressive / Infini-Attention** (Rae'19 / Munkhdalai'24) | attention | compressive/attention memory | softmax | — | Same problem class (long context); WB's memory is VSA superposition + intent bus, not compressed KV |
| **RetNet** (Sun'23) | retention (multi-scale decay) | implicit (decay state) | softmax | recurrent inference | RetNet's decayed state ≈ WB's `alpha_diag` time-constants, but WB learns per-dimension τ via residual variance |
| **Mamba / SSM** (Gu'23) | selective state space | linear recurrent state | softmax | — | Both avoid quadratic attention; WB keeps an *attention block* but adds VSA + metacognition. SSM state is denser; WB's bus is sparse/low-rank |
| **RWKV** (Peng'23) | linear-attention (WKV) | linear recurrence | softmax | — | Same "O(n) recurrence" family; WB differs by VSA binding and intent routing |
| **Titans** (Behrouz'24) | attention + MLP *memory* | surprise-gated MLP memory (test-time learning) | softmax | test-time learning | Closest in spirit (learned, surprise-driven long-term memory). WB's memory is VSA superposition + mirror signals rather than an MLP memory; WB adapts at *training* time, not test time |
| **Product-Key Memory / PKM** (Lample'19) | attention | key-value memory | PKM head | — | `SigmoidCodedHead` is a codebook head like PKM, but uses **independent sigmoid log-odds + base term + mirror/bridge bias** instead of a single softmax over keys |
| **Mixture-of-Experts** (Shazeer'17, GShard) | attention | — | softmax | expert routing | `GroupedMLP` is MoE-like, but experts are **opened by gradient alignment**, not a router; no token-dropping |
| **PonderNet / Adaptive Compute** (Goyal'21) | attention | — | — | adaptive depth/steps | `DepthController` (progressive unfreeze) + explicit reasoning loop are the same *family* of "compute where needed" ideas |
| **Fast Weight Networks / Hebbian** (Schmidhuber'92, Munkhdalai'18) | attention | Hebbian fast weights | softmax | test-time | WB's mirror `alpha_diag`/private-memory readout is a mild, slow form of fast-weighting, but not the core mechanism |

---

## 4. What is genuinely distinctive

1. **VSA superposition memory + hybrid HRR/elementwise bind** as a first-class layer
   primitive (`TrajectorySpiralBind`), frequency-modulated by hidden-state "trajectories".
2. **The Intent Bus** — a detached, salience-gated, cross-layer gist stream that
   modulates *both* expert gates and the output head. This is the central novelty and
   has no direct counterpart in the table above.
3. **Cognitive mirror signals** — a structured set of *slow* auxiliary signals
   (temporal deviation, global deviation, predictive error, symmetry, helpfulness) that
   shape gates, plus a **learned per-dimension time-constant** (`alpha_diag`). This is a
   metacognitive layer unusual in mainstream LLMs.
4. **Sigmoid-coded codebook head** with a background `base` term and bridge/head-stencil
   bias — a non-softmax, bitwise-independent output distribution that the bridge can steer.
5. **Principled, magic-number-free adaptation** — LR can *boost above base* when
   validation trends down (negative-feedback self-limiting), damp on divergence, and
   depth/progressive-unfreeze + spectral aux are all ratio-driven.

---

## 5. Honest caveats (read before judging)

- **Early research stage.** No standard LM benchmark numbers yet; all evidence is from a
  single ~143M-param from-scratch run on a small corpus.
- **Unusual training dynamics by design.** CE routinely spikes (e.g. 10→32) during
  subsystem co-adaptation; this is expected, not a bug, per the monitoring policy.
- **Phase-1 of the bridge is empirically not yet fully awake** at the checkpoint analyzed
  (`w_intent` active only at the bottom layer L0; deeper layers still 0). Phase-2 (head
  stencil `bus_head_proj`) is clearly active and growing. This is consistent with the
  intended staged wake-up (bridge → mirrors → meta-core).
- **Hybrid, not minimal.** WideBind stacks many mechanisms per block; compute/parameter
  efficiency vs a clean Transformer/SSM baseline is an open empirical question.

---

## 6. One-paragraph summary for a busy reviewer

WideBind is a 24-layer, ~143M-param cognitive-inspired LM that keeps a local
softmax-attention block per layer but surrounds it with VSA binding (`TrajectorySpiralBind`),
vector-superposition memory, depthwise conv, spectral shaping, and gradient-aligned grouped
experts. Its defining idea is an **Intent Bridge**: a detached, salience-gated cross-layer
"gist bus" written by per-layer intent probes and read by both expert gates (`w_intent`)
and the output head (via a zero-init stencil). A "cognitive mirror" layer adds slow
metacognitive signals and a learned per-dimension time-constant, and the whole system is
trained under principled online controllers (mirror-adaptive LR with an upward boost path,
progressive depth unfreeze, 3σ failure rollback, spectral aux balancing). Relative to
Transformers it adds VSA+mirror+bridge; relative to SSMs/RWKV it keeps attention and adds
metacognition; relative to Titans/Infini-Attention it pursues *compressed intent routing*
rather than stored KV or MLP memory. It is early-stage and not yet benchmarked.
