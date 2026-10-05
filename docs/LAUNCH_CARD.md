# LAUNCH CARD — свежий прогон bounded_residual (Этап 3)

## Что запускаем
Свежий прогон EVA-CLM с нуля, `bounded_residual=True` (ограниченный резидуал:
инъекция банка `h + tanh(W_g·h_n+b_g)⊙unit(fused)` + RMSNorm потока), gc=False.
Старый чекпоинт — только диагностический референс, не резюмировать.

## Шаги
1. Colab: runtime A100 (40GB), открыть `notebooks/eva_colab.ipynb` из свежего клона main.
2. Прогнать ячейки 1-4 по порядку. В выводе cell4 проверить: `bounded_residual` есть,
   `gradient_checkpointing=False`, `use_amp=False` (bf16 включить ПОСЛЕ первого eval).
3. Перед стартом обучения зафиксировать память (ячейка с модельным билдом, ~0.7GB
   параметров; ожидаемый пик шага ~23-26GB, запас ~14GB до 40GB).
4. Запустить обучение (cell-цикл). Watchdog-логи смотреть начиная с шага ~100.

## Watchdog (красные флаги = остановка и разбор)
- `b_flow` (norm потока): ожидание O(1)-O(10); рост на порядок+ -> стоп.
- `b_drift` (дрейф против EMA): автоматический RED FLAG >10x в логе/analyze -> стоп.
- train/eval CE: должны падать синхронно; растущий train-eval разрыв как у патологии
  (eval растёт при падающем train) -> стоп.
- dctx_own/dctx_mix (KPI): оценивать на каждом eval; dctx_mix должен уходить в плюс.
- Периодически (на eval): `python scripts/analyze.py` по последнему чекпоинту —
  ветки видимы (knockout bind ΔCE > шума), head/bind не 1e4+, часы/таблицы без NaN.

## Fallback
- OOM на старте: вернуть `gradient_checkpointing=True` (потеря ~1.66x, не блокер);
  bounded-путь под gc=True проверен (OFF бит-идентичен, ON конечен).
- OOM на eval: `eval_windows_budget` уже 64; recovery seq-shrink в cell9 вооружён.

## Бюджет
Один цикл. Продление — только если watchdog чист на 3-5k шагах и dctx растёт.

## Артефакты
- Летопись: `docs/ARCHITECTURE_JOURNAL.md` (решения, все замеры, очередь).
- Стенд: `scripts/bench_calibrated.py` (воспроизведение патологии/проверка фиксов).
- Тесты: `python -m pytest tests -q` -> 704 passed.
- HANDOFF: `docs/HANDOFF.md`; внешний аудит: `EVA_CLM_AUDIT_VERIFICATION.md` (rev.4).
