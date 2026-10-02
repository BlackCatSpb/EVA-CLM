"""Smoke: все scripts/*.py компилируются и импортируются (M65-opt).

Аудит: probe_*/sim/bench/smart_* не имели ни одного теста — сломанный импорт
(например, мёртвый test_window_accuracy.py импортировал несуществующий
core.logit_cache_v2) обнаруживался только при ручном запуске. Здесь:
компиляция ВСЕХ скриптов + импорт каждого (module-level код обязан быть
безопасным; probes с файлами на уровне модуля обёрнуты в main()).
"""
import glob
import importlib.util
import os
import py_compile
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = sorted(glob.glob(os.path.join(ROOT, 'scripts', '*.py')))


def test_all_scripts_compile():
    for f in SCRIPTS:
        py_compile.compile(f, doraise=True)


def test_all_scripts_import_clean():
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    failures = []
    for f in SCRIPTS:
        name = 'script_' + os.path.basename(f)[:-3]
        try:
            spec = importlib.util.spec_from_file_location(name, f)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            failures.append(f'{os.path.basename(f)}: {type(e).__name__}: {e}')
    assert not failures, 'скрипты не импортируются:\n' + '\n'.join(failures)
