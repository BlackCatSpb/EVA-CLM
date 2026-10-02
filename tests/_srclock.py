"""Утилиты source-локов, устойчивые к форматированию (M65-opt).

Аудит: 12 тестов читали исходники как СЫРОЙ ТЕКСТ и ломались от любого
переформатирования (реальный инцидент: рефакторинг decode-блока уронил
test_t9_generate_parity). Здесь:
  - `norm(path)` — исходник без комментариев и сжатыми пробелами;
  - `calls(path, func)` — число вызовов функции/метода по AST (устойчиво к
    переносам строк, комментариям и переименованию переменных).
Полная замена source-локов на поведенческие — в очереди батча 6 (регистр).
"""
import ast
import re


def norm(path: str) -> str:
    src = open(path, encoding='utf-8').read()
    src = re.sub(r'#.*', '', src)
    return re.sub(r'\s+', ' ', src)


def calls(path: str, func: str) -> int:
    tree = ast.parse(open(path, encoding='utf-8').read())
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == func:
                n += 1
            elif isinstance(f, ast.Attribute) and f.attr == func:
                n += 1
    return n
