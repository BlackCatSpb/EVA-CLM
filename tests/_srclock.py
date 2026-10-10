"""Утилиты source-локов, устойчивые к форматированию (M65-opt).

Аудит: тесты читали исходники как СЫРОЙ ТЕКСТ и ломались от любого
переформатирования (реальный инцидент: рефакторинг decode-блока уронил
test_t9_generate_parity). Здесь:
  - `norm(path)` — исходник без комментариев (tokenize; `#` внутри строковых
    литералов НЕ вырезается) и со сжатыми пробелами;
  - `calls(path, func)` — число вызовов функции/метода по AST (устойчиво к
    переносам строк, комментариям и переименованию переменных), с учётом
    алиасов (`b = f; b(...)`) и без учёта заведомо мёртвых веток;
  - `has_call`/`assigns`/`has_name`/`top_stmts`/`call_sites` — точечные
    AST-проверки для локов (сравнение аргументов через `ast.unparse`, т.е.
    эквивалентность узла, а не наличие имени в тексте).

ОСТАТОЧНЫЕ ОГРАНИЧЕНИЯ `calls` (базовый проход, сознательно не идеальный):
  * алиасы отслеживаются только для простых `x = f`/`x = mod.f`
    (транзитивно) и `from mod import f as g`; НЕ отслеживаются
    `self.f = f`, `f2 = functools.partial(f)`, распаковки, callable-объекты,
    вызовы через `getattr` и параметры-функции;
  * алиас, переопределённый позже другим значением, даёт переучёт (мы
    считаем по объединению всех связываний);
  * мёртвыми считаются только ветки с константно-ложным `if`/`while`
    (включая `if False:`/`while 0:`), a также код после безусловного
    `return`/`raise` в том же блоке; динамическая недостижимость,
    dead-store/сопроцедуры не анализируются;
  * вызовы внутри `try`/`except`, comprehensions и вложенных def считаются
    (они достижимы/реальны).
Полная замена source-локов на поведенческие — очередь батча 6 (регистр).
"""
import ast
import io
import os
import tokenize
from typing import Any, Dict, Iterable, List, Optional, Sequence

__all__ = [
    'read', 'parse', 'parse_src', 'strip_comments', 'norm', 'norm_src',
    'calls', 'calls_src', 'call_sites', 'call_sites_src', 'call_sites_in',
    'has_call', 'has_call_in', 'assigns', 'has_name', 'identifiers',
    'has_augassign', 'has_if', 'has_dict_entry', 'has_compare', 'has_str',
    'str_values', 'top_stmts', 'find_def',
    'unparse', 'const_bool',
]


def read(path: str) -> str:
    with open(path, encoding='utf-8', errors='replace') as f:
        return f.read()


def _as_src(obj: Any) -> str:
    """Строка-путь (существующий файл без переводов строк) или исходник."""
    if isinstance(obj, ast.AST):
        return ''
    if isinstance(obj, str) and '\n' not in obj and os.path.exists(obj):
        return read(obj)
    return obj


def parse(path: str) -> ast.Module:
    return ast.parse(read(path))


def parse_src(src: str) -> ast.Module:
    return ast.parse(src)


def strip_comments(src: str) -> str:
    """Удалить ТОЛЬКО комментарии, не трогая `#` в строковых литералах.

    Комментарии заменяются пробелами (позиции строк сохраняются), поэтому
    переводы строк и отступы остаются валидными. При ошибке токенизации
    (неполный фрагмент) возвращает исходник без изменений — вызывающий код
    всё равно должен быть парсабельным для AST-проверок.
    """
    if '#' not in src:
        return src
    lines = src.splitlines(keepends=True)
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return src
    # по строкам: вырезаем спаны комментариев (1-based)
    base = [0]
    for ln in lines:
        base.append(base[-1] + len(ln))
    out = list(src)
    for tok in toks:
        if tok.type != tokenize.COMMENT:
            continue
        (sr, sc), (er, ec) = tok.start, tok.end
        if sr == er:
            i = base[sr - 1] + sc
            j = base[er - 1] + ec
            out[i:j] = ' ' * (j - i)
    return ''.join(out)


def norm_src(src: str) -> str:
    return ' '.join(strip_comments(src).split())


def norm(path: str) -> str:
    """Исходник без комментариев (tokenize) и сжатыми пробелами."""
    return norm_src(read(path))


# ───────────────────────── константная достижимость ─────────────────────────

_UNK = object()


def _const_value(node: ast.AST):
    if isinstance(node, ast.Constant):
        return node.value
    return _UNK


def const_bool(node: ast.AST) -> Optional[bool]:
    """Константная булева свёртка теста (None = неизвестно)."""
    if isinstance(node, ast.Constant):
        try:
            return bool(node.value)
        except Exception:
            return None
    if isinstance(node, ast.Name):
        if node.id == 'True':
            return True
        if node.id in ('False', 'None'):
            return False
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        v = const_bool(node.operand)
        return None if v is None else (not v)
    if isinstance(node, ast.BoolOp):
        vals = [const_bool(v) for v in node.values]
        if isinstance(node.op, ast.And):
            if any(v is False for v in vals):
                return False
            return True if all(v is True for v in vals) else None
        if isinstance(node.op, ast.Or):
            if any(v is True for v in vals):
                return True
            return False if all(v is False for v in vals) else None
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        left = _const_value(node.left)
        right = _const_value(node.comparators[0])
        if left is _UNK or right is _UNK:
            return None
        op = node.ops[0]
        try:
            if isinstance(op, ast.Eq):
                return bool(left == right)
            if isinstance(op, ast.NotEq):
                return bool(left != right)
            if isinstance(op, ast.Is):
                return left is right
            if isinstance(op, ast.IsNot):
                return left is not right
            if isinstance(op, ast.Lt):
                return bool(left < right)
            if isinstance(op, ast.LtE):
                return bool(left <= right)
            if isinstance(op, ast.Gt):
                return bool(left > right)
            if isinstance(op, ast.GtE):
                return bool(left >= right)
        except Exception:
            return None
    return None


# ───────────────────────────── вызовы по AST ────────────────────────────────

def _dotted(node: ast.AST) -> Optional[str]:
    """Полное точечное имя узла (Name/Attribute), иначе None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f'{base}.{node.attr}' if base else node.attr
    return None


def _collect_aliases(tree: ast.AST, func: str) -> set:
    """Имена/цепочки, ссылающиеся на `func` (простые связывания + import-as)."""
    aliases: set = set()
    changed = True
    while changed:
        changed = False

        def _points_to(value: ast.AST) -> bool:
            if isinstance(value, ast.Name):
                return value.id == func or value.id in aliases
            if isinstance(value, ast.Attribute):
                return value.attr == func
            return False

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for a in node.names:
                    if a.name == func:
                        nm = a.asname or a.name
                        if nm not in aliases:
                            aliases.add(nm)
                            changed = True
            elif isinstance(node, ast.Assign):
                if _points_to(node.value):
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id not in aliases:
                            aliases.add(t.id)
                            changed = True
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                if isinstance(node.target, ast.Name) and _points_to(node.value):
                    if node.target.id not in aliases:
                        aliases.add(node.target.id)
                        changed = True
    return aliases


class _CallFinder(ast.NodeVisitor):
    """Сбор вызовов target'а по достижимому коду (см. ограничения в хедере)."""

    def __init__(self, target: str):
        self.target = target
        self.aliases: set = set()
        self.sites: List[ast.Call] = []

    def _match(self, func: ast.AST) -> bool:
        if '.' in self.target:
            return _dotted(func) == self.target
        if isinstance(func, ast.Name):
            return func.id == self.target or func.id in self.aliases
        if isinstance(func, ast.Attribute):
            return func.attr == self.target
        return False

    def visit_Call(self, node: ast.Call) -> None:
        if self._match(node.func):
            self.sites.append(node)
        for a in node.args:
            self.visit(a)
        for kw in node.keywords:
            self.visit(kw.value)

    def _visit_body(self, stmts: Sequence[ast.AST]) -> None:
        for s in stmts:
            self.visit(s)
            # код после безусловного return/raise в том же блоке недостижим
            if isinstance(s, (ast.Return, ast.Raise)):
                break

    def visit_Module(self, node: ast.Module) -> None:
        self._visit_body(node.body)

    def visit_FunctionDef(self, node) -> None:
        self._visit_body(node.body)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node) -> None:
        self._visit_body(node.body)

    def visit_If(self, node: ast.If) -> None:
        t = const_bool(node.test)
        if t is False:
            self._visit_body(node.orelse)
        elif t is True:
            self._visit_body(node.body)
        else:
            self._visit_body(node.body)
            self._visit_body(node.orelse)

    def visit_While(self, node: ast.While) -> None:
        t = const_bool(node.test)
        if t is False:
            self._visit_body(node.orelse)
            return
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        t = const_bool(node.test)
        if t is False:
            self.visit(node.orelse)
        elif t is True:
            self.visit(node.body)
        else:
            self.visit(node.body)
            self.visit(node.orelse)


def call_sites_in(node: ast.AST, func: str) -> List[ast.Call]:
    """Вызовы `func` внутри готового AST-узла (напр. FunctionDef)."""
    f = _CallFinder(func)
    f.aliases = _collect_aliases(node, func)
    f.visit(node)
    return f.sites


def call_sites_src(src: str, func: str) -> List[ast.Call]:
    """Вызовы `func` в исходнике (алиасы учтены, мёртвые ветки пропущены)."""
    return call_sites_in(ast.parse(src), func)


def call_sites(obj: Any, func: str) -> List[ast.Call]:
    return call_sites_src(_as_src(obj), func)


def calls(path: str, func: str) -> int:
    return len(call_sites(path, func))


def calls_src(src: str, func: str) -> int:
    return len(call_sites_src(src, func))


def unparse(node: ast.AST) -> str:
    return ast.unparse(node)


def has_call(obj: Any, func: str, args: Optional[Iterable[str]] = None,
             kwargs: Optional[Dict[str, str]] = None) -> bool:
    """Есть ли вызов `func` с указанными (частично) позиционными/kw-аргами.

    `args` — ожидаемые исходники первых позиционных аргументов (ast.unparse),
    `kwargs` — {имя: исходник значения}. Дополнительные аргументы разрешены.
    Для точечного `func` (с точками) имя сравнивается целиком, для короткого —
    как в `calls` (Name/алиас/Attribute.attr).
    """
    sites = call_sites(obj, func) if not isinstance(obj, ast.AST) \
        else call_sites_in(obj, func)
    return _match_sites(sites, func, args, kwargs)


def has_call_in(node: ast.AST, func: str, args: Optional[Iterable[str]] = None,
                kwargs: Optional[Dict[str, str]] = None) -> bool:
    return _match_sites(call_sites_in(node, func), func, args, kwargs)


def _match_sites(sites: List[ast.Call], func: str,
                 args: Optional[Iterable[str]],
                 kwargs: Optional[Dict[str, str]]) -> bool:
    if args is None and kwargs is None:
        return bool(sites)
    want_args = list(args or ())
    want_kw = dict(kwargs or {})
    for site in sites:
        if len(site.args) < len(want_args):
            continue
        if any(unparse(site.args[i]) != a for i, a in enumerate(want_args)):
            continue
        got = {k.arg: unparse(k.value) for k in site.keywords if k.arg}
        if any(got.get(k) != v for k, v in want_kw.items()):
            continue
        return True
    return False


# ─────────────────────────── присваивания / имена ───────────────────────────

def assigns(obj: Any, target: str, value: Optional[str] = None) -> bool:
    """Есть ли присваивание (в т.ч. annotated/augmented) в цель `target`.

    Цель/значение сравниваются через `ast.unparse` (эквивалентность узла).
    `value=None` — любой RHS; иначе исходник RHS должен совпасть.
    """
    tree = ast.parse(_as_src(obj))
    for node in ast.walk(tree):
        tgts: List[ast.AST] = []
        rhs: Optional[ast.AST] = None
        if isinstance(node, ast.Assign):
            tgts, rhs = list(node.targets), node.value
        elif isinstance(node, ast.AnnAssign):
            tgts, rhs = [node.target], node.value
        elif isinstance(node, ast.AugAssign):
            tgts, rhs = [node.target], node.value
        for t in tgts:
            if unparse(t) != target:
                continue
            if value is None or (rhs is not None and unparse(rhs) == value):
                return True
    return False


def identifiers(obj: Any) -> set:
    """Все идентификаторы AST: Name.id, Attribute.attr, имена def/class/arg/kw."""
    tree = ast.parse(_as_src(obj))
    out: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.arg):
            out.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            out.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                out.add((a.asname or a.name).split('.')[0])
    return out


def has_name(obj: Any, name: str) -> bool:
    return name in identifiers(obj)


def has_augassign(obj: Any, target: str, op: Optional[str] = None) -> bool:
    """Аугментированное присваивание в цель (напр. 'cfg.seq_len', op='floordiv')."""
    for node in ast.walk(ast.parse(_as_src(obj))):
        if isinstance(node, ast.AugAssign) and unparse(node.target) == target:
            if op is None or type(node.op).__name__.lower() == op.lower():
                return True
    return False


def has_if(obj: Any, test: str) -> bool:
    """Есть ли `if` с условием, чей AST-unparse совпадает с `test`."""
    for node in ast.walk(ast.parse(_as_src(obj))):
        if isinstance(node, ast.If) and unparse(node.test) == test:
            return True
    return False


_CMP_OPS = {
    '==': ast.Eq, '!=': ast.NotEq, '<': ast.Lt, '<=': ast.LtE,
    '>': ast.Gt, '>=': ast.GtE, 'is': ast.Is, 'is not': ast.IsNot,
    'in': ast.In, 'not in': ast.NotIn,
}


def has_compare(obj: Any, left: str, op: str, right: str) -> bool:
    """Есть ли сравнение `left op right` (сравнение — по AST, операнды unparse)."""
    if op not in _CMP_OPS:
        raise ValueError(f'unknown op {op!r}; use {sorted(_CMP_OPS)}')
    for node in ast.walk(ast.parse(_as_src(obj))):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        if type(node.ops[0]) is _CMP_OPS[op] \
                and unparse(node.left) == left \
                and unparse(node.comparators[0]) == right:
            return True
    return False


def has_str(obj: Any, value: str) -> bool:
    """Есть ли строковый литерал ровно со значением `value`."""
    for node in ast.walk(ast.parse(_as_src(obj))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and node.value == value:
            return True
    return False


def str_values(obj: Any) -> List[str]:
    """Все строковые константы дерева (для поиска подстрок в help/docstring)."""
    return [n.value for n in ast.walk(ast.parse(_as_src(obj)))
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def has_dict_entry(obj: Any, key: Any, value: Optional[str] = None) -> bool:
    """Есть ли ключ в литерале словаря (значение — по unparse, если задано)."""
    want_key = repr(key)
    for node in ast.walk(ast.parse(_as_src(obj))):
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values):
            if k is None:
                continue
            if unparse(k) != want_key:
                continue
            if value is None or (v is not None and unparse(v) == value):
                return True
    return False


def top_stmts(obj: Any) -> List[ast.stmt]:
    return list(ast.parse(_as_src(obj)).body)


def find_def(obj: Any, name: str):
    """FunctionDef/AsyncFunctionDef/ClassDef верхнего уровня по имени."""
    tree = ast.parse(_as_src(obj))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and node.name == name:
            return node
    return None
