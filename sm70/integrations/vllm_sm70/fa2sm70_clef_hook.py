# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# КРЮК ПОПРАВКИ CLEF В ТЕЛАХ УМНОЖЕНИЯ (W8 / tm8 / W12). Лёгкий модуль без зависимостей: его
# импортируют fa2sm70_w8 и fa2sm70_w12 ДО регистрации своих операций.
#
# Почему внутри тела, а не отдельной операцией после него: отдельная мутирующая операция в графе
# torch.compile проходит через функционализацию (копия y на КАЖДОМ шаге, если переукладка не
# сработает) -- а это цена для всех запросов. Внутри уже непрозрачного тела поправка не видна
# компилятору вовсе: сигнатура операции, граф и захват те же. Модуль узнаётся по указателю веса
# (таблица в расширении), поэтому новых аргументов нет.
#
# FA2SM70_CLEF=0 (умолчание): с_поправкой() возвращает тело КАК ЕСТЬ -- ни одной лишней команды.
import functools
import os
from contextlib import contextmanager
from contextvars import ContextVar

ВКЛ = os.environ.get("FA2SM70_CLEF", "0") == "1"
# диагностика (12.10): тела W8/tm8/W12 без обёртки -- поправка на них не действует вовсе
_БЕЗ_ОБЁРТКИ = os.environ.get("FA2SM70_CLEF_NOWRAP", "0") == "1"
ХУК = [None]   # ставит fa2sm70_clef после загрузки весов: f(x, y, [ключи]) -> None
# окно строк шага: f(x, y, [ключи], r0) -- x/y покрывают строки [r0, r0+M) (конвейер fa2sm80_rowpipe)
ХУК_ДИАП = [None]
# Only the synchronous Python submission is scoped. CUDA kernels receive the
# resolved offset at enqueue/capture time; no Python state is read during replay.
_ROW_WINDOW = ContextVar("clef_row_window", default=None)


def row_window_offset():
    window = _ROW_WINDOW.get()
    return None if window is None else window[0]


@contextmanager
def row_window(start, rows):
    """Map local projection rows to the original target batch, including nesting."""
    parent = _ROW_WINDOW.get()
    if start < 0 or rows < 0 or (parent is not None and start + rows > parent[1]):
        raise ValueError("invalid Clef row window")
    offset = start + (parent[0] if parent is not None else 0)
    token = _ROW_WINDOW.set((offset, rows))
    try:
        yield
    finally:
        _ROW_WINDOW.reset(token)


def с_поправкой(fn, ключи):
    """ключи -- номера позиционных аргументов тела, чьи тензоры служат ключом модуля"""
    if not ВКЛ or _БЕЗ_ОБЁРТКИ:
        return fn

    @functools.wraps(fn)
    def обёртка(*a):
        y = fn(*a)
        h = ХУК[0]
        if h is not None:
            h(a[0], y, [a[i] for i in ключи])
        return y

    return обёртка
