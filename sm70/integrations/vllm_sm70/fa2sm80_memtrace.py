# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
"""ПРИБОР ПАМЯТИ ХОСТА (FA2SM80_MEMTRACE=<секунды>, 26.10).

Повод: боевой сервер sm80 убит по памяти cgroup (oom-kill 26.10 02:42, потолок 14 ГБ) после ~9 ч нагрузки; рост
~0.6 ГБ/ч под нагрузкой, только у Worker_TP0, в безымянной памяти процесса (не драйверной, не shmem). Прибор пишет
раз в период в stderr каждого процесса сервера (API, движок, воркеры):
  * RSS по классам отображений: куча malloc ([heap]), безымянные mmap ([anon]), файлы, shmem -- и прирост;
  * кэш закреплённой памяти torch (если доступен);
  * FA2SM80_MEMTRACE_TM=1 -- tracemalloc: топ строк по приросту выделений Python с прошлого замера;
    FA2SM80_MEMTRACE_TM_ONLY=<подстрока имени процесса> -- только в нём (имя VLLM::Worker_TP0 выставляется после
    старта процесса, поэтому tracemalloc включается на первом замере, когда имя уже известно);
    FA2SM80_MEMTRACE_TM_BASE=1 -- прирост от первого снимка, а не от прошлого (медленный рост);
  * FA2SM80_MEMTRACE_GC=1 -- перепись объектов gc по типам с приростом (держит GIL ~секунду: на воркерах может
    сорвать ожидание флага в обмене -- включать только на стенде).
Ничего не меняет в вычислениях: поток только читает /proc и статистику.
"""
import os
import sys
import threading
import time


def _лог(msg):
    print(f"[fa2sm80 MEMTRACE] {msg}", file=sys.stderr, flush=True)


def _имя():
    try:
        return open("/proc/self/cmdline", "rb").read().split(b"\0")[0].decode(errors="ignore")[-40:]
    except Exception:  # noqa: BLE001
        return "?"


def _классы():
    """RSS (КиБ) по классам отображений из /proc/self/smaps."""
    out = {"heap": 0, "anon": 0, "file": 0, "shm": 0, "other": 0, "n_anon": 0, "big_anon": 0}
    имя = "anon"
    try:
        with open("/proc/self/smaps") as f:
            for line in f:
                c = line[:1]
                if c in "0123456789abcdef" and "-" in line.split(" ", 1)[0]:
                    parts = line.split()
                    p = parts[5] if len(parts) > 5 else ""
                    if p == "[heap]":
                        имя = "heap"
                    elif p == "":
                        имя = "anon"
                        out["n_anon"] += 1
                    elif p.startswith("/dev/shm") or p.startswith("/SYSV"):
                        имя = "shm"
                    elif p.startswith("/"):
                        имя = "file"
                    else:
                        имя = "other"
                elif line.startswith("Rss:"):
                    kb = int(line.split()[1])
                    out[имя] += kb
                    if имя == "anon" and kb >= 65536:
                        out["big_anon"] += 1
    except Exception as ex:  # noqa: BLE001
        _лог(f"smaps не читается: {ex}")
    return out


def _закреплённая():
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        st = torch.cuda.host_memory_stats() if hasattr(torch.cuda, "host_memory_stats") else None
        if not st:
            return None
        return {k: v for k, v in st.items() if k.endswith(".current") and "bytes" in k}
    except Exception:  # noqa: BLE001
        return None


def _видеопамять():
    """[ПАМЯТЬ 29.10] Сводка аллокатора CUDA ЭТОГО процесса -- только если CUDA здесь уже поднята (API-сервер и ядро
    движка не трогаются) и только по картам с зарезервированной памятью. Без вызовов CUDA API (только учёт
    аллокатора на хосте): фоновый поток не должен сорвать захват графа в другом потоке. Свободное у драйвера пишет
    наблюдатель (NVML). Приватные пулы (CUDA-графы) -- по сегментам снимка: так видно, как запас VRAM съедают
    графы. Снимок держит замок аллокатора миллисекунды; раз в период прибора (боевой -- 900 с)."""
    try:
        import torch
        if not torch.cuda.is_initialized():
            return []
        out = []
        for d in range(torch.cuda.device_count()):
            st = torch.cuda.memory_stats(d)
            if not st.get("reserved_bytes.all.current", 0):
                continue
            пул = пул_занято = n = 0
            по_пулам = {}                 # [01.11] каждый пул отдельно: графы цикла и тела MTP, графы цели
            for seg in torch.cuda.memory_snapshot():
                if seg.get("device") != d:
                    continue
                pid = seg.get("segment_pool_id")
                if pid is not None and tuple(pid) != (0, 0):
                    пул += seg.get("total_size", 0)
                    пул_занято += seg.get("allocated_size", 0)
                    n += 1
                    п = по_пулам.setdefault(tuple(pid), [0, 0, 0])
                    п[0] += seg.get("total_size", 0)
                    п[1] += seg.get("allocated_size", 0)
                    п[2] += 1
            out.append((d, st.get("allocated_bytes.all.current", 0), st.get("reserved_bytes.all.current", 0),
                        st.get("allocated_bytes.all.peak", 0), пул, пул_занято, n,
                        st.get("num_alloc_retries", 0), st.get("num_ooms", 0), по_пулам))
        return out
    except Exception:  # noqa: BLE001
        return []


# [ПАМЯТЬ 29.10] FA2SM80_MEMHIST=<префикс> -- ТОЛЬКО для диагностического экземпляра: в процессе, чьё имя содержит
# FA2SM80_MEMHIST_ONLY (умолчание Worker_TP0), включается история выделений CUDA со стеками Python (дорого: каждая
# аллокация снимает стек) по появлению файла <префикс>.start; при появлении <префикс>.trigger пишется снимок <префикс>.<n>.pickle (кто держит
# память, сегменты по потокам и пулам). В бою не задаётся.
_ИСТОРИЯ = {"вкл": False, "n": 0}


def _история_памяти():
    префикс = os.environ.get("FA2SM80_MEMHIST", "")
    if not префикс or os.environ.get("FA2SM80_MEMHIST_ONLY", "Worker_TP0") not in _имя():
        return
    try:
        import torch
        if not torch.cuda.is_initialized():
            return
        if not _ИСТОРИЯ["вкл"] and os.path.exists(префикс + ".start"):
            os.remove(префикс + ".start")
            torch.cuda.memory._record_memory_history(max_entries=int(os.environ.get("FA2SM80_MEMHIST_N", "200000")),
                                                     stacks="python")
            _ИСТОРИЯ["вкл"] = True
            _лог(f"история выделений CUDA включена в {_имя()}")
        if os.path.exists(префикс + ".trigger"):
            os.remove(префикс + ".trigger")
            _ИСТОРИЯ["n"] += 1
            путь = f"{префикс}.{_ИСТОРИЯ['n']}.pickle"
            torch.cuda.memory._dump_snapshot(путь)
            _лог(f"снимок памяти CUDA записан: {путь}")
    except Exception as ex:  # noqa: BLE001
        _лог(f"история выделений: {type(ex).__name__}: {ex}")


def _цикл(период, tm, gc_on):
    import collections
    прежн = None
    снимок = None
    типы_прежн = collections.Counter()
    только = os.environ.get("FA2SM80_MEMTRACE_TM_ONLY", "")
    tm_идёт = False
    while True:
        time.sleep(период)
        if tm and not tm_идёт and (not только or только in _имя()):
            import tracemalloc
            tracemalloc.start(int(os.environ.get("FA2SM80_MEMTRACE_TM_FRAMES", "1")))
            tm_идёт = True
            _лог(f"tracemalloc включён в {_имя()} pid={os.getpid()}")
        к = _классы()
        d = "" if прежн is None else " | прирост " + " ".join(
            f"{x}{(к[x] - прежн[x]) / 1024:+.0f}" for x in ("heap", "anon", "file", "shm"))
        _лог(f"{_имя()} pid={os.getpid()}: heap={к['heap'] / 1024:.0f} anon={к['anon'] / 1024:.0f} "
             f"(отображений {к['n_anon']}, >=64M {к['big_anon']}) file={к['file'] / 1024:.0f} "
             f"shm={к['shm'] / 1024:.0f} МиБ{d}")
        прежн = к
        _история_памяти()
        for d, a, r, pk, pl, plu, n, rt, oo, пп in _видеопамять():
            _лог(f"{time.strftime('%H:%M:%S')} GPU{d}: выделено {a / 2**20:.0f} резерв {r / 2**20:.0f} пик {pk / 2**20:.0f} | приватные пулы "
                 f"{pl / 2**20:.0f} (занято {plu / 2**20:.0f}, сегментов {n}) МиБ | повторов {rt}, OOM {oo}")
            if пп:
                _лог(f"{time.strftime('%H:%M:%S')} GPU{d} по пулам (id: всего/занято МиБ, сегментов): " + " ".join(
                    f"{pid[0]}.{pid[1]}: {v[0] / 2**20:.0f}/{v[1] / 2**20:.0f} ({v[2]})"
                    for pid, v in sorted(пп.items(), key=lambda kv: -kv[1][0])[:8]))
        з = _закреплённая()
        if з:
            _лог("закреплённая (torch host): " + " ".join(f"{k}={v / 2**20:.0f}M" for k, v in sorted(з.items())))
        if tm_идёт:
            import tracemalloc
            нов = tracemalloc.take_snapshot()
            if снимок is not None:
                top = нов.compare_to(снимок, "lineno")[:12]
                for s in top:
                    if s.size_diff > 256 * 1024:
                        fr = s.traceback[0]
                        _лог(f"  tm {s.size_diff / 2**20:+.1f}M (всего {s.size / 2**20:.1f}M, x{s.count}) "
                             f"{fr.filename[-60:]}:{fr.lineno}")
            # [01.11] FA2SM80_MEMTRACE_TM_BASE=1 -- прирост от ПЕРВОГО снимка (медленный рост за период ниже порога)
            if снимок is None or os.environ.get("FA2SM80_MEMTRACE_TM_BASE", "0") != "1":
                снимок = нов
        if gc_on:
            import gc
            t0 = time.time()
            типы = collections.Counter(type(o).__name__ for o in gc.get_objects())
            дельта = {k: типы[k] - типы_прежн.get(k, 0) for k in типы}
            топ = sorted(дельта.items(), key=lambda kv: -kv[1])[:10]
            _лог(f"  gc ({time.time() - t0:.1f} с): " + " ".join(f"{k}{v:+d}({типы[k]})" for k, v in топ if v > 0))
            типы_прежн = типы


def запустить():
    """Поток прибора в ЭТОМ процессе (если рычаг задан). Повторный вызов -- без эффекта."""
    период = int(os.environ.get("FA2SM80_MEMTRACE", "0") or 0)
    if период <= 0 or getattr(запустить, "_идёт", False):
        return
    запустить._идёт = True
    tm = os.environ.get("FA2SM80_MEMTRACE_TM", "0") == "1"
    gc_on = os.environ.get("FA2SM80_MEMTRACE_GC", "0") == "1"
    threading.Thread(target=_цикл, args=(период, tm, gc_on), name="fa2sm80-memtrace", daemon=True).start()
    _лог(f"включён в {_имя()} pid={os.getpid()}: период {период} с, tracemalloc={tm}, gc={gc_on}")
