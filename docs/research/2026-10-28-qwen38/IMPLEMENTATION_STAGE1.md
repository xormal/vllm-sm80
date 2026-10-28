# Этап 1: первый пакет правок памяти, CPU-проверки

2026-10-28. Только кандидат, исходная точка `b8294d701`. Разрешены правки и CPU-тесты. Остановка production и запуск кандидата на GPU0–1 требуют последующего разрешения владельца. GPU2–3 исключены: там идёт обучение другого проекта.

## Реализовано

| Изменение | Файлы | Поведение |
| --- | --- | --- |
| Ленивый all-gather workspace | `fa2sm70_exch_i12.py` | Успешный CE-путь создаёт только send buffer; gather нужен лишь NCCL/fallback/capture |
| Владение captured storage | `fa2sm70_workspace.py` | Eager-only старые поколения освобождаются при росте; адреса, использованные в capture, сохраняются. Ошибка нового allocation не разрушает прежнего владельца |
| Lifetime fallback | `fa2sm80_blockpipe.py` | y освобождается после enqueue нормы, h — после gate_up, gu — после SiLU; residual остаётся fp32, порядок операций прежний |
| Кусочный fallback, opt-in | `fa2sm80_fallback.py`, `fa2sm80_blockpipe.py` | Out/norm/gate_up/SiLU/down через прежние модули на ограниченных окнах строк. Итоговые output/residual остаются полными |
| Смещения Clef | `fa2sm70_clef_hook.py`, `fa2sm70_clef.py` | ContextVar окна переводит локальные строки в глобальные, включая вложенный range hook; состояние восстанавливается при исключении |
| Ограничение устройств | оба launchers кандидата | `CARDS`, отличный от `0,1`, отклоняется до toolchain/device операций |

Файлы runtime находятся в `sm70/integrations/vllm_sm70`. Штатные scheduler batch, лимиты контекста, параллельность, KV, MTP_K, attention и GDN не изменены. Активные CUDA-ядра и формат весов не менялись; полная пересборка не выполнялась.

Для n=8192×5120 размер упакованной строки обмена даёт `2 × (n/128) × 194 = 127139840` байт all-gather storage, или 121,25 МиБ на ранг. Это устранённое лишнее выделение на успешном CE-пути, **не измеренная экономия VRAM процесса**. Если более ранний NCCL/capture уже создал gather buffer, он сохраняется; в capture CE по-прежнему не используется.

Новый cache сохраняет captured generations без eviction. Он исправляет владение адресами, но не является конечным бюджетом графов. Concurrent independent streams, очистка после destruction graphs и ограничение числа captures остаются отдельной задачей. Порядок TP collectives и out-of-place unpack сохранены.

## Кусочный путь

`FA2SM80_BLOCKPIPE_FALLBACK_ROWS=0` по умолчанию: нарезка выключена. Для будущего стендового A/B первый вариант — 1024. Путь включается только для большого 2D fallback при M≥BLOCKPIPE_MIN и M>FALLBACK_ROWS. Декод и обычный all-W8 blockpipe продолжают прежний маршрут.

Разрезаются только stateless projection/MLP вычисления после mixer. Не меняется размер шага attention/GDN и не выключается sparse prefill через уменьшение scheduler chunk. Модули по-прежнему сами применяют bias и TP reduction. Clef correction выполняется в их прежнем месте до row-parallel reduction, с глобальным смещением строк.

При активном Clef без range hook остаётся исходный fallback. Это ограничение совместимости, а не гарантия отсутствия OOM в такой конфигурации. Опциональный путь через границу слоёв (`BLOCKPIPE_X`) и его `_след_прямо` пока не ограничены по промежуточной памяти; текущий production X не включает.

Куски меняют M для GEMM и TP-пороги; последний короткий кусок может сменить маршрут W8/AR2/EXCH12. Поэтому одинаковые математические операции не доказывают побитовое совпадение. Пересчёт W8 dequant в отдельных кусках тоже может стоить скорости. До GPU-гейтов этот флаг не включать по умолчанию.

## Проверено без GPU

Команда под лимитами 6 ГиБ RAM / CPU400%, с `CUDA_VISIBLE_DEVICES=""`:

```bash
scripts/candidate_build.py -- .venv/bin/python sm70/tests/test_sm80_memory_cpu.py -v
```

**10 тестов прошли**, включая подслучаи:

- CE не создаёт gather, повтор меньшей формы не растит allocation;
- capture, превышение CE capacity, отсутствие CE state, отключённый CE, PyNCCL/torch.distributed и world_size=3 сохраняют fallback transport;
- captured адрес переживает рост, eager-only поколение освобождается; отказ allocation сохраняет старое состояние;
- промежуточные y/h/gu не удерживаются до down; core/residual не меняются, residual output fp32;
- куски и хвост покрывают все строки; Clef correction применяется один раз к нужным глобальным строкам, nested offset складывается один раз;
- исключение восстанавливает row context; отсутствие range hook, малый batch и выключенный флаг сохраняют прежний путь.

Положительный контроль на исходных функциях старого дерева: оба теста обнаружили прежнее лишнее all-gather allocation и прежний lifetime fallback (2/2 ожидаемых отказа, без ошибок самого harness). Проверены Python/Bash syntax, `git diff --check` и ранний отказ обоих launchers на `CARDS=2,3`. Этот тест не обращался к GPU2–3.

Тесты используют настоящие Python-тела функций и CPU transport/norm doubles. Они доказывают маршрутизацию, ownership и row indexing; CUDA arithmetic, allocator peak, stream ordering, TP rank skew, replay и скорость ими не проверены. В CPU-процессе запрещён `torch.cuda._lazy_init`.

## Следующие обязательные проверки

После разрешения владельца — только GPU0–1, с контролируемой заменой production, по [PLAN.md](PLAN.md):

1. Baseline без кусочного флага; затем lifetime/lazy exchange. Проверить реальные allocation peaks и eager/capture transitions, stale-pointer stress, одинаковые результаты и скорость.
2. Кусочный fallback 1024 против 0: M=8191/8192 и хвосты, слой 0 и другие неподходящие blockpipe слои, Clef строки на границах кусков, mixed C8+R4. Оценить смену GEMM/TP маршрутов.
3. Проверить FP16/Clef/logits/длинные ответы относительно baseline и независимых эталонов, throughput/p99/Clef backlog. Успешный CPU-тест не разрешает заявить устранение OOM или отсутствие просадки качества.

Бюджет MTP graphs, безопасный async Clef head и общий mixed graph ещё не реализованы. Этот пакет — начало устранения рабочего пика памяти, а не завершение всего плана.
