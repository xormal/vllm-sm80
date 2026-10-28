# Кандидат: отдельный код и ограниченные сборки

Создан 2026-10-28 от `351e4c74b` (исходный runtime checkpoint — `cbe00eb0c`). Ветка `kandidat`. Рабочий каталог: `/home/alex/devpack_sm70_20261022/кандидат/vllm-sm70`.

Текущее состояние GPU-проверок: [FULL_VALIDATION_PROGRESS.md](docs/research/2026-10-28-qwen38/FULL_VALIDATION_PROGRESS.md). Начальные build/smoke-проверки ниже — история этапов. Исправлены цикл preemption/recompute и адреса чтения/записи свёртки на границе KV-страницы. Длинные коды после реального вытеснения проходят8/8; очереди40chat и20chat+10Clef завершаются. Опциональное разбиение MLP выключено из-за изменений численных результатов. Отдельное fused integer metadata ядро прошло48 exact parity случаев, CUDA graph replay и функциональный model-level gate,но сквозного выигрыша не подтвердило и оставлено выключенным. Текущий launcher кандидата задаёт admission1,GRAN_CONV_ALL1,fallback0,fused0;длинный/HTTP/мультимодальный стресс завершён. [Итог](docs/research/2026-10-28-qwen38/VALIDATION_RESULT.md):OOM не наблюдался,но прямой NVML free доходил до5MiB;достаточный allocator reserve не доказан. Общий mixed graph не реализован, полный release pass не объявлен. [Диагностика](docs/research/2026-10-28-qwen38/GDN_BOUNDARY_FOLLOWUP.md). Активный временный service указан в `../state/acceptance/active-unit`; порт только localhost8086.

## Текущий эксплуатационный режим,29.10

Владелец разрешил боевой порт8085,автозапуск кандидата и восстановление при падении. Кандидат принимает реальные запросы;основной service и два наблюдателя включены при загрузке,старый `sm80-vllm.service` disabled. Старый код и unit не изменены. [Запуск,наблюдение и откат](docs/research/2026-10-28-qwen38/LIVE_DEPLOYMENT.md). Описания localhost8086 и прежнего автозапуска ниже относятся к предыдущему этапу проверки.

## Изоляция

Боевой `sm80-vllm.service` при запуске использует `/home/alex/devpack_sm70_20261022/sm80_work/bin/serve_prod_sm80.sh`. Его код, `.venv`, конфиги и unit не изменены. Кандидат не запускается при загрузке. После разрешения владельца создан временный `kandidat-validation.service` для локальных GPU-проверок.

У кандидата отдельные Git-репозитории vLLM и fa2_sm70_kernels, копия `.venv` без общих hardlinks, исправленные пути `.pth`/entrypoints и отдельные mutable caches в `../state`. Веса доступны через `../model`, ссылку на прежние веса; их не редактировать. Скопированы таблицы CG/W8 tuning, но не старые CUDA/Triton graph caches и не каталоги CMake со старыми абсолютными путями. Результаты сравнения после сборки и последующих исправлений приведены в документах приёмки; перенос окружения сам по себе не доказывает эквивалентности.

`../sm80_work/bench` и исследовательские документы сохранены. Исторические управляющие скрипты находятся в `../sm80_work/archive_bin`: это материалы для чтения и переноса, а не команды запуска — внутри есть старые stop/kill и внешние пути. В активном `../sm80_work/bin` только новые обёртки. Источник runtime-ядер — `sm70/` внутри этого vLLM; соседний `fa2_sm70_kernels` является отдельной копией библиотеки.

Remote `baseline` указывает на старое дерево для чтения истории; push URL отключён. Коммиты кандидата не изменяют старую ветку.

## Почему такие лимиты

На момент подготовки: 2 сокета × 36 физических ядер, SMT2 = 144 логических CPU; RAM 62,5 ГиБ, MemAvailable около 20 ГиБ, swap 8 ГиБ почти заполнен. Боевой vLLM занимает около 12,6 ГиБ в cgroup и может вырасти до своего лимита 21 ГиБ. Поэтому узкое место сборки — память, а не количество CPU.

Установлен `~/.config/systemd/user/kandidat-build.slice`, копия в [configs/kandidat-build.slice](configs/kandidat-build.slice). Он отделён от `sm80.slice` и автоматически активируется при запуске сборки; enable не нужен.

| Ограничение | Значение |
| --- | --- |
| MemoryHigh | 5 ГиБ: начинается reclaim/throttling |
| MemoryMax | 6 ГиБ: жёсткий общий предел группы сборок |
| MemorySwapMax | 0 |
| CPUQuota | 400%: суммарное время четырёх логических CPU, не привязка к четырём физическим ядрам |
| CPUWeight / nice | 10 / +15 |
| TasksMax | 256 процессов и потоков |
| MAX_JOBS / CMAKE_BUILD_PARALLEL_LEVEL | 2 / 2 |
| NVCC_THREADS | 1 |
| BLAS/OpenMP/Inductor compiler threads | 1 внутри сборочной обёртки |
| Начало сборки | MemAvailable ≥16 ГиБ |
| Аварийная остановка сборки | MemAvailable <8 ГиБ, проверка каждые 0,5 с |

Дополнительно: одна сборка одновременно через flock, OOM score 1000, ionice idle. IOWeight=10 записан в unit, но io controller в пользовательской cgroup не делегирован; действующую I/O-квоту не заявляем. Порог памяти проверяется по всей системе; watcher останавливает только собственную `kandidat-build-*.scope`. Это запас до существующего production memguard (<3 ГиБ), но не гарантия от мгновенного исчерпания RAM посторонним процессом.

Если отдельной единице компиляции не хватает 6 ГиБ, сборка может завершиться OOM. Не повышать лимит и не включать swap автоматически: выбирать меньше jobs/облегчённую цель или отдельное окно. Старый memrun с общим sm80.slice не использовать.

## Команды

Из каталога этого репозитория полная пересборка vLLM:

```bash
scripts/build_candidate_sm80.sh
```

Любая адресная сборка или проверка с потенциально большим RAM:

```bash
scripts/candidate_build.py -- .venv/bin/python your_build_script.py
```

Обёртка задаёт отдельные пути, скрывает GPU, фиксирует архитектуру 8.0 и запускает дочернее дерево в cgroup. Добавлен `FA2SM70_BUILD_ONLY=1`: preflight допускает компиляцию с явной архитектурой без доступа к CUDA device; поведение обычного runtime не меняется. Полная сборка vLLM не запускалась; 28.10 начата адресная сборка собственных CUDA-расширений для первого GPU-запуска.

Просмотр путей будущего сервера без запуска:

```bash
scripts/serve_candidate_sm80.sh --paths
```

`--dry-run` также проверяет toolchain; `--run` требует `CARDS` и свободной пары GPU. По умолчанию кандидат слушает `127.0.0.1:8086`. Вызов без аргументов только показывает пути. Предварительная сборка собственных CUDA-ядер при запуске кандидата идёт через ограниченную обёртку. Ресурсные лимиты относятся к сборочному процессу; будущий inference-процесс и его runtime JIT требуют отдельного бюджета перед испытанием. Для GPU-проверок используется временный service ниже.

Уточнение владельца: до явного разрешения — только CPU. После разрешения замена боевого сервиса возможна только на GPU0–1; GPU2–3 запрещены, там обучение другого проекта. Оба launchers кандидата отклоняют `CARDS`, отличающийся от `0,1`, до проверки CUDA toolchain и обращения к устройствам.

## Первые runtime-правки

Подготовлены ленивый NCCL workspace в CE-ветке, сохранение адресов exchange storage после capture и сокращение lifetime промежуточных тензоров fallback. Дополнительно реализован opt-in `FA2SM80_BLOCKPIPE_FALLBACK_ROWS=1024`: статeless projection/MLP fallback обрабатывается по строкам при прежнем attention/GDN и scheduler batch. По умолчанию значение 0 — старый размер вычислений. Малый decode не режется. Clef получает глобальный row offset; без доступного range hook остаётся прежний путь.

CPU-проверка под установленными лимитами, с невидимыми GPU:

```bash
scripts/candidate_build.py -- .venv/bin/python sm70/tests/test_sm80_memory_cpu.py -v
```

Состояние реализации и оставшиеся GPU-гейты: [IMPLEMENTATION_STAGE1.md](docs/research/2026-10-28-qwen38/IMPLEMENTATION_STAGE1.md). Эти изменения пока не доказывают устранение всех OOM или сохранение качества/скорости; общий mixed graph ещё не реализован.

## Выполненные проверки

- Прочитаны реальные cgroup-файлы дочернего процесса: memory.high=5368709120, memory.max=6442450944, memory.swap.max=0, cpu.max=`400000 100000`, pids.max=256.
- Импорты vLLM/fa2_sm70 и sys.prefix указывают в кандидат; `torch.cuda.is_initialized()` = false.
- Проверены отказ старта при недостатке памяти, эксклюзивный lock и остановка реальной тестовой scope с `sleep` при имитации низкого MemAvailable. RAM намеренно не исчерпывалась.
- Build-only preflight проверен с запретом GPU queries; отсутствие явной архитектуры отклоняется. Ядра не компилировались.
- Bash/Python syntax и `git diff --check` пройдены. Снимки проверок: `../state/build-limits-verified.json`, `../state/build-guard-verified.json`.

Дальнейшая работа: [план OOM и графов](docs/research/2026-10-28-qwen38/PLAN.md), [исследование оптимизаций и качества](docs/research/2026-10-28-qwen38/BREAKTHROUGH_OPTIMIZATIONS.md).

## Локальная GPU-проверка после разрешения 28.10

Владелец разрешил остановить production и работать с кандидатом на GPU0–1. Production остановлен; порт агентов 8085 оставлен закрытым. Его unit/автозапуск и прежний код сохранены. Остановка API превысила штатные 90 секунд: systemd завершил остаток группы, результат unit — timeout; GPU0–1 освободились раньше. Это не ошибка кандидата.

Воспроизводимый запуск (предварительно старый service должен быть остановлен):

```bash
scripts/start_candidate_validation.sh --run
```

Кандидат слушает только `127.0.0.1:8086`, имя `Qwen3.8-27B-ABLITERATED-candidate`. Unit временный, Restart=no, при загрузке не поднимается. Сервер: MemoryHigh=20G, MemoryMax=21G, swap=0, CPUQuota=1600%, Inductor compile threads=1. Для inference используется прежний sm80.slice и его memory guard; сборки по-прежнему вынесены в отдельный kandidat-build.slice с лимитами 6G/400%. Ограничение CPU сервера учитывать при интерпретации замеров.

Проверка компилятора и предварительная сборка расширений, включая cg_sm80, выполняются через candidate_build.py. Управляющий Python запускается с -I: не исполняет runtime sitecustomize из PYTHONPATH в системном Python без torch. Это не меняет Python рабочих процессов.

Журнал: `../state/gpu-validation/server.log`. Запросы smoke направлены только на фиксированный localhost:8086:

```bash
python3 scripts/smoke_candidate_http.py --stage basic --output ../state/gpu-validation/basic.json
python3 scripts/smoke_candidate_http.py --stage boundaries --output ../state/gpu-validation/boundaries.json
python3 scripts/smoke_candidate_http.py --stage concurrent --output ../state/gpu-validation/concurrent.json
python3 scripts/smoke_candidate_http.py --stage clef --output ../state/gpu-validation/clef.json
```

Smoke не заменяет quality/long-run и mixed C8+R4 gates. Отдельный `sm70/tests/test_workspace_cuda.py` проверяет реальный capture/replay после роста storage; запускать отдельно от модельной нагрузки с CUDA_VISIBLE_DEVICES=0,1, использует только cuda:0.

Остановить только стенд: `systemctl --user stop kandidat-validation.service`. Для возврата старой версии после освобождения GPU0–1: `systemctl --user start sm80-vllm.service`. Последняя команда вновь откроет 8085 ожидающим агентам.

Результат первого GPU-запуска: [GPU_VALIDATION_STAGE1.md](docs/research/2026-10-28-qwen38/GPU_VALIDATION_STAGE1.md). Кандидат оставлен работающим на localhost:8086; smoke до 208K и 16 HTTP-запросов прошли, полная приёмка качества и mixed OOM ещё не завершена.
