# Подготовленные тесты OOM 29.10.2026, 04:54

**Статус: написаны, НЕ ЗАПУСКАЛИСЬ.** По указанию владельца действующий vLLM продолжает работать и накапливать статистику. Не запускались ни CPU/GPU-тесты, ни их импорты, ни сборка. Runtime, сервисы, наблюдатели, конфигурация и лимиты не изменены. Автоматическое расписание тестов не создано.

Код подготовлен в кандидате:

- [oom_regression_micro.py](../../../scripts/oom_regression_micro.py): отдельный вызов реального selector extension либо `torch.nn.functional.linear` с контролируемым allocator budget, retained CUDA graph pool и сохранением результатов.
- [oom_regression_service.py](../../../scripts/oom_regression_service.py): один вручную выбранный HTTP-сценарий на изолированном TP2-сервере; использует прежние acceptance clients.
- [oom_regression_guard.py](../../../scripts/oom_regression_guard.py): обязательное окно обслуживания, отсутствие live-сервиса/порта, ограничения cgroup и GPU, запрет скрывать сбой перезапуском.

## 1. Что известно об аварии

Один инцидент 04:54:17 на GPU0 содержит два пути:

| Путь | Неудачное выделение | Driver free | Дополнительные сведения |
| --- | --- | --- | --- |
| `mtp.layers.0.self_attn.attn` → `_prefill_row` → `sparse_select` | 144 MiB | 126,25 MiB | 511,63 MiB private pools; 595,27 MiB reserved-but-unallocated |
| `fa2sm80_blockpipe.py::_прямой_кусок` → `L.mlp.gate_up_proj(h)` → `default_unquantized_gemm` → `torch.nn.functional.linear` | 272 MiB | 206,25 MiB | 511,63 MiB private pools; 657,78 MiB reserved-but-unallocated |

Источник: `кандидат/state/acceptance/live-20261029/server.log`, первые стеки около строк10900–10984 и12610–12708. Дубли одного traceback не считать отдельными падениями. Сервис перезапущен один раз, API восстановился в04:57:06.

Стек подтверждает **MTP-prefill**, а не только target attention. Полные формы, вход пользователя и точная последовательность allocations не сохранены; размеры ниже — совместимые гипотезы, не реконструкция пользовательского запроса.

При текущих `B=64`, `TI=128`, `PROBE=16`, TP-shard `Hkv=2`, `H=12` одна fp32 score matrix занимает:

```text
ceil(Sq / 128) * 2 * (16 * 6) * ceil(T / 64) * 4 байт
Sq=8192, T=196608 -> 144 MiB
Sq=8192, T=212992 -> 156 MiB
Sq=8192, T=262140 -> 192 MiB
```

Для fp16 gate/up TP2: `hidden_size=5120`, `intermediate_size=17408`, выход shard имеет17408 столбцов. `8192 * 17408 * 2 = 272 MiB`. Сам output при этой форме не может поместиться в206 MiB доступного allocator budget: исправление полного пути должно освобождать другую живую память или менять владение/поток обработки с сохранением качества. Нельзя требовать невозможного от неизменённого `F.linear`.

## 2. Микротесты операций

Каждый запуск — отдельный процесс, один physical GPU0 **или** GPU1. До CUDA import проверяется, что оба GPU0–1 свободны. GPU2–3 не опрашиваются. Сборка внутри теста запрещена: sparse использует явно переданный существующий `.so` из кандидата, сохраняет его SHA256. ABI/свежесть сборки до будущего запуска нужно сверить с проверяемым commit.

| Ось | Значения / назначение |
| --- | --- |
| Selector | Фактические GEMM=1, ROWNORM=1, gamma0,5, beta0, B64, mass0,85; без kbarPool, как в первом стеке |
| Длины контекста | 196607/196608/196609: граница score allocation; 212992: предыдущий156MiB случай; 262140: почти максимальный контекст |
| Query rows | 8191/8192/8193: границы tile и формы. 8193 — только локальная проверка ядра, не повышение server batch8192 |
| Запас allocator | 126/144/206/272/384/512/768/1024 MiB сверх текущего reserved baseline |
| Graph pool | 0 или512 MiB retained capture allocation; проверка replay sentinel после вызова |
| Layout | Нетривиальный обратный порядок KV-страниц; fp16 Q, int8 K, fp32 scales |
| Проверка данных | Finite outputs, допустимые counts/indices; `--reference` сравнивает все выходы побитово с сохранённым просторным baseline той же формы/seed |

Тест сначала создаёт inputs и прогревает операцию/handles, затем ограничивает allocator через `set_per_process_memory_fraction`. Он не занимает балластом всю VRAM. Это **искусственная граница allocator**, не равнозначная126MiB driver free при595MiB неиспользуемого cache. Истинную фрагментацию, прямые CUDA allocations, остальные слои и взаимодействие TP проверяет полный сервер.

Порядок будущего исполнения:

1. Просторный baseline1024MiB, graph pool0; затем512MiB pool. Получить `outputs.pt`, peak allocated/reserved и cap. Setup OOM считается ошибкой подготовки, а не успешным воспроизведением.
2. Один central case: sparse8192×196608, затем linear8192×5120→17408. Снижать headroom отдельными процессами. `--expect oom` использовать только для намеренно недостаточного бюджета и подтверждения механизма; такой результат **не является прохождением регрессионной приёмки**.
3. Для нового исправления сравнить тот же case/headroom с `--expect pass --reference ...`. Не менять reference, seed, gamma, mass или форму ради совпадения.
4. Расширить на соседние формы, максимальный контекст и второй TP-rank GPU1. При различии selector indices/данных остановить приёмку: меньший workspace не оправдывает смену разреженности.

На OOM сохраняются сообщение, allocator snapshot, allocated/reserved/peaks/retries; повторного вызова операции для скрытого восстановления нет. Непредвиденная ошибка завершает процесс. Успех микротеста не означает сохранения качества модели.

## 3. Полный сервер: реальные пути и накопленная память

Скрипт не поднимает и не останавливает сервисы. Для будущего окна нужен отдельный уже запущенный кандидат на `127.0.0.1:8086`, GPU0–1, TP2, прежние context/maxseq/batch/KV/MTP. У unit должен быть `Restart=no`; рабочий live unit должен быть вручную остановлен в рамках разрешённого обслуживания. Наблюдатель уважает ручной inactive.

| Фаза скрипта | Нагрузка | Назначение |
| --- | --- | --- |
| `quality` | Базовые chat/Clef fixtures | Контроль до/после памяти и состояний |
| `sparse196k` | C1, prompt196608, output2048, MTP включён | Кандидат на144MiB score, большой MTP-prefill |
| `sparse213k` | C1, prompt212992, output2048 | Ранее наблюдавшийся156MiB путь |
| `sparse-limit` | C1, prompt262140, output4 | Верхняя граница контекста без его уменьшения |
| `mixed-boundary` | C8×(49147+2048), 4 Clef34816 | Совместные активации, Clef, GDN boundary, MTP |
| `images` | Chat/Clef с изображениями | Предварительная история encoder/cache перед повтором mixed |
| `queue40` | 40×(8192+1024) | Очередь2,5×maxseq16, рабочие8192-row chunks |
| `mixed-queue` | 20chat+10Clef | Давление на admission, preemption и временные буферы |

Сначала выполнить фазы на свежем сервере, затем **без restart** последовательность quality → sparse196k → sparse213k → images → mixed-boundary → queue40 → mixed-queue → sparse-limit → quality. Это отличает холодный успех от эффекта прогретых графов, encoder cache и смены форм. Каждая фаза выбирается вручную; автоматического цикла/soak пока нет.

Нельзя рассчитывать, что один prompt196608 означает `Sq=8192,T=196608` именно в MTP: chunking, prefix hits и scheduler могут дать другую форму. По этой причине успешная HTTP-фаза помечается `workload_passed_path_coverage_unverified`, а не release pass. Повторы прежнего corpus могут попадать в prefix cache; для доказательства покрытия нужен отдельный диагностический запуск с записанной фактической формой и cache hit/miss. Менять рабочий prefix cache сейчас не требуется.

Для окончательной приёмки в отдельном диагностическом процессе ещё нужно зарегистрировать без содержимого запросов:

- target/MTP phase, layer, Sq/T, total rows, Clef rows, graph mode и pool;
- вход/выход selector, gamma/ROWNORM и размеры его живых temporaries;
- вход/выход gate_up, dtype/shape, живые residual/activation buffers;
- allocator allocated/reserved/private pools и direct CUDA/driver free на каждом ранге;
- preemption/resume/cancel, image-cache history и ключи новых графов.

Такую instrumentation не добавлять в исполняемый сейчас процесс. Нельзя объявить тест закрывающим144/272MiB OOM, пока он не попал в требуемые ветви или пока эти ветви не получили доказанный предел памяти.

Клиент сохраняет метрики и GPU0–1 free/reserved/used каждую примерно секунду, cgroup RAM events, MainPID/InvocationID/NRestarts, SSE gaps и результаты существующих fixtures. Смена процесса, ошибка HTTP/client, CUDA OOM в журнале или wall-time>1800с — failure. Автоповтор и автоматический restart отсутствуют; остановить можно только процесс тестового клиента. Секундный NVML sample может пропустить пик; allocator instrumentation остаётся обязательной.

## 4. Команды-шаблоны — НЕ выполнять сейчас

Все команды ниже только для будущего разрешённого окна. Они сами не останавливают live vLLM. Из корня кандидата, с новыми каталогами output:

```bash
# Сначала явно выбрать проверенный существующий candidate .so.
python3 scripts/candidate_build.py -- .venv/bin/python scripts/oom_regression_micro.py \
  sparse ../state/acceptance/oom-sparse-roomy \
  --maintenance-window --gpu 0 --rows 8192 --context 196608 \
  --headroom-mib 1024 --graph-pool-mib 512 \
  --extension /ABSOLUTE/CANDIDATE/BUILD/fa2sm70_prefill.so

python3 scripts/candidate_build.py -- .venv/bin/python scripts/oom_regression_micro.py \
  linear ../state/acceptance/oom-linear-206 \
  --maintenance-window --gpu 0 --rows 8192 --headroom-mib 206 --expect oom

# Только когда тестовый TP2 unit уже подготовлен отдельно на8086, Restart=no.
python3 scripts/candidate_build.py -- .venv/bin/python scripts/oom_regression_service.py \
  mixed-boundary ../state/acceptance/oom-mixed-boundary \
  --maintenance-window --unit kandidat-acceptance-REPLACE.service \
  --server-log /ABSOLUTE/CANDIDATE/state/acceptance/REPLACE/server.log
```

Build runner используется как существующая оболочка RAM6GiB/CPU400%/host guard; микротест не вызывает JIT, не устанавливает пакеты и не меняет allocator настройки другого процесса. До будущего GPU-запуска проверить код и CLI в CPU-only режиме — эта проверка по текущему запрету тоже пока не выполнялась.

## 5. Что считать закрытием регрессии

Нужны совместно: подтверждённое покрытие MTP selector и полного fp16 gate/up, отсутствие OOM в прогретом полном профиле, тот же контекст/KV/параллельность, качество/состояния до и после нагрузки, ограниченный расход памяти и отсутствие потери скорости. После локальных фаз — контролируемые6h/24h из [общего плана](COMPLETION_REPORT_AND_PLAN.md). Перезапуск наблюдателем — эксплуатационное восстановление, не успешный тест устранения OOM.
