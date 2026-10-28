# Кандидат на боевом порту и автоматическое восстановление

29.10.2026 владелец разрешил принимать реальную нагрузку на8085, затем явно запросил автозапуск и наблюдателя с перезапуском. Это заменяет прежнее ограничение на localhost8086 и сохранение старого автозапуска. Код и unit исходного vLLM сохранены; его автозапуск теперь **disabled**, чтобы два экземпляра не заняли GPU0–1.

## Действующий запуск

- `kandidat-acceptance-live-20261029.service`: `0.0.0.0:8085`, только GPU0–1.
- Прежние четыре model aliases сохранены: `huihui-qwen3.5-35b-a3b-abliterated-q8_0`, `gpt-4-0125-preview`, `Qwen3.6-27B-AWQ`, `Qwen3.8-27B-ABLITERATED`.
- Профиль: admission1, convolution read/write1, MLP chunks0, metadata fusion0. Контекст262144, maxseq16, batch8192, MTPK4 и KV11295047936байт/GPU сохранены.
- MemoryHigh20GiB, MemoryMax21GiB, swap0, CPUQuota1600%, TasksMax4096. Сборки по-прежнему используют отдельный runner6GiB/CPU400%.
- Исходники unit-файлов и drop-ins сохранены в `configs/`; установлены в `~/.config/systemd/user/`. Все три unit включены через `default.target.wants`, `loginctl Linger=yes`.

Основной процесс не прерывался при включении восстановления: MainPID4172692 сохранился. Его текущий FragmentPath ещё указывает на transient unit; persistent unit и install symlink уже существуют на диске, а drop-in применяет restart policy сейчас. После загрузки системы используется постоянный unit из пользовательского config. Такой переход проверен отдельным CPU-only systemd probe.

## Восстановление

1. Systemd перезапускает завершившийся процесс через30с (`Restart=always`), с ограничением3 запуска за5мин. Ручной `systemctl stop` не отменяется политикой restart.
2. Перед запуском preflight ждёт MemAvailable≥16GiB, остановленного старого сервиса и доступности драйвера GPU0–1. Startup timeout900с; параметры модели ради восстановления не уменьшаются.
3. `kandidat-live-observer.service` проверяет `/health`, метрики прогресса, RAM и NVML каждые примерно10с. До первого успешного health даётся900с на старт. Шесть последовательных неудачных health-проверок после этого либо600с без изменения счётчиков при ожидающих/активных запросах инициируют restart.
4. Решения observer о восстановлении разделены минимум300с и требуют MemAvailable≥16GiB. После перехода в failed/start-limit observer выдерживает ещё300с перед повторным запуском. Он не поднимает вручную остановленный inactive unit.
5. `kandidat-acceptance-hostwatch.service` проверяет MemAvailable каждые0,5с и при падении ниже8GiB завершает только активный кандидат. Новый запуск ждёт восстановления16GiB. Оба наблюдателя имеют собственный Restart=always/5с и автозапуск.

Наблюдатель не читает и не записывает содержимое пользовательских сообщений. Метрики и действия: `кандидат/state/acceptance/live-20261029/telemetry.jsonl`, ротация8MiB×3; свежий снимок —`latest.json`. Журнал модели —`server.log` в том же каталоге. Host guard сохраняет `state/acceptance/host-memory.jsonl`.

## Проверено

- Шесть CPU-тестов политики: startup grace, health failures, cooldown, недостаток RAM, ручная остановка, отсутствие/наличие прогресса и новая invocation.
- Реальный SIGKILL отдельного CPU-only probe привёл к автоматическому запуску с новым PID и NRestarts=1. Рабочую модель для этого не прерывали.
- `systemd-analyze --user verify` принял постоянные units; install symlinks и linger проверены.
- `/health` отвечает200, `/v1/models` возвращает все четыре прежних имени и контекст262144. Реальные агенты уже получают HTTP200 на chat/completions и systemone.
- Исходный checkout остаётся чистым на351e4c74b; SHA256 его launcher, unit, конфигов и venv pth не изменились. Изменён только выбор автозапуска по явному запросу владельца.

Это эксплуатационный прогон по разрешению владельца, а не отмена открытых пунктов [нагрузочной приёмки](VALIDATION_RESULT.md): общий mixed graph не реализован, достаточный allocator reserve и длительный soak ещё не доказаны.

## Ручная остановка и откат

Для обслуживания остановить модель; observer уважает inactive после ручной остановки:

```bash
systemctl --user stop kandidat-acceptance-live-20261029.service
```

Возврат старого vLLM, включая его автозапуск; текущие запросы кандидата при остановке завершатся ошибкой:

```bash
systemctl --user disable --now kandidat-live-observer.service kandidat-acceptance-hostwatch.service kandidat-acceptance-live-20261029.service
systemctl --user enable --now sm80-vllm.service
```

Сохранённые файлы кандидата не удалять: это позволяет повторить запуск и изучить причину отказа. GPU2–3 во всех командах запуска и мониторинга исключены.
