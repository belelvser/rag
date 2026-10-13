# Mini Helpdesk RAG

Локальный RAG-сервис с FastAPI, A2A, Qdrant и локальной Qwen через Ollama.

```text
вопрос → FastAPI → A2A agent → MiniLM → Qdrant → Qwen/Ollama → ответ
```

## Стек

| Компонент | Назначение |
|---|---|
| FastAPI / Swagger | Пользовательский HTTP API и документация |
| A2A Python SDK | Обмен API с отдельным агентом через JSON-RPC |
| all-MiniLM-L6-v2 | Embeddings вопросов и фрагментов |
| Qdrant | Хранение векторов и поиск |
| Qwen / Ollama | Локальная генерация по найденному контексту |
| Docker Compose | Запуск сервисов и одноразового init |

## Запуск

Нужен Docker Desktop в режиме Linux containers и готовые файлы `data/processed`.
Команды PowerShell выполняются из корня проекта; локальный Python-агент должен быть остановлен.

```powershell
docker compose up --build -d
# Прогресс подготовки — в отдельном терминале:
docker compose logs -f init
```

Первый запуск скачивает образы и отсутствующие модели. Повторный использует сохранённые
volumes; init импортирует готовые embeddings без повторного расчёта.

```powershell
# Остановить сервисы, сохранив данные:
docker compose stop
# Повторный запуск:
docker compose up -d --no-build --pull never
```

- [Swagger](http://127.0.0.1:8000/docs)
- [Agent Card — локальный запуск](http://127.0.0.1:9999/.well-known/agent-card.json)

В текущем Compose порт агента 9999 не опубликован; внутри сети его Agent Card доступна
по `http://agent:9999/.well-known/agent-card.json`.

## Пример `/ask`

```powershell
$body = @{ question = 'What Guinness record was set during the filming of Spectre?' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/ask' -ContentType 'application/json' -Body $body -TimeoutSec 330 | ConvertTo-Json -Depth 6
```

Ответ содержит текст модели, переданные ей источники и время поиска и генерации.

## Папки

| Папка | Содержимое |
|---|---|
| `app/` | FastAPI, схемы и A2A-клиент |
| `agent/` | A2A-сервер RAG-агента |
| `scripts/` | Подготовка, CLI и проверки |
| `data/processed/` | Корпус, фрагменты, embeddings и manifests |
| `results/` | Локальные отчёты проверок и оценки |

## Результаты

| Показатель | Значение |
|---|---:|
| Документы | 500 |
| Фрагменты | 666 |
| Размерность embeddings | 384 |
| DocHit@3 | 20/20 |
| AnswerSpanCoverage@3 | 20/20 |
| Retrieved Exact Match | 60% на 20 dev-вопросах |
| Retrieved Token F1 | 72.88% на 20 dev-вопросах |

## Ограничения

Учебный проект: CPU, локальная Qwen, один worker на Python-сервис. Первый запуск дольше
из-за скачивания и подготовки моделей. Без fine-tuning и внешнего API генерации.
