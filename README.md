# RAG Evaluation with RAGAS

Оценка качества RAG-системы с помощью метрик, вдохновлённых RAGAS.

## Что внутри
- data/docs/ — документы (Python, Git, Docker).
- data/questions.json — 5 вопросов с эталонными ответами.
- src/evaluate_ragas.py — скрипт оценки.
- results/ragas_scores.json — метрики.

## Метрики
- context_precision — доля релевантных чанков в выдаче.
- context_recall — доля найденных релевантных чанков.
- faithfulness — насколько ответ опирается на контекст.
- answer_relevancy — насколько ответ релевантен вопросу.

## Как запустить
python3 src/evaluate_ragas.py

## Режимы
- Demo — упрощённые метрики без API (по умолчанию).
- Live — настоящий RAGAS с LLM-судьёй (требуется OpenAI API).

## Автор
redsea1970
