# Diagnostics LLM Dynamic Few-Shot Consensus Ensemble

## Что уже готово

Сейчас реализован пайплайн:

1. **Нормализация исходного датасета**
2. **LLM-preprocess клинических кейсов**
3. **Финальная очистка кейсов для retrieval**
4. **Векторизация кейсов**
5. **Семантический поиск top-K похожих кейсов**
6. **Few-shot prompt для LLM**
7. **Несколько прогонов LLM**
8. **Consensus между ответами**
   - frequency
   - weighted
   - embedding
   - hybrid
9. **CoT pipeline**
   - генерация CoT для кейсов
   - отдельные CoT embeddings
   - CoT few-shot режим
10. **Dialogue pipeline**
   - symptom intake
   - adaptive multi-turn anamnesis
11. **Evaluation scaffold**
   - запуск сравнения методов на тестовом датасете

## Общая логика пайплайна

### Baseline режим
1. Пользователь вводит жалобу
2. Система строит embedding запроса
3. Находит top-K похожих кейсов
4. Собирает few-shot prompt
5. Делает N прогонов LLM
6. Парсит structured JSON output
7. Строит consensus
8. Возвращает итоговые гипотезы, red flags и уточняющие вопросы

### CoT режим
1. Пользователь вводит жалобу
2. Система ищет top-K похожих кейсов с CoT
3. Формирует prompt с `complaint + cot + diagnosis`
4. Делает N прогонов LLM
5. Строит consensus

### Dialogue режим
1. Пользователь вводит жалобу
2. Система задает intake-вопросы
3. Формирует enriched query
4. Делает первичный анализ
5. Генерирует targeted follow-up questions
6. После каждого ответа обновляет гипотезы
7. Останавливается по stop condition

---

# Структура проекта

## `src/config.py`
Центральный файл конфигурации:
- пути к `data/`
- названия моделей
- `TOP_K`, `N_RUNS`
- параметры dialogue / CoT / evaluation

---

## `src/preprocessing/normalize_cases.py`
Приводит исходный `RuMedPrimeData.tsv` к единому JSON-формату.

**Результат:**
- `data/processed/RuMedPrimeData_normalized_full.json`

---

## `src/preprocessing/preprocess_llm.py`
LLM-preprocess кейсов:
- собирает complaint + anamnesis + findings
- делает краткое summary через T5
- формирует `text_for_embedding`

**Результат:**
- `data/processed/RuMedPrimeData_processed_llm.json`

---

## `src/preprocessing/finalize_processed_cases.py`
Финальная очистка кейсов:
- чистит `important_findings`
- строит `complaint_clean`
- строит `summary_llm_clean`
- собирает улучшенный `text_for_embedding_v2`

**Результат:**
- `data/processed/RuMedPrimeData_processed_final.json`

---

## `src/retrieval/embed_utils.py`
Общие функции для embeddings:
- `query:` / `passage:`
- получение embeddings через HF API
- нормализация векторов
- cosine similarity

---

## `src/retrieval/vectorize_api_HF.py`
Строит векторную базу по финализированным кейсам.

Использует:
- `text_for_embedding_v2`

**Результат:**
- `data/embeddings/RuMedPrimeData_embeddings_final.npy`
- `data/embeddings/RuMedPrimeData_embeddings_final_meta.json`

---

## `src/retrieval/search.py`
Semantic search по клиническим кейсам:
- embedding запроса пользователя
- similarity search
- top-K retrieval

---

## `src/llm/client.py`
Клиент для OpenRouter через `OPENROUTER_API_KEY`.

Используется для:
- baseline prompting
- CoT generation
- CoT few-shot
- dialogue analysis

---

## `src/llm/prompts.py`
Шаблоны prompt'ов:
- baseline few-shot prompt
- CoT prompt

---

## `src/llm/parse_output.py`
Парсит ответы модели:
- вытаскивает JSON
- нормализует probability
- приводит ответ к структурированному виду

---

## `src/consensus/similarity_utils.py`
Вспомогательная логика для embedding consensus:
- векторизация ответов LLM
- матрица сходств `N x N`
- centrality scores
- выбор central answer

---

## `src/consensus/consensus.py`
Реализует методы consensus:
- `frequency`
- `weighted`
- `embedding`
- `hybrid`

---

## `src/cot/generate_cot_cases.py`
Генерирует CoT для кейсов базы:
- complaint
- reasoning
- differential
- final hypothesis
- teaching points

**Результат:**
- `data/cot/cot_cases.json`

---

## `src/cot/cot_embeddings.py`
Строит embeddings для CoT-кейсов.

**Результат:**
- `data/cot/cot_embeddings.npy`
- `data/cot/cot_embeddings_meta.json`

---

## `src/cot/cot_fewshot.py`
Реализует CoT few-shot режим:
- retrieval по CoT-кейсам
- prompt с рассуждениями
- LLM runs
- consensus

---

## `src/dialogue/symptom_intake.py`
Pre-submission enhancement:
- определяет тип жалобы
- генерирует intake-вопросы
- собирает enriched query

---

## `src/dialogue/dialogue_state.py`
Хранит состояние диалога:
- исходный запрос
- intake answers
- asked questions
- answer history
- current consensus
- turn count

---

## `src/dialogue/adaptive_dialogue.py`
Adaptive multi-turn dialog:
- первичный анализ
- targeted questions
- обновление гипотез после каждого ответа
- stop criteria

---

## `src/evaluation/run_eval.py`
Скрипт для сравнения методов на тестовом датасете:
- baseline retrieval + prompting
- multiple LLM runs
- consensus comparison

**Поддерживаемые методы:**
- frequency
- weighted
- embedding
- hybrid

---

## `src/semantic_search.py`
Главный entrypoint проекта.

Поддерживает режимы:
- `baseline`
- `cot`
- `dialogue`

Также поддерживает методы consensus:
- `frequency`
- `weighted`
- `embedding`
- `hybrid`

---

# Используемые модели

## Summary / preprocess
- `IlyaGusev/rut5_base_sum_gazeta`

Используется для:
- краткого описания кейсов

## Embeddings
- `intfloat/multilingual-e5-large`

Используется для:
- retrieval кейсов
- embedding consensus
- CoT retrieval

## Основная LLM
- `deepseek/deepseek-r1-0528:free`

Используется через OpenRouter для:
- генерации гипотез
- CoT generation
- CoT few-shot
- dialogue analysis

---

# Основные команды

## Подготовка данных
```bash
python -m src.preprocessing.normalize_cases
python -m src.preprocessing.preprocess_llm
python -m src.preprocessing.finalize_processed_cases
python -m src.retrieval.vectorize_api_HF
