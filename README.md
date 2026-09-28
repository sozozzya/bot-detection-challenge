# Bot Detection

## Задача

Для каждой `cookie_id` необходимо оценить вероятность принадлежности к трафику известных сервисов автоматизированного сбора данных. `target=1` соответствует автоматизированному трафику, `target=0` — трафику, отнесённому к человеческому.

Основная метрика — **Precision при Recall >= 0.70**. В submission передаётся непрерывный `score` от 0 до 1; порог классификации не передаётся, поскольку его выбирает проверяющая система.

## Финальный pipeline

`solution.py` выполняет полный воспроизводимый pipeline:

1. загрузка `train.csv`, `test.csv`, `events.csv.gz`;
2. проверка схемы и временных ограничений;
3. фильтрация событий по индивидуальному суточному окну каждой cookie:
   `window_start_ts <= event_ts < window_end_ts`;
4. построение cookie-level признаков;
5. проверка покрытия train/test и выравнивание feature schema;
6. обучение финальной модели на всех размеченных train-данных;
7. получение `predict_proba(... )[:, 1]` для test;
8. проверка и сохранение `submission.csv`.

Запуск из корня проекта:

```bash
python solution.py
```

Результат:

```text
submission.csv
```

## Feature Engineering

Все поведенческие признаки строятся только из событий внутри соответствующего окна наблюдения.

### Metadata

- `cookie_age_hours` — возраст cookie на момент начала окна.

Признак `cookie_age_seconds`, использовавшийся во время исследования, в финальной матрице удалён как дублирующее представление возраста.

### Activity

- `n_events`;
- `n_unique_items`;
- `n_unique_categories`;
- `n_unique_locations`;
- `n_unique_search_queries`;
- `n_unique_event_types`;
- `n_unique_platforms`.

Они отражают интенсивность и разнообразие активности cookie.

### Event distribution

Для каждого наблюдаемого типа события строятся:

- количество событий `event_count__...`;
- доля событий `event_share__...`.

Используются типы:

- `item_view`;
- `search_results_view`;
- `photo_swipe`;
- `favorite_add`;
- `seller_page_view`;
- `contact_phone_show`;
- `login`;
- `contact_chat_open`;
- `contact_message_sent`.

### Temporal behaviour

Используются:

- `first_event_offset_seconds`;
- `last_event_offset_seconds`;
- `n_active_hours`;
- `mean_inter_event_seconds`;
- `median_inter_event_seconds`;
- `std_inter_event_seconds`;
- `min_inter_event_seconds`;
- `max_inter_event_seconds`;
- `active_span_seconds`;
- `events_per_active_hour`;
- `inter_event_burstiness`.

Эти признаки описывают интенсивность, длительность и регулярность поведения внутри окна.

## Что было исследовано

Исследование проводилось в `notebooks/03_model_experiments.ipynb`.

Последовательное добавление групп признаков дало следующие результаты на временной валидации:

| Эксперимент | Признаки | P@R>=0.70 | PR-AUC | ROC-AUC |
|---|---|---:|---:|---:|
| E00 | baseline | 0.102470 | 0.174609 | 0.607396 |
| E01 | activity | 0.210037 | 0.379974 | 0.790239 |
| E02 | event distribution | 0.242424 | 0.473567 | 0.810268 |
| E03 | temporal behaviour | 0.332264 | 0.606980 | 0.850155 |
| E04 | User-Agent + platform | 0.302217 | 0.603115 | 0.850368 |

User-Agent и platform features не вошли в финальную модель: на проведённой временной проверке они ухудшили основную метрику относительно E03.

### Baseline и сравнение моделей

В качестве baseline использовалась простая модель Random Forest на двух признаках из quickstart: `n_events` и `n_unique_items`. Она дала `P@R>=0.70 = 0.102470` на исходном временном validation split.

Далее сравнивались Logistic Regression, несколько конфигураций Random Forest, HistGradientBoosting и CatBoost. Для окончательного выбора использовалась walk-forward validation по датам начала окна: для каждой validation date модель обучалась только на более ранних датах.

Итоговые результаты walk-forward validation для основных кандидатов:

| Model | P@R>=0.70 pooled | Mean daily P@R | PR-AUC pooled | ROC-AUC pooled |
|---|---:|---:|---:|---:|
| **CatBoost** | **0.4575** | **0.5208** | 0.6695 | 0.8856 |
| HistGradientBoosting | 0.4427 | 0.4903 | **0.6807** | 0.8861 |
| Random Forest, `min_samples_leaf=10` | 0.4308 | 0.4103 | 0.5955 | 0.8721 |

На исходном validation split также были получены: Random Forest `leaf=3/5/1` — `0.430769/0.424812/0.422642`, Logistic Regression — `0.270531`, HistGradientBoosting — `0.441406`, CatBoost — `0.434109`. Эти значения использовались как диагностическая часть исследования; окончательный выбор делался по walk-forward validation.

Основной критерий выбора — pooled `P@R>=0.70`, поскольку именно эта метрика используется в задании. Поэтому финальной моделью выбран CatBoost.

Значение `0.4575` является результатом внутренней walk-forward validation на train и не является результатом скрытого теста.

## Финальная модель

Используется `CatBoostClassifier`:

```python
CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=False,
    thread_count=-1,
    allow_writing_files=False,
)
```

После выбора модели она обучается на **всём размеченном train**.

Порог не применяется: в `submission.csv` записываются непрерывные значения `predict_proba(... )[:, 1]`.

## Обработка пропусков, категориальных признаков и пустых окон

Event-level поля содержат пропуски. В финальном решении исходные категориальные поля не передаются в модель как сырые категориальные значения: они используются для построения числовых признаков разнообразия (`n_unique_*`) и распределения событий. Нормализация `platform` используется только для расчёта `n_unique_platforms`.

Cookie без событий не теряются: metadata-строка сохраняется, а отсутствующие event-derived признаки после left join заполняются нулями.

В финальной матрице проверяется отсутствие `NaN`, `inf` и `-inf`.

## Дубликаты событий

В исследовании были обнаружены повторяющиеся строки событий. В финальном pipeline не выполняется безусловный `drop_duplicates()`: событие рассматривается как элемент истории активности, а автоматическое удаление повторов могло бы изменить поведенческие счётчики. Используется та же логика, на которой получены validation results.

## Leakage control

События сначала ограничиваются индивидуальным интервалом:

```text
window_start_ts <= event_ts < window_end_ts
```

и только после этого агрегируются.

В модель не передаются:

- `target`;
- `cookie_id`;
- события после `window_end_ts`;
- test labels;
- внешние данные;
- ручные правила для отдельных cookie.

## Ограничения задания

Финальное решение:

- не использует LLM;
- не использует внешние API;
- не использует ручную разметку test;
- не содержит hardcoded scores или решений для конкретных `cookie_id`;
- использует только локальную open-source библиотеку CatBoost и стандартные Python ML/data библиотеки;
- фиксирует `random_seed=42`;
- содержит `requirements.txt` с версиями зависимостей.

## Reproducibility

Тестовое окружение, в котором подготовлен код:

- Python `3.13.5`;
- NumPy `2.3.5`;
- pandas `2.2.3`;
- scikit-learn `1.8.0`;
- CatBoost `1.2.8`.

Установка:

```bash
pip install -r requirements.txt
```

Запуск:

```bash
python solution.py
```

## Submission format

`submission.csv` содержит ровно две колонки:

```text
cookie_id,score
```

Для каждой cookie из `test.csv` присутствует ровно одна строка. Перед сохранением выполняются проверки количества строк, уникальности и полного совпадения `cookie_id` с `test.csv`, отсутствия пропусков/бесконечных значений и диапазона `[0, 1]` для `score`.

## Использованные open-source библиотеки

- **CatBoost** — open-source библиотека градиентного бустинга; используется как финальная классификационная модель.
- **pandas** — загрузка данных и feature engineering.
- **NumPy** — численные операции и проверки.

Внешние API и внешние модели в решении не используются.
