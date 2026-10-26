# recommender_metrics.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ============================================================================
# Глобальное состояние модуля
# ============================================================================
results_table = pd.DataFrame()
_models_registry: dict[str, object] = {}     # name -> sklearn-модель

# Метрики, для которых «меньше = лучше» (штрафы)
_LOWER_IS_BETTER = {"train_fine", "test_fine"}


# ============================================================================
# Вспомогательные функции
# ============================================================================
def get_score(score: pd.DataFrame, class_nums, pred) -> float:
    """Суммарный штраф по предсказаниям.

    Для каждой строки i берётся значение score.loc[i, class_nums[pred[i]]],
    затем все значения суммируются.

    score : pd.DataFrame — строки соответствуют pred, колонки — имена классов
    class_nums : list[str] | np.ndarray — имена классов в порядке id (0..K-1)
    pred : np.ndarray[int] — предсказанные id классов
    """
    if score is None:
        return float("nan")

    class_nums_arr = np.asarray(class_nums, dtype=object)
    pred_arr = np.asarray(pred, dtype=int)
    names_per_row = class_nums_arr[pred_arr]

    # индексы колонок, соответствующие именам классов в каждой строке
    col_index = {name: i for i, name in enumerate(score.columns)}
    cols = np.array([col_index[n] for n in names_per_row])
    rows = np.arange(len(pred_arr))

    values = score.to_numpy()[rows, cols]
    return float(values.sum())


def _roc_auc(y_true, proba) -> float:
    """ROC AUC для бинарной или многоклассовой задачи. NaN, если не посчитать."""
    if proba is None:
        return float("nan")
    try:
        if proba.ndim == 2 and proba.shape[1] == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        return float(roc_auc_score(y_true, proba, multi_class="ovr"))
    except Exception:
        return float("nan")


# ============================================================================
# Оценка модели
# ============================================================================
def evaluate_model(
    model_info: dict,
    X_train, y_train,
    X_test, y_test,
    score_train: pd.DataFrame | None = None,
    score_test: pd.DataFrame | None = None,
    class_names: list | None = None,
    average: str = "weighted",
    append: bool = True,
) -> pd.DataFrame:
    """Обучает/оценивает модель и добавляет строку метрик в results_table.

    Модель должна быть УЖЕ обучена (train_gb возвращает обученную модель).
    Объект модели регистрируется в _models_registry, чтобы save_best_model
    мог её потом сохранить.

    Параметры
    ---------
    model_info : dict
        {"model": <sklearn-модель>, "name": "имя"}
    score_train, score_test : DataFrame
        Таблица штрафов (колонки — имена классов). Нужна для *_fine метрик.
    class_names : list[str]
        Имена классов в порядке id (совпадает с model.classes_).
    average : str
        'binary' | 'macro' | 'weighted' — для precision/recall/f1.
    append : bool
        True — дописать в глобальную results_table и вернуть её.
        False — вернуть только одну строку.
    """
    global results_table

    model = model_info["model"]
    name = model_info.get("name", type(model).__name__)

    # --- регистрируем модель, чтобы save_best_model её нашёл ---
    _models_registry[name] = model

    # --- предсказания ---
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # --- вероятности ---
    proba_train = None
    proba_test = None
    if hasattr(model, "predict_proba"):
        proba_train = model.predict_proba(X_train)
        proba_test = model.predict_proba(X_test)

    # --- метрики ---
    row = {
        "model": name,

        "acc_train": accuracy_score(y_train, y_train_pred),
        "acc_test": accuracy_score(y_test, y_test_pred),

        "precision_train": precision_score(
            y_train, y_train_pred, average=average, zero_division=0
        ),
        "precision_test": precision_score(
            y_test, y_test_pred, average=average, zero_division=0
        ),

        "recall_train": recall_score(
            y_train, y_train_pred, average=average, zero_division=0
        ),
        "recall_test": recall_score(
            y_test, y_test_pred, average=average, zero_division=0
        ),

        "f1_train": f1_score(
            y_train, y_train_pred, average=average, zero_division=0
        ),
        "f1_test": f1_score(
            y_test, y_test_pred, average=average, zero_division=0
        ),

        "roc_auc_train": _roc_auc(y_train, proba_train),
        "roc_auc_test": _roc_auc(y_test, proba_test),

        "train_fine": (
            get_score(score_train, class_names, y_train_pred)
            if score_train is not None else float("nan")
        ),
        "test_fine": (
            get_score(score_test, class_names, y_test_pred)
            if score_test is not None else float("nan")
        ),
    }

    row_df = pd.DataFrame([row])

    if append:
        results_table = pd.concat([results_table, row_df], ignore_index=True)
        return results_table

    return row_df


# ============================================================================
# Просмотр результатов
# ============================================================================
def show_results(sort_by: str = "f1_test", ascending: bool | None = None) -> pd.DataFrame:
    """Вернуть results_table, отсортированную по нужной колонке."""
    global results_table
    if results_table.empty:
        print("Таблица пуста — ещё не оценивали ни одной модели.")
        return results_table
    if ascending is None:
        ascending = sort_by in _LOWER_IS_BETTER
    return results_table.sort_values(sort_by, ascending=ascending).reset_index(drop=True)


# ============================================================================
# Сохранение лучшей модели
# ============================================================================
def save_best_model(
    path: str,
    metric: str = "f1_test",
    ascending: bool | None = None,
    features: list | None = None,
    class_names: list | None = None,
    extra: dict | None = None,
    verbose: bool = True,
) -> dict:
    """Находит лучшую модель по `metric` и сохраняет в `path` (joblib).

    Параметры
    ---------
    path : куда сохранять (.joblib)
    metric : колонка results_table, по которой выбираем лучшую
    ascending : None → автоопределение (штрафы — True, метрики — False)
    features : список колонок-признаков, кладётся в payload для инференса
    class_names : имена классов в порядке model.classes_
    extra : доп. поля в payload (например, {"base_dir": "..."})
    """
    global results_table

    if results_table.empty:
        raise RuntimeError("results_table пуст — сначала вызови evaluate_model()")
    if metric not in results_table.columns:
        raise KeyError(
            f"Метрики {metric!r} нет. Доступны: {results_table.columns.tolist()}"
        )

    if ascending is None:
        ascending = metric in _LOWER_IS_BETTER

    ordered = results_table.sort_values(metric, ascending=ascending).reset_index(drop=True)
    best = ordered.iloc[0]
    best_name = best["model"]

    if best_name not in _models_registry:
        raise KeyError(
            f"Модель {best_name!r} не зарегистрирована в _models_registry. "
            f"Убедись, что evaluate_model() вызывался после train_gb()."
        )

    model = _models_registry[best_name]

    payload = {
        "model": model,
        "name": best_name,
        "features": list(features) if features is not None else None,
        "class_names": list(class_names) if class_names is not None else None,
        "classes": [int(c) for c in getattr(model, "classes_", [])],
        "metric": metric,
        "metric_value": float(best[metric]),
        "metrics": best.to_dict(),
    }
    if extra:
        payload.update(extra)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out)

    if verbose:
        print(f"Сохранена лучшая модель: {best_name}")
        print(f"  метрика : {metric} = {best[metric]:.4f}  (ascending={ascending})")
        print(f"  файл    : {out.resolve()}")
        print(f"  features: {len(payload['features']) if payload['features'] else '—'}")
        print(f"  classes : {payload['classes']}")

    return payload


def get_best_model(metric: str = "f1_test", ascending: bool | None = None):
    """Вернуть (name, {"model": ..., "name": ...}) лучшей модели БЕЗ сохранения."""
    if results_table.empty:
        raise RuntimeError("results_table пуст")
    if ascending is None:
        ascending = metric in _LOWER_IS_BETTER
    best = results_table.sort_values(metric, ascending=ascending).iloc[0]
    name = best["model"]
    return name, {"model": _models_registry[name], "name": name}


# ============================================================================
# Утилиты
# ============================================================================
def reset_results() -> None:
    """Очистить таблицу и реестр моделей (удобно между экспериментами)."""
    global results_table, _models_registry
    results_table = pd.DataFrame()
    _models_registry = {}


def list_registered() -> list[str]:
    """Имена моделей, которые сейчас в реестре."""
    return list(_models_registry.keys())