import numpy as np
import pandas as pd
from typing import Optional, Union
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.ensemble import HistGradientBoostingClassifier
from typing import Optional


def train_gb(
    train: pd.DataFrame,
    features: list[str],
    target: str,
    model_name: str,
    learning_rate: float = 0.05,
    max_iter: int = 500,
    max_leaf_nodes: int = 31,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 20,
    l2_regularization: float = 0.0,
    early_stopping: bool = True,
    validation_fraction: float = 0.1,
    n_iter_no_change: int = 20,
    random_state: int = 42,
):
    model = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        random_state=random_state,
    )

    model.fit(train[features], train[target])

    return {
        "model": model,
        "name": model_name,
    }

def train_xgb(
    train: pd.DataFrame,
    features: list[str],
    target: str,
    model_name: str,
    # --- core ---
    n_estimators: int = 2000,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    min_child_weight: int = 3,
    # --- регуляризация ---
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    gamma: float = 0.1,
    reg_alpha: float = 1.0,
    reg_lambda: float = 1.5,
    # --- early stopping ---
    early_stopping_rounds: int = 30,
    eval_fraction: float = 0.15,
    # --- балансировка классов ---
    class_weight: Union[str, dict, None] = "balanced",
    sample_weight: Optional[np.ndarray] = None,
    # --- прочее ---
    random_state: int = 42,
    n_jobs: int = -1,
    eval_set: Optional[tuple] = None,
):
    """
    Обучает XGBClassifier с early stopping и балансировкой классов.

    class_weight : "balanced" | dict | None
        Для мультикласса — через sample_weight.
        Для бинарки "balanced" эквивалентно scale_pos_weight = neg/pos.

    eval_set : (X_val, y_val) | None
        Свой валидационный сет. Если None — отрезаем eval_fraction от train.
    """
    y = train[target]
    n_classes = y.nunique()

    # --- 1. Веса объектов ---
    if sample_weight is None and class_weight is not None:
        sample_weight = compute_sample_weight(class_weight=class_weight, y=y)

    # --- 2. eval_set для early stopping ---
    if eval_set is None:
        from sklearn.model_selection import train_test_split
        strat = y if min(np.bincount(pd.factorize(y)[0])) >= 2 else None
        X_tr, X_val, y_tr, y_val = train_test_split(
            train[features], y,
            test_size=eval_fraction,
            random_state=random_state,
            stratify=strat,
        )
        sw_tr = (compute_sample_weight(class_weight=class_weight, y=y_tr)
                 if class_weight is not None else None)
        fit_X, fit_y, fit_sw = X_tr, y_tr, sw_tr
        eval_X, eval_y = X_val, y_val
    else:
        eval_X, eval_y = eval_set
        fit_X, fit_y, fit_sw = train[features], y, sample_weight
        eval_X = eval_X[features] if isinstance(eval_X, pd.DataFrame) else eval_X

    # --- 3. Модель ---
    objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
    eval_metric = "mlogloss" if n_classes > 2 else "logloss"

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective=objective,
        eval_metric=eval_metric,
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method="hist",            # быстро на средних данных, ест NaN
        verbosity=0,
    )

    # --- 4. fit ---
    model.fit(
        fit_X, fit_y,
        sample_weight=fit_sw,
        eval_set=[(eval_X, eval_y)],
        verbose=False,
    )

    # --- 5. Диагностика ---
    if hasattr(model, "best_iteration"):
        print(f"[{model_name}] early stopping: "
              f"best_iteration={model.best_iteration} / {n_estimators}")

    return {
        "model": model,
        "name": model_name,
    }