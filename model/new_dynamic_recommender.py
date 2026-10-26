# recommend.py
import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from lib.feature_parser import ProblemFeatureExtractorWithExtras
from lib.dataset_builder import DatasetBuilder  # положи путь под свой проект

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

# ============ Единый источник правды (должен совпадать с C++) ============
CLASS_NAMES = [
    "Dummy", "Directive", "Stoppable",
    "Robin", "FineSorter", "Dependent",
]

DROP_COLS = [
    'file', 'file_resolved', 'file_exists', 'has_label',
    'best_class', 'best_class_id', 'best_score',
    'score_Dummy', 'score_Directive', 'score_Stoppable',
    'score_Robin', 'score_FineSorter', 'score_Dependent',
]
SCORE_COLUMNS = [
    'score_Dummy', 'score_Directive', 'score_Stoppable',
    'score_Robin', 'score_FineSorter', 'score_Dependent',
]
TARGET = "best_class_id"


# ============================ Обучение ====================================
def _features_list(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DROP_COLS]


def train_model(csv_path: str,
                base_dir: str,
                model_path: str,
                model_name: str = "HistGB"):
    """Строит датасет через DatasetBuilder, фильтрует, обучает через train_gb."""

    db = DatasetBuilder(csv_path, base_dir)
    data = db.build()

    if len(data) == 0:
        raise RuntimeError(f"DatasetBuilder вернул 0 строк из {csv_path}")

    # 1) оставить только строки с >= 3 различными score_*
    n_unique = data[SCORE_COLUMNS].nunique(axis=1, dropna=True)
    data = data[n_unique >= 3].reset_index(drop=True)

    # 2) отбросить строки без валидной метки
    data = data[data["has_label"]].reset_index(drop=True)
    data = data[data["best_class_id"] >= 0].reset_index(drop=True)

    if len(data) == 0:
        raise RuntimeError("После фильтрации не осталось валидных строк для обучения")

    features = _features_list(data)
    # выкинуть признаки, которые целиком NaN (иначе HistGB упадёт)
    features = [c for c in features if data[c].notna().any()]

    # 3) обучение — используем ровно твою train_gb
    result = train_gb(
        train=data,
        features=features,
        target=TARGET,
        model_name=model_name,
    )
    model = result["model"]

    payload = {
        "model": model,
        "features": features,
        "classes": [int(c) for c in model.classes_],   # id классов (0..5)
        "class_names": CLASS_NAMES,                    # имена (str)
        "model_name": result["name"],
    }
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, model_path)
    return payload


# ======================== Загрузка / ленивое обучение =====================
def load_or_train(model_path: str,
                  train_csv: str | None,
                  base_dir: str = "."):
    """Если модель есть — грузим. Иначе — обучаем на train_csv и сохраняем."""
    mp = Path(model_path)
    if mp.exists():
        return joblib.load(mp)

    if not train_csv or not Path(train_csv).exists():
        raise FileNotFoundError(
            f"Модель не найдена ({model_path}), а обучающий CSV не передан/не существует"
        )

    return train_model(train_csv, base_dir, model_path)


# ============================ Инференс ====================================
def predict_top(problem_path: str,
                model_path: str,
                top_n: int = 3,
                train_csv: str | None = None,
                base_dir: str = ".") -> list[str]:

    payload = load_or_train(model_path, train_csv, base_dir)
    model    = payload["model"]
    features = payload["features"]
    names    = payload["class_names"]
    classes  = payload["classes"]

    # 1) парсим задачу обёрткой
    ex = ProblemFeatureExtractorWithExtras()
    feats = ex.extract_from_file(problem_path) or {}
    df = pd.DataFrame([feats])

    # 2) выравниваем колонки по обучающему набору
    for f in features:
        if f not in df.columns:
            df[f] = 0
    X = df[features].fillna(0)

    # 3) вероятности по классам
    proba = model.predict_proba(X)[0]
    order = np.argsort(-proba)

    top_n = max(1, min(int(top_n), len(order)))
    return [names[int(classes[i])] for i in order[:top_n]]


# ============================== CLI =======================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["names_only", "recommend", "train"],
                   default="names_only")
    p.add_argument("--model",   required=True)
    p.add_argument("--problem", default=None)
    p.add_argument("--top_n",   type=int, default=3)
    p.add_argument("--train_csv", default=None,
                   help="CSV для обучения, если модели ещё нет")
    p.add_argument("--base_dir", default=".",
                   help="относительно чего в CSV считаются пути к task-файлам")
    args = p.parse_args()

    if args.mode == "train":
        if not args.train_csv:
            print("--train_csv обязателен для --mode train", file=sys.stderr)
            sys.exit(2)
        train_model(args.train_csv, args.base_dir, args.model)
        return

    if not args.problem:
        print("--problem обязателен", file=sys.stderr)
        sys.exit(2)

    try:
        top = predict_top(
            args.problem, args.model, args.top_n,
            train_csv=args.train_csv, base_dir=args.base_dir,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # ЕДИНСТВЕННЫЙ stdout — имена через пробел
    print(" ".join(top))


if __name__ == "__main__":
    main()

# python3 new_dynamic_recommender.py --mode names_only     --model saved_models/histgb.joblib     --train_csv ./experiments/for_ml_traning.csv     --base_dir /home/samoh/FactoryPlanningV2/model/    --problem ../../factory/dataset_directory_big/ml_remains/465677_50_70_0.300000_0.450000_0.300000_1000_100_.txt_itaration_3.txt --top_n 3
# python3 new_dynamic_recommender.py --mode names_only --model saved_models/histgb.joblib --problem ../../factory/dataset_directory_big/ml_remains/465677_50_70_0.300000_0.450000_0.300000_1000_100_.txt_itaration_3.txt --top_n 3
# python3 new_dynamic_recommender.py --mode names_only --model experiments/histgb1.joblib --problem ../../factory/dataset_directory_big/ml_remains/465677_50_70_0.300000_0.450000_0.300000_1000_100_.txt_itaration_3.txt --top_n 3