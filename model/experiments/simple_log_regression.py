# ============================================================================
# ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ ПО ПРИЗНАКАМ ИЗ DatasetBuilder
# ============================================================================
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.pipeline        import Pipeline
from sklearn.impute          import SimpleImputer
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
)
from sklearn.metrics         import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
)


def train_logreg(
    train: pd.DataFrame,
    features: list[float],
    target: int,
    model_name: str,
    penalty = None,
    C = 1,
    solver: str = "lbfgs",
    max_iter = 1000
):
    
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("lr",      LogisticRegression(
            C=C,
            solver=solver,
            max_iter=max_iter,
            penalty=penalty
        )),
    ])

    model.fit(train[features], train[target])

    return {
        "model" : model,
        "name" : model_name
    }

def train_dummy(
    train: pd.DataFrame,
    features: list[float],
    target: int,
    model_name: str,
):
    
    model = DummyClassifier(strategy="most_frequent")
    model.fit(train[features], train[target])

    return {
        "model" : model,
        "name" : model_name
    }


# ================================ пример ===================================
# if __name__ == "__main__":
#     # 1) сначала DatasetBuilder (лучше drop_bad_labels=True — метки уже валидны)
#     df = DatasetBuilder(
#         csv_path="randomResults.csv",
#         base_dir=".",
#         drop_bad_labels=True,
#         verbose=False,
#     ).build()

#     # 2) обучение
#     res = train_logreg(
#         df,
#         # feature_cols=["n_tools", "n_operations", "avg_time", ...],  # можно явно
#         class_weight="balanced",
#         C=1.0,
#         cv_folds=5,
#     )

#     print("\nКоэффициенты по классам (топ-3 по |весу| на класс):")
#     lr   = res["model"].named_steps["lr"]
#     cols = res["feature_cols"]
#     for ci, cls in enumerate(res["classes"]):
#         w = lr.coef_[ci]
#         top = np.argsort(-np.abs(w))[:3]
#         print(f"  {cls:12s}: " + ", ".join(
#             f"{cols[i]}={w[i]:+.3f}" for i in top
#         ))