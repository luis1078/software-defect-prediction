"""
models.py
=========
Define los cuatro modelos de Stacking con optimización de hiperparámetros,
replicando la metodología de Daza Vergaray et al. (2025).

Configuraciones de Stacking
----------------------------
S1  : Random Forest + Naive Bayes           → meta: Logistic Regression
S2  : Gradient Boosting + Naive Bayes       → meta: Logistic Regression
S3A : Gradient Boosting + Random Forest     → meta: Logistic Regression  ★
S3B : Random Forest + Gradient Boosting + Naive Bayes → meta: LR

Técnicas de balanceo evaluadas
-------------------------------
none       : sin balanceo
oversample : Random Oversampling de la clase minoritaria
smote      : Synthetic Minority Over-sampling Technique

Autor  : [Tu nombre]
Curso  : Agentes Inteligentes — 2026-1
"""

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler, SMOTE

RANDOM_STATE = 42


def get_stacking_models() -> dict:
    """
    Instancia y devuelve las cuatro configuraciones de stacking.
    Cada llamada genera instancias frescas (sin entrenar).

    Returns
    -------
    dict : {'S1': modelo, 'S2': modelo, 'S3A': modelo, 'S3B': modelo}
    """
    def _rf():
        return RandomForestClassifier(
            n_estimators=100, max_depth=None,
            random_state=RANDOM_STATE, n_jobs=-1
        )
    def _gb():
        return GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1,
            max_depth=3, random_state=RANDOM_STATE
        )
    def _nb():
        return GaussianNB()

    def _meta():
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    # Stacking 1: Random Forest + Naive Bayes
    s1 = StackingClassifier(
        estimators=[("rf", _rf()), ("nb", _nb())],
        final_estimator=_meta(), cv=5, n_jobs=-1
    )
    # Stacking 2: Gradient Boosting + Naive Bayes
    s2 = StackingClassifier(
        estimators=[("gb", _gb()), ("nb", _nb())],
        final_estimator=_meta(), cv=5, n_jobs=-1
    )
    # Stacking 3A: Gradient Boosting + Random Forest ← modelo principal
    s3a = StackingClassifier(
        estimators=[("gb", _gb()), ("rf", _rf())],
        final_estimator=_meta(), cv=5, n_jobs=-1
    )
    # Stacking 3B: RF + GB + Naive Bayes (triple)
    s3b = StackingClassifier(
        estimators=[("rf", _rf()), ("gb", _gb()), ("nb", _nb())],
        final_estimator=_meta(), cv=5, n_jobs=-1
    )

    return {"S1": s1, "S2": s2, "S3A": s3a, "S3B": s3b}


def tune_model(model, X_train: np.ndarray, y_train: np.ndarray,
               verbose: bool = True) -> tuple:
    """
    Optimiza los hiperparámetros del meta-clasificador (Logistic Regression)
    mediante GridSearchCV con validación cruzada k-fold (k=5).

    Espacio de búsqueda
    -------------------
    - C (regularización): [0.01, 0.1, 1, 10]
    - solver            : ['lbfgs', 'liblinear']

    Returns
    -------
    best_estimator : modelo entrenado con los mejores parámetros
    best_params    : dict con los parámetros seleccionados
    """
    param_grid = {
        "final_estimator__C":      [0.01, 0.1, 1, 10],
        "final_estimator__solver": ["lbfgs", "liblinear"],
    }
    gs = GridSearchCV(
        model, param_grid, cv=5,
        scoring="roc_auc", n_jobs=-1,
        verbose=1 if verbose else 0
    )
    gs.fit(X_train, y_train)

    if verbose:
        print(f"  Mejores parámetros : {gs.best_params_}")
        print(f"  Mejor AUC-ROC (CV) : {gs.best_score_:.4f}")

    return gs.best_estimator_, gs.best_params_


def apply_balancing(X_train: np.ndarray, y_train: np.ndarray,
                    method: str = "none",
                    random_state: int = RANDOM_STATE) -> tuple:
    """
    Aplica una técnica de balanceo de clases al conjunto de entrenamiento.

    Parameters
    ----------
    method : 'none' | 'oversample' | 'smote'

    Returns
    -------
    X_bal, y_bal : arrays balanceados
    """
    if method == "none":
        return X_train, y_train

    elif method == "oversample":
        ros = RandomOverSampler(random_state=random_state)
        X_bal, y_bal = ros.fit_resample(X_train, y_train)
        print(f"  [Balanceo] Oversampling → {np.bincount(y_bal.astype(int))}")
        return X_bal, y_bal

    elif method == "smote":
        smote = SMOTE(random_state=random_state, k_neighbors=5)
        X_bal, y_bal = smote.fit_resample(X_train, y_train)
        print(f"  [Balanceo] SMOTE → {np.bincount(y_bal.astype(int))}")
        return X_bal, y_bal

    else:
        raise ValueError(
            f"Método desconocido: '{method}'. Use 'none', 'oversample' o 'smote'."
        )
