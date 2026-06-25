"""
train.py
========
Pipeline completo de entrenamiento, optimización y evaluación de los
cuatro modelos de Stacking para la predicción de defectos de software.

Flujo de ejecución
------------------
1. Cargar y consolidar los 37 CSVs del dataset real
2. Análisis exploratorio (EDA) → guarda figuras en outputs/
3. Dividir en train/test (80/20 estratificado)
4. Por cada experimento (modelo × balanceo):
   a. Aplicar balanceo de clases
   b. Entrenar + optimizar hiperparámetros (GridSearchCV)
   c. Evaluar sobre el conjunto de prueba
5. Generar gráficas comparativas (ROC, barras de métricas, conf. matrix)
6. Exportar tabla de resultados a CSV
7. Serializar el mejor modelo para uso en la app Streamlit

Experimentos evaluados (7 en total)
-------------------------------------
S1  + Sin balanceo
S2  + Sin balanceo
S3A + Sin balanceo
S3A + Oversampling   ← modelo principal (Daza Vergaray et al., 2025)
S3A + SMOTE
S3B + Sin balanceo
S3B + Oversampling

Autor  : [Tu nombre]
Curso  : Agentes Inteligentes — 2026-1
"""

import os, sys, time, warnings, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve,
)

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import load_and_clean, run_eda, split_data, OUTPUT_DIR, DATA_DIR
from models import get_stacking_models, tune_model, apply_balancing

# ── Experimentos ──────────────────────────────────────────────────────────────
EXPERIMENTS = [
    ("S1  + Sin balanceo",  "S1",  "none"),
    ("S2  + Sin balanceo",  "S2",  "none"),
    ("S3A + Sin balanceo",  "S3A", "none"),
    ("S3A + Oversampling",  "S3A", "oversample"),  # ← modelo principal
    ("S3A + SMOTE",         "S3A", "smote"),
    ("S3B + Sin balanceo",  "S3B", "none"),
    ("S3B + Oversampling",  "S3B", "oversample"),
]


# ── Evaluación ────────────────────────────────────────────────────────────────

def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Calcula métricas de clasificación sobre el conjunto de prueba."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "F1-Score":  round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        "AUC-ROC":   round(roc_auc_score(y_test, y_proba) * 100, 2),
    }


# ── Visualizaciones ───────────────────────────────────────────────────────────

def plot_roc_curves(roc_data: list, output_dir: str) -> None:
    """Curvas ROC comparativas de todos los experimentos."""
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_data)))

    for (name, fpr, tpr, auc), color in zip(roc_data, colors):
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={auc:.2f}%)")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Clasificador aleatorio")
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=12)
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=12)
    ax.set_title("Figura 5\nCurvas ROC — Comparación de Modelos",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_roc_curves.png"), dpi=150)
    plt.close()
    print("[VIZ] Figura 5 guardada: curvas ROC.")


def plot_metrics_comparison(df_results: pd.DataFrame, output_dir: str) -> None:
    """Gráfico de barras agrupadas comparando métricas de todos los modelos."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    x = np.arange(len(df_results))
    width = 0.15
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i * width, df_results[metric], width,
               label=metric, color=color, edgecolor="white", alpha=0.9)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df_results["Modelo"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Porcentaje (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Figura 6\nComparación de Métricas por Modelo",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.axhline(y=80, color="gray", linestyle="--", lw=1, alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_metricas_comparacion.png"), dpi=150)
    plt.close()
    print("[VIZ] Figura 6 guardada: comparación de métricas.")


def plot_confusion_matrix(model, X_test, y_test, name, output_dir) -> None:
    """Matriz de confusión del mejor modelo."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Sin defecto (0)", "Con defecto (1)"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Figura 7\nMatriz de Confusión — {name}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "07_confusion_matrix_best.png"), dpi=150)
    plt.close()
    print("[VIZ] Figura 7 guardada: matriz de confusión.")


# ── Pipeline principal ────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("PASO 1: Carga y consolidación del dataset")
    print("="*60)
    df = load_and_clean(DATA_DIR)

    print("\n" + "="*60)
    print("PASO 2: Análisis Exploratorio (EDA)")
    print("="*60)
    run_eda(df, OUTPUT_DIR)

    print("\n" + "="*60)
    print("PASO 3: División de datos (80/20 estratificado)")
    print("="*60)
    X_train, X_test, y_train, y_test, scaler, features = split_data(df)

    print("\n" + "="*60)
    print("PASO 4-5: Entrenamiento, optimización y evaluación")
    print("="*60)

    all_results = []
    roc_data    = []
    best_model  = None
    best_auc    = 0
    best_name   = ""

    for exp_name, model_key, balance_method in EXPERIMENTS:
        print(f"\n── {exp_name} ──")

        model = get_stacking_models()[model_key]
        X_bal, y_bal = apply_balancing(X_train, y_train, method=balance_method)

        t0 = time.time()
        best, params = tune_model(model, X_bal, y_bal, verbose=False)
        elapsed = time.time() - t0

        metrics = evaluate(best, X_test, y_test)
        metrics["Modelo"]    = exp_name
        metrics["Tiempo(s)"] = round(elapsed, 1)
        all_results.append(metrics)

        y_proba = best.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data.append((exp_name, fpr, tpr, metrics["AUC-ROC"]))

        print(f"  Accuracy={metrics['Accuracy']}%  "
              f"F1={metrics['F1-Score']}%  "
              f"AUC-ROC={metrics['AUC-ROC']}%  "
              f"({elapsed:.0f}s)")

        if metrics["AUC-ROC"] > best_auc:
            best_auc   = metrics["AUC-ROC"]
            best_model = best
            best_name  = exp_name

    print("\n" + "="*60)
    print("PASO 6: Visualizaciones")
    print("="*60)
    cols_order = ["Modelo", "Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC", "Tiempo(s)"]
    df_results = pd.DataFrame(all_results)[cols_order]

    plot_roc_curves(roc_data, OUTPUT_DIR)
    plot_metrics_comparison(df_results, OUTPUT_DIR)
    plot_confusion_matrix(best_model, X_test, y_test, best_name, OUTPUT_DIR)

    print("\n" + "="*60)
    print("PASO 7: Exportar resultados y serializar modelo")
    print("="*60)
    results_path = os.path.join(OUTPUT_DIR, "resultados_modelos.csv")
    df_results.to_csv(results_path, index=False)
    print(f"[OK] Resultados exportados → {results_path}")

    with open(os.path.join(OUTPUT_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(OUTPUT_DIR, "features.pkl"), "wb") as f:
        pickle.dump(features, f)
    print("[OK] Modelo, scaler y features serializados en outputs/")

    print("\n" + "="*60)
    print("RESULTADOS FINALES")
    print("="*60)
    print(df_results.to_string(index=False))
    print(f"\n★  Mejor modelo : {best_name}")
    print(f"   AUC-ROC      : {best_auc}%")


if __name__ == "__main__":
    main()