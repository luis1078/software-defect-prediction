"""
preprocessing.py
================
Carga, limpieza, análisis exploratorio (EDA) y preparación del dataset
para la predicción de defectos de software.

Dataset real: 37 proyectos open source (Apache Ant, Camel, Log4j, etc.)
              provenientes del repositorio PROMISE/Kaggle.
              Cada CSV contiene 20 métricas CK (Chidamber & Kemerer) y
              la variable objetivo 'bug' (0 = sin defecto, ≥1 = con defecto).

Autor  : [Tu nombre]
Curso  : Agentes Inteligentes — 2026-1
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ── Rutas ────────────────────────────────────────────────────────────────────
# Estructura esperada:
#   AgentesInteligentes/
#   ├── data/   ← CSVs del dataset
#   ├── src/    ← este archivo
#   └── outputs/
#
# os.path.realpath garantiza ruta absoluta correcta en Windows y Linux
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
TARGET_COL = "bug"

# Métricas CK: las 20 columnas predictoras presentes en todos los CSVs
CK_FEATURES = [
    "wmc",    # Weighted Methods per Class
    "dit",    # Depth of Inheritance Tree
    "noc",    # Number of Children
    "cbo",    # Coupling Between Objects
    "rfc",    # Response for a Class
    "lcom",   # Lack of Cohesion of Methods
    "ca",     # Afferent Couplings
    "ce",     # Efferent Couplings
    "npm",    # Number of Public Methods
    "lcom3",  # Lack of Cohesion (variant 3)
    "loc",    # Lines of Code
    "dam",    # Data Access Metric
    "moa",    # Measure of Aggregation
    "mfa",    # Measure of Functional Abstraction
    "cam",    # Cohesion Among Methods
    "ic",     # Inheritance Coupling
    "cbm",    # Coupling Between Methods
    "amc",    # Average Method Complexity
    "max_cc", # Maximum Cyclomatic Complexity
    "avg_cc", # Average Cyclomatic Complexity
]


# ────────────────────────────────────────────────────────────────────────────
# 1. CARGA Y CONSOLIDACIÓN
# ────────────────────────────────────────────────────────────────────────────

def load_and_clean(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Lee todos los CSVs del directorio, consolida en un único DataFrame,
    binariza la variable objetivo y aplica limpieza básica.

    Pasos:
    1. Leer cada CSV y agregar columna 'project' con el nombre del archivo.
    2. Concatenar todos los DataFrames.
    3. Binarizar 'bug': cualquier valor > 0 se convierte en 1.
    4. Eliminar duplicados.
    5. Imputar valores nulos con la mediana de cada columna.

    Returns
    -------
    df : pd.DataFrame limpio y consolidado
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos CSV en: {data_dir}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df["project"] = os.path.basename(f).replace(".csv", "")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] {len(csv_files)} archivos cargados → {combined.shape[0]:,} filas × {combined.shape[1]} columnas")

    # Binarizar variable objetivo: bug > 0 → 1
    combined[TARGET_COL] = (combined[TARGET_COL] > 0).astype(int)

    # Eliminar duplicados
    antes = len(combined)
    combined.drop_duplicates(inplace=True)
    print(f"[INFO] Duplicados eliminados: {antes - len(combined):,}")

    # Imputar nulos con mediana (por seguridad, el dataset no tiene nulos)
    nulos = combined[CK_FEATURES].isnull().sum().sum()
    if nulos > 0:
        combined[CK_FEATURES] = combined[CK_FEATURES].fillna(
            combined[CK_FEATURES].median()
        )
        print(f"[INFO] Valores nulos imputados: {nulos}")
    else:
        print("[INFO] Sin valores nulos.")

    print(f"[INFO] Shape final: {combined.shape}")
    print(f"[INFO] Distribución — Sin defecto: {(combined[TARGET_COL]==0).sum():,} | "
          f"Con defecto: {(combined[TARGET_COL]==1).sum():,} "
          f"({combined[TARGET_COL].mean()*100:.1f}%)")
    return combined


# ────────────────────────────────────────────────────────────────────────────
# 2. ANÁLISIS EXPLORATORIO (EDA)
# ────────────────────────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> None:
    """
    Genera y guarda cuatro figuras de análisis exploratorio:
    1. Distribución de la variable objetivo por proyecto.
    2. Histogramas de las 6 métricas más relevantes.
    3. Mapa de calor de correlaciones (solo features CK).
    4. Boxplots de métricas clave por clase (bug 0 vs 1).
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Fig 1: Distribución de clases por proyecto ───────────────────────────
    bug_by_proj = df.groupby("project")[TARGET_COL].agg(
        total="count",
        defectuosos="sum"
    ).assign(pct=lambda x: x["defectuosos"] / x["total"] * 100).sort_values("pct", ascending=False)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(bug_by_proj))
    ax.bar(x, bug_by_proj["total"] - bug_by_proj["defectuosos"],
           label="Sin defecto", color="#4C72B0", edgecolor="white")
    ax.bar(x, bug_by_proj["defectuosos"], bottom=bug_by_proj["total"] - bug_by_proj["defectuosos"],
           label="Con defecto", color="#DD8452", edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels(bug_by_proj.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Número de módulos")
    ax.set_title("Figura 1\nDistribución de Defectos por Proyecto", fontsize=13, fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_distribucion_por_proyecto.png"), dpi=150)
    plt.close()
    print("[EDA] Figura 1 guardada: distribución por proyecto.")

    # ── Fig 2: Histogramas de métricas principales ───────────────────────────
    top6 = ["wmc", "cbo", "rfc", "lcom", "loc", "avg_cc"]
    labels = {
        "wmc": "WMC — Weighted Methods per Class",
        "cbo": "CBO — Coupling Between Objects",
        "rfc": "RFC — Response for a Class",
        "lcom": "LCOM — Lack of Cohesion",
        "loc": "LOC — Lines of Code",
        "avg_cc": "avg_cc — Cyclomatic Complexity (avg)"
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flat, top6):
        ax.hist(df[col].clip(upper=df[col].quantile(0.99)),
                bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
        ax.set_title(labels[col], fontsize=9, fontweight="bold")
        ax.set_xlabel("Valor")
        ax.set_ylabel("Frecuencia")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Figura 2\nDistribución de Métricas CK Principales", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_histogramas_metricas.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[EDA] Figura 2 guardada: histogramas.")

    # ── Fig 3: Mapa de correlación ───────────────────────────────────────────
    corr_cols = CK_FEATURES + [TARGET_COL]
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(13, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.3, ax=ax, vmin=-1, vmax=1,
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
    ax.set_title("Figura 3\nMatriz de Correlación — Métricas CK", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_correlacion.png"), dpi=150)
    plt.close()
    print("[EDA] Figura 3 guardada: correlación.")

    # ── Fig 4: Boxplots por clase ─────────────────────────────────────────────
    box_cols = ["wmc", "cbo", "loc", "avg_cc"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    for ax, col in zip(axes, box_cols):
        data_0 = df[df[TARGET_COL] == 0][col].clip(upper=df[col].quantile(0.95))
        data_1 = df[df[TARGET_COL] == 1][col].clip(upper=df[col].quantile(0.95))
        ax.boxplot([data_0, data_1],
                   labels=["Sin defecto\n(0)", "Con defecto\n(1)"],
                   boxprops=dict(color="#4C72B0"),
                   medianprops=dict(color="#DD8452", linewidth=2),
                   whiskerprops=dict(color="#4C72B0"),
                   capprops=dict(color="#4C72B0"),
                   flierprops=dict(marker="o", color="#4C72B0", alpha=0.3, markersize=3))
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Figura 4\nDistribución de Métricas Clave por Clase", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_boxplots_por_clase.png"), dpi=150)
    plt.close()
    print("[EDA] Figura 4 guardada: boxplots.")


# ────────────────────────────────────────────────────────────────────────────
# 3. DIVISIÓN DE DATOS
# ────────────────────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame, test_size: float = 0.20,
               random_state: int = 42) -> tuple:
    """
    Separa features y target, escala con StandardScaler y divide
    en conjuntos de entrenamiento (80%) y prueba (20%) estratificados.

    El scaler se ajusta SOLO con datos de entrenamiento para evitar
    data leakage hacia el conjunto de prueba.

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays NumPy
    scaler                           : StandardScaler ajustado
    feature_names                    : lista de nombres de columnas
    """
    X = df[CK_FEATURES]
    y = df[TARGET_COL]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)   # fit + transform en train
    X_test  = scaler.transform(X_test_raw)         # solo transform en test

    print(f"[INFO] Train: {X_train.shape[0]:,} muestras | Test: {X_test.shape[0]:,} muestras")
    print(f"[INFO] % defectos — Train: {y_train.mean()*100:.1f}% | Test: {y_test.mean()*100:.1f}%")

    return X_train, X_test, y_train, y_test, scaler, CK_FEATURES


# ────────────────────────────────────────────────────────────────────────────
# Ejecución directa
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_and_clean()
    run_eda(df)
    X_train, X_test, y_train, y_test, scaler, features = split_data(df)
    print("\n[OK] Preprocesamiento completado.")
    print(f"     Features ({len(features)}): {features}")
    print(f"     X_train: {X_train.shape} | X_test: {X_test.shape}")