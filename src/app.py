"""
app.py
======
Interfaz web Streamlit para la predicción de defectos de software en
tiempo real, usando el modelo entrenado con el dataset real de 37 proyectos
open source (métricas CK — Chidamber & Kemerer).

Uso
---
    streamlit run src/app.py

Requisito previo
----------------
    python src/train.py   →  genera outputs/best_model.pkl, scaler.pkl, features.pkl

Autor  : [Tu nombre]
Curso  : Agentes Inteligentes — 2026-1
"""

import os, pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ── Descripción de cada métrica CK para el formulario ────────────────────────
FEATURE_META = {
    "wmc":    ("WMC — Weighted Methods per Class",       0,  500,  10),
    "dit":    ("DIT — Depth of Inheritance Tree",        0,   20,   2),
    "noc":    ("NOC — Number of Children",               0,  100,   1),
    "cbo":    ("CBO — Coupling Between Objects",         0,  200,  10),
    "rfc":    ("RFC — Response for a Class",             0,  800,  30),
    "lcom":   ("LCOM — Lack of Cohesion of Methods",    0, 5000,  50),
    "ca":     ("CA — Afferent Couplings",                0,  200,   5),
    "ce":     ("CE — Efferent Couplings",                0,  200,   8),
    "npm":    ("NPM — Number of Public Methods",         0,  300,   8),
    "lcom3":  ("LCOM3 — Lack of Cohesion (variant 3)",  0.0,  2.0, 0.5),
    "loc":    ("LOC — Lines of Code",                    0, 5000, 100),
    "dam":    ("DAM — Data Access Metric",               0.0,  1.0, 0.5),
    "moa":    ("MOA — Measure of Aggregation",           0,   50,   1),
    "mfa":    ("MFA — Functional Abstraction",           0.0,  1.0, 0.3),
    "cam":    ("CAM — Cohesion Among Methods",           0.0,  1.0, 0.5),
    "ic":     ("IC — Inheritance Coupling",              0,   50,   1),
    "cbm":    ("CBM — Coupling Between Methods",         0,   50,   2),
    "amc":    ("AMC — Average Method Complexity",        0.0,200.0,10.0),
    "max_cc": ("max_cc — Max Cyclomatic Complexity",     0,  100,   3),
    "avg_cc": ("avg_cc — Avg Cyclomatic Complexity",     0.0, 50.0, 1.5),
}


@st.cache_resource
def load_artifacts():
    """Carga el modelo, scaler y features desde disco (con caché)."""
    paths = {k: os.path.join(OUTPUT_DIR, f"{k}.pkl")
             for k in ["best_model", "scaler", "features"]}
    if not all(os.path.exists(p) for p in paths.values()):
        return None, None, None
    with open(paths["best_model"], "rb") as f: model    = pickle.load(f)
    with open(paths["scaler"],     "rb") as f: scaler   = pickle.load(f)
    with open(paths["features"],   "rb") as f: features = pickle.load(f)
    return model, scaler, features


def make_gauge(prob: float) -> plt.Figure:
    """Gauge semicircular con la probabilidad de defecto."""
    fig, ax = plt.subplots(figsize=(4, 2.6), subplot_kw={"aspect": "equal"})
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.1, 1.2)
    ax.axis("off")

    theta_bg = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta_bg), np.sin(theta_bg), lw=20, color="#E0E0E0",
            solid_capstyle="round")

    color = "#27AE60" if prob < 0.4 else "#E67E22" if prob < 0.7 else "#E74C3C"
    theta_v = np.linspace(np.pi, np.pi - prob * np.pi, 200)
    ax.plot(np.cos(theta_v), np.sin(theta_v), lw=20, color=color,
            solid_capstyle="round")

    ax.text(0, 0.38, f"{prob*100:.1f}%", ha="center", va="center",
            fontsize=22, fontweight="bold", color=color)
    ax.text(0, 0.08, "Prob. de defecto", ha="center", va="center",
            fontsize=9, color="#555555")
    return fig


def main():
    st.set_page_config(
        page_title="Predictor de Defectos de Software",
        page_icon=None, layout="wide"
    )

    st.title("Predictor de Defectos de Software")
    st.markdown(
        "**Replicación de Daza Vergaray et al. (2025)** — Stacking con Hiperparámetros  \n"
        "Dataset: 37 proyectos open source · 12,455 módulos · Métricas CK (Chidamber & Kemerer)"
    )
    st.divider()

    model, scaler, features = load_artifacts()
    if model is None:
        st.error("Modelo no encontrado. Ejecute primero: `python src/train.py`")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("Acerca del modelo")
        st.markdown("""
**Modelo:** Stacking 3A + Sin balanceo  
**Nivel 0:** Gradient Boosting + Random Forest  
**Nivel 1 (meta):** Logistic Regression  
**Optimización:** GridSearchCV k-fold (k=5)

---
**Dataset:**  
37 proyectos open source (Apache Ant, Camel, Log4j, Lucene, POI, Xerces...)  
11,688 módulos (tras limpieza) · 20 métricas CK · variable objetivo: `bug`

---
**Resultados obtenidos (mejor AUC-ROC):**
- Accuracy:  71.64%
- F1-Score:  40.64%
- AUC-ROC:   70.32%

**Mejor F1-Score:** S3A + SMOTE — 51.73%
        """)
        st.divider()
        st.caption("Agentes Inteligentes — USIL 2026-1")

    # Formulario de métricas CK
    st.subheader("Metricas CK del Modulo de Software")
    st.caption("Ingrese las métricas estáticas del módulo que desea analizar.")

    cols = st.columns(4)
    input_values = {}

    for idx, feat in enumerate(features):
        col = cols[idx % 4]
        label, mn, mx, default = FEATURE_META.get(feat, (feat, 0.0, 1000.0, 0.0))
        with col:
            if isinstance(default, float):
                input_values[feat] = st.number_input(
                    label, min_value=float(mn), max_value=float(mx),
                    value=float(default), step=float((mx - mn) / 100), key=feat
                )
            else:
                input_values[feat] = st.number_input(
                    label, min_value=int(mn), max_value=int(mx),
                    value=int(default), step=1, key=feat
                )

    st.divider()

    btn_col, _, res_col = st.columns([1, 0.3, 2.5])
    with btn_col:
        predict = st.button("Predecir Defecto", use_container_width=True, type="primary")

    if predict:
        X_in = np.array([[input_values[f] for f in features]])
        X_sc = scaler.transform(X_in)
        pred = model.predict(X_sc)[0]
        prob = model.predict_proba(X_sc)[0][1]

        with res_col:
            if pred == 1:
                st.error("**DEFECTO DETECTADO** — El módulo presenta alta probabilidad de contener defectos.")
            else:
                st.success("**SIN DEFECTO** — El módulo no presenta indicadores de defecto.")

            st.pyplot(make_gauge(prob), use_container_width=False)
            st.markdown(f"""
| Métrica | Valor |
|---|---|
| Probabilidad de defecto | **{prob*100:.2f}%** |
| Predicción | **{"Con defecto (1)" if pred==1 else "Sin defecto (0)"}** |
| Umbral de decisión | 0.50 |
            """)

        with st.expander("Metricas ingresadas"):
            st.dataframe(
                pd.DataFrame({"Métrica": list(input_values.keys()),
                              "Valor": list(input_values.values())}),
                use_container_width=True, hide_index=True
            )

    # Resultados del entrenamiento
    st.divider()
    st.subheader("Resultados del Entrenamiento")

    results_path = os.path.join(OUTPUT_DIR, "resultados_modelos.csv")
    if os.path.exists(results_path):
        df_res = pd.read_csv(results_path)
        st.dataframe(
            df_res.style.highlight_max(
                subset=["Accuracy", "F1-Score", "AUC-ROC"], color="#d4edda"
            ),
            use_container_width=True, hide_index=True
        )
        imgs = st.columns(2)
        for i, img in enumerate(["05_roc_curves.png", "06_metricas_comparacion.png",
                                  "07_confusion_matrix_best.png", "01_distribucion_por_proyecto.png"]):
            p = os.path.join(OUTPUT_DIR, img)
            if os.path.exists(p):
                with imgs[i % 2]:
                    st.image(p, use_container_width=True)
    else:
        st.info("Ejecute `python src/train.py` para generar los resultados.")


if __name__ == "__main__":
    main()