# Predicción de Defectos de Software — Stacking con Hiperparámetros

**Curso:** Agentes Inteligentes — Universidad San Ignacio de Loyola (2026-1)  
**Carrera:** Ingeniería de Software  
**Referencia:** Daza Vergaray et al. (2025). *Software defect prediction based on a multiclassifier with hyperparameters*. Results in Engineering.

---

## Dataset

37 proyectos open source reales del repositorio PROMISE:

| Familia | Proyectos |
|---|---|
| Apache Ant | ant-1.3, ant-1.4, ant-1.5, ant-1.6, ant-1.7 |
| Apache Camel | camel-1.0, camel-1.2, camel-1.4, camel-1.6 |
| Apache Ivy | ivy-1.1, ivy-1.4, ivy-2.0 |
| jEdit | jedit-3.2, jedit-4.0, jedit-4.1, jedit-4.2, jedit-4.3 |
| Log4j | log4j-1.0, log4j-1.1, log4j-1.2 |
| Apache Lucene | lucene-2.0, lucene-2.2, lucene-2.4 |
| Apache POI | poi-1.5, poi-2.0, poi-2.5, poi-3.0 |
| Apache Synapse | synapse-1.0, synapse-1.1, synapse-1.2 |
| Apache Velocity | velocity-1.4, velocity-1.5, velocity-1.6 |
| Apache Xerces | xerces-1.2, xerces-1.3, xerces-1.4, xerces-init |

**Total consolidado:** 12,455 módulos · 20 métricas CK · variable objetivo: `bug`

---

## Métricas (features)

| Métrica | Descripción |
|---|---|
| wmc | Weighted Methods per Class |
| dit | Depth of Inheritance Tree |
| noc | Number of Children |
| cbo | Coupling Between Objects |
| rfc | Response for a Class |
| lcom | Lack of Cohesion of Methods |
| ca / ce | Afferent / Efferent Couplings |
| npm | Number of Public Methods |
| loc | Lines of Code |
| max_cc / avg_cc | Max / Average Cyclomatic Complexity |
| … | (20 métricas CK en total) |

---

## Modelos implementados

| ID | Nivel 0 (base) | Balanceo | Descripción |
|----|---|---|---|
| S1 | Random Forest + Naive Bayes | — | baseline |
| S2 | Gradient Boosting + Naive Bayes | — | — |
| **S3A** | **Gradient Boosting + Random Forest** | **Oversampling** | **modelo principal** |
| S3B | RF + GB + Naive Bayes | Oversampling | triple base |

---

## Estructura del proyecto

```
software-defect-prediction/
├── data/                        # 37 CSVs del dataset PROMISE
│   ├── ant-1.3.csv
│   ├── camel-1.0.csv
│   └── ...
├── outputs/                     # Figuras, resultados y modelos .pkl
├── src/
│   ├── preprocessing.py         # Carga, EDA y split
│   ├── models.py                # Modelos de stacking y balanceo
│   ├── train.py                 # Pipeline completo
│   └── app.py                   # Interfaz web Streamlit
├── requirements.txt
└── README.md
```

---

## Instalación y uso

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/software-defect-prediction.git
cd software-defect-prediction

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Entrenar los modelos (~5-15 min)
python src/train.py

# 4. Lanzar la interfaz web
streamlit run src/app.py
```

---

## Tecnologías

Python 3.10+ · scikit-learn · imbalanced-learn · pandas · numpy · matplotlib · seaborn · Streamlit
