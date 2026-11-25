import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42


def normalize_target(series: pd.Series) -> pd.Series:
    if series.dtype == object or str(series.dtype).startswith("string"):
        mapped = series.astype(str).str.strip().str.lower().map({
            "1": 1, "true": 1, "yes": 1, "churn": 1, "churnd": 1, "y": 1,
            "0": 0, "false": 0, "no": 0, "not churn": 0, "retained": 0, "n": 0,
        })
        if mapped.notna().all():
            return mapped.astype(int)
    return pd.to_numeric(series, errors="raise").astype(int)


def main() -> None:
    df = pd.read_csv("customer_churn_basic_dataset.csv")

    print("=== 1. ANÁLISIS INICIAL ===")
    print(f"Filas y columnas: {df.shape[0]} filas, {df.shape[1]} columnas")
    print("\nTipos de datos:")
    print(df.dtypes)
    print("\nValores faltantes por columna:")
    print(df.isnull().sum())

    y = normalize_target(df["churn"])
    print("\nDistribución de churn (0 = no abandona, 1 = abandona):")
    print(y.value_counts())
    print(f"Tasa de abandono: {y.mean():.2%}")

    # Excluir 'churn' (objetivo) y 'customer_id' (identificador, no predictora)
    drop_cols = [c for c in ["churn", "customer_id"] if c in df.columns]
    X = df.drop(columns=drop_cols).copy()

    cat_cols = X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== 2. EVALUACIÓN EN CONJUNTO DE PRUEBA (20%, estratificado) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nMatriz de confusión [filas = real, columnas = predicho]:")
    print(cm)
    print("\nReporte detallado:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn (0)", "Churn (1)"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix - Logistic Regression (Test Set)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()