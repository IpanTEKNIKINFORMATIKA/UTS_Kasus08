"""Skrip pelatihan, evaluasi, dan penyimpanan model Kasus 8: Studi Infertilitas."""
from pathlib import Path
import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler

from fitur import (
    FITUR_MENTAH,
    FITUR_NUMERIK,
    FITUR_ORDINAL,
    KOLOM_DIBUANG,
    TARGET_KOLOM,
    URUTAN_PENDIDIKAN,
    buat_fitur_turunan,
)

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = "/content/drive/My Drive/UTS_Kasus08/kasus08_infert.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "model_infertilitas_kasus08.joblib"


def audit_data(df: pd.DataFrame) -> dict:
    """Melakukan audit kualitas data awal (duplikat, missing value, nilai tidak wajar)."""
    return {
        "jumlah_baris": int(df.shape[0]),
        "jumlah_kolom": int(df.shape[1]),
        "duplikat_penuh": int(df.duplicated().sum()),
        "missing_total": int(df.isna().sum().sum()),
        "usia_tidak_wajar": int(((df["age"] < 15) | (df["age"] > 60)).sum()),
        "negatif_kehamilan": int(
            ((df["parity"] < 0) | (df["induced" ] < 0) | (df["spontaneous"] < 0)).sum()
        ),
    }


def bangun_pipeline(classifier) -> Pipeline:
    """Membangun pipeline lengkap dari input mentah hingga prediksi klasifikasi."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FITUR_NUMERIK,
            ),
            (
                "ord",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                categories=[URUTAN_PENDIDIKAN],
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                FITUR_ORDINAL,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("feature_engineering", FunctionTransformer(buat_fitur_turunan)),
            ("preprocessing", preprocessor),
            ("classifier", classifier),
        ]
    )


def main():
    df = pd.read_csv(DATA_PATH)
    hasil_audit = audit_data(df)
    print("=== AUDIT DATA ===")
    for k, v in hasil_audit.items():
        print(f"- {k}: {v}")

    # Buang kolom desain studi (stratum, pooled.stratum) agar tidak terjadi data leakage
    df_model = df.drop(columns=[c for c in KOLOM_DIBUANG if c in df.columns])

    X = df_model[FITUR_MENTAH].copy()
    y = df_model[TARGET_KOLOM].astype(int)

    # Pembagian train & test SEBELUM imputasi/scaling/encoding (anti-leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nData Train: {X_train.shape[0]} baris | Data Test: {X_test.shape[0]} baris")

    # Bandingkan model menggunakan Stratified 5-Fold CV pada data TRAIN saja
    kandidat = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=5, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE
        ),
        "KNN (k=7)": KNeighborsClassifier(n_neighbors=7, weights="distance"),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    print("\n=== PERBANDINGAN MODEL (5-FOLD CV PADA DATA TRAIN) ===")
    ringkasan_cv = []
    for nama, clf in kandidat.items():
        pipe = bangun_pipeline(clf)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        ringkasan_cv.append(
            {
                "model": nama,
                "cv_accuracy": round(float(scores["test_accuracy"].mean()), 4),
                "cv_precision": round(float(scores["test_precision"].mean()), 4),
                "cv_recall": round(float(scores["test_recall"].mean()), 4),
                "cv_f1": round(float(scores["test_f1"].mean()), 4),
                "cv_roc_auc": round(float(scores["test_roc_auc"].mean()), 4),
            }
        )

    df_cv = pd.DataFrame(ringkasan_cv).sort_values(by="cv_roc_auc", ascending=False)
    print(df_cv.to_string(index=False))

    nama_terbaik = df_cv.iloc[0]["model"]
    print(f"\nModel terpilih berdasarkan CV: {nama_terbaik}")

    # Latih model terbaik pada seluruh data train & evaluasi SATU KALI pada data test
    model_final = bangun_pipeline(kandidat[nama_terbaik])
    model_final.fit(X_train, y_train)

    y_pred = model_final.predict(X_test)
    y_proba = model_final.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrik_test = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_proba)), 4),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }

    print("\n=== EVALUASI AKHIR PADA DATA TEST (1 KALI) ===")
    for k, v in metrik_test.items():
        print(f"- {k}: {v}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Kontrol (0)", "Kasus (1)"]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": model_final,
        "nama_model": nama_terbaik,
        "fitur_mentah": FITUR_MENTAH,
        "opsi_education": URUTAN_PENDIDIKAN,
        "threshold": 0.5,
        "hasil_audit": hasil_audit,
        "perbandingan_cv": ringkasan_cv,
        "metrik_test": metrik_test,
        "versi_sklearn": sklearn.__version__,
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)
    print(f"Model berhasil disimpan ke: {MODEL_PATH}")


if __name__ == "__main__":
    main()
