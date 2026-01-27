import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


CSV_PATH = "bank_statement_ml_dataset.csv"

TEXT_COL = "narration"
LABEL_PAYMENT = "payment_method"
LABEL_TXN = "transaction"


def train_and_save(df, text_col, label_col, out_path):
    X = df[text_col].astype(str).fillna("")
    y = df[label_col].astype(str).fillna("UNKNOWN")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\n==============================")
    print(f"RESULTS: {label_col}")
    print("==============================")
    print(classification_report(y_test, preds, zero_division=0))

    joblib.dump(model, out_path)
    print(f"✅ Saved model to: {out_path}")


def main():
    df = pd.read_csv(CSV_PATH)

    # Train model 1: payment method (UPI/CASH/IMPS/...)
    train_and_save(df, TEXT_COL, LABEL_PAYMENT, r"models\payment_method_model.pkl")

    # Train model 2: debit/credit (DEBIT/CREDIT)
    train_and_save(df, TEXT_COL, LABEL_TXN, r"models\transaction_type_model.pkl")

    print("\n🎉 Done! Two models created in the models folder.")


if __name__ == "__main__":
    main()
