# bank-statement-analyzer-ml
Bank statement analysis using Flask and Machine Learning
This project extends a Bank Statement Analyzer framework by integrating
Machine Learning to automatically identify:

- Transaction Type (DEBIT / CREDIT)
- Payment Method (UPI, NEFT, IMPS, CASH, etc.)

The goal of this project is to demonstrate how Machine Learning can be
applied to real-world bank statement data and integrated into a Flask-based
backend system.
# Bank Statement Analyzer with ML Extension

This project is based on the Bank Statement Analyzer framework.

## My Contribution (ML Part)

I added a Machine Learning module to:
- Predict transaction type (DEBIT / CREDIT)
- Predict payment method (UPI, NEFT, IMPS, CASH, etc.)

## Folder Structure

- backend/
  - Flask backend
  - ML prediction integrated into analyzeModel.py

- ml_training/
  - bank_statement_ml_dataset.csv
  - train_models.py
  - test_model.py
  - trained ML models

## How ML Works

Narration text is used as input.
Two models are trained using TF-IDF + RandomForest.

Models:
- payment_method_model.pkl
- transaction_type_model.pkl

## How to Run

1. Create virtual environment
python -m venv venv
venv\Scripts\activate

2. Install requirements
pip install -r requirements.txt

3. Run backend
python run.py
