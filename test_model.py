import joblib

# Load trained models
payment_model = joblib.load("models/payment_method_model.pkl")
transaction_model = joblib.load("models/transaction_type_model.pkl")

# Example narrations to test
test_narrations = [
    "UPI payment to Amazon",
    "NEFT transfer from salary account",
    "ATM cash withdrawal",
    "Cheque deposit",
    "IMPS transfer received"
]

print("\n🔍 PREDICTION RESULTS\n")

for narration in test_narrations:
    payment = payment_model.predict([narration])[0]
    transaction = transaction_model.predict([narration])[0]

    print("Narration:", narration)
    print("➡ Payment Method:", payment)
    print("➡ Transaction Type:", transaction)
    print("-" * 40)
