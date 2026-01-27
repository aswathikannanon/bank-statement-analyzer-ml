from flask import jsonify
import pandas as pd
import pdfplumber
import re
import os
from datetime import datetime
import requests
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier


from app.constants.constants import get_status_code
from app.config.config import Config
from app.ml_models.predictor import predict_payment_method, predict_transaction_type



class AnalyzeModel:
    def __init__(self, db=None):
        self.db = db

    @staticmethod
    def bank_statement_analysis(data):
        local_path = data.get("local_file_path")

        if not local_path or not os.path.exists(local_path):
            return (
                jsonify(
                    {
                        "success": 0,
                        "message": "File not found or invalid path.",
                        "status_code": get_status_code("BAD_REQUEST"),
                        "result": {},
                    }
                ),
                get_status_code("BAD_REQUEST"),
            )

        try:
            analyzer = BankStatementAnalyzer(local_path)
            response_data = analyzer.extract_transactions()
            status_code = response_data.get(
                "status_code", get_status_code("INTERNAL_SERVER_ERROR")
            )

            return (
                jsonify(response_data),
                status_code,
            )
        except Exception as e:
            return (
                jsonify(
                    {
                        "success": 0,
                        "message": "Failed to analyze bank statement due to an unexpected error.",
                        "status_code": get_status_code("INTERNAL_SERVER_ERROR"),
                        "errors": [str(e)],
                    }
                ),
                get_status_code("INTERNAL_SERVER_ERROR"),
            )


class BankStatementAnalyzer:

    def __init__(self, file_path):
        self.file_path = file_path

    @staticmethod
    def clean_column_name(col):
        if not isinstance(col, str):
            col = str(col)
        return (
            col.strip()
            .lower()
            .replace(" ", "_")
            .replace(".", "")
            .replace("/", "_")
            .replace("\\", "_")
            .replace("-", "_")
        )

    @staticmethod
    def parse_amount(val):
        if (
            not val
            or pd.isna(val)
            or str(val).strip().lower() in ["", "nan", "none", "n/a", "-"]
        ):
            return None

        if isinstance(val, (pd.Series, dict)):
            print("[parse_amount] Skipping non-primitive value: %s", type(val))
            return None

        try:
            val_str = str(val).strip()

            # Reject date-like formats (e.g. '2025-02-05', '2025/02/05', '2025-02-05 00:00:00')
            if re.match(r"\d{4}[-/]\d{2}[-/]\d{2}", val_str):
                print("[parse_amount] Rejected date-like value: '%s'", val_str)
                return None

            # Remove currency and non-numeric symbols
            val_str = (
                val_str.replace(",", "")
                .replace("₹", "")
                .replace("$", "")
                .replace("€", "")
                .replace("£", "")
            )

            # Remove 'Cr.' or 'Dr.'
            val_str = re.sub(
                r"\b(Cr\.?|Dr\.?)\b", "", val_str, flags=re.IGNORECASE
            ).strip()

            # Handle negative in parentheses e.g., "(100.00)"
            if val_str.startswith("(") and val_str.endswith(")"):
                val_str = "-" + val_str[1:-1]

            amount = float(val_str)
            if pd.isna(amount):
                return None
            return amount

        except ValueError:
            print(
                "[parse_amount] Could not parse amount: '%s'. Returning None.", val
            )
            return None
        except Exception as e:
            print("[parse_amount] Unexpected error parsing '%s'", val)
            return None

    @staticmethod
    def find_column(possible_keywords, columns):
        # Normalize columns (strip and lowercase)
        normalized_columns = [col.strip().lower() for col in columns]

        # First pass: Exact match (case-insensitive)
        for keyword in possible_keywords:
            keyword_clean = keyword.strip().lower()
            for i, col in enumerate(normalized_columns):
                if col == keyword_clean:
                    return columns[i]  # Return original column name

        # Second pass: Partial match (case-insensitive)
        for keyword in possible_keywords:
            keyword_clean = keyword.strip().lower()
            for i, col in enumerate(normalized_columns):
                if keyword_clean in col:
                    return columns[i]  # Return original column name

        return None

    @staticmethod
    def detect_header_row(df_raw, max_rows_to_check=20):
        header_keywords = [
            "date",
            "transaction_date",
            "value_date",
            "description",
            "narration",
            "remark",
            "particulars",
            "credit",
            "debit",
            "balance",
            "amount",
            "txn_type",
            "type",
            "chq_no",
            "cheque_number",
            "withdrawals",
            "deposits",
        ]

        for i in range(min(max_rows_to_check, len(df_raw))):
            row = df_raw.iloc[i]
            match_count = 0
            for cell in row:
                if isinstance(cell, str):
                    cleaned_cell = BankStatementAnalyzer.clean_column_name(cell)
                    if any(keyword in cleaned_cell for keyword in header_keywords):
                        match_count += 1
            if match_count >= 2:  # Threshold: at least 2 matching keywords
                print("Detected header row at index: %s", i)
                return i
        print("No clear header row detected, defaulting to row 0.")
        return 0  # Fallback to row 0

    def _get_statement_range_from_df(self, df):
        date_col = self.find_column(
            ["date", "txn_date", "transaction_date", "value_date"], df.columns
        )
        if date_col:
            # Attempt to parse dates robustly
            dates = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
            valid_dates = dates.dropna()
            if not valid_dates.empty:
                return {
                    "from": valid_dates.min().strftime("%Y-%m-%d"),
                    "to": valid_dates.max().strftime("%Y-%m-%d"),
                }

        print("Could not determine statement date range.")
        return {}

    def extract_transactions(self):
        file_extension = os.path.splitext(self.file_path)[1].lower()

        if file_extension in (".csv", ".xlsx", ".xls"):
            return self._process_excel_csv()
        elif file_extension == ".pdf":
            return self._process_pdf_transactions()
        else:
            print("Unsupported file type: %s", file_extension)
            return {
                "success": 0,
                "status_code": get_status_code("BAD_REQUEST"),
                "message": "Unsupported file type",
                "result": {},
            }

    def _process_excel_csv(self):
        try:
            if self.file_path.endswith(".csv"):
                raw_df = pd.read_csv(self.file_path, header=None, dtype=str)
            else:
                raw_df = pd.read_excel(self.file_path, header=None, dtype=str)

            header_row_index = self.detect_header_row(raw_df)

            # Read again with the detected header
            if self.file_path.endswith(".csv"):
                df = pd.read_csv(self.file_path, header=header_row_index, dtype=str)
            else:
                df = pd.read_excel(self.file_path, header=header_row_index, dtype=str)

            # Drop unnamed columns (often created by pandas for empty columns)
            df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False, na=False)]
            df.columns = [self.clean_column_name(col) for col in df.columns]
            print("Excel/CSV Normalized Columns: %s", df.columns.tolist())

            # Detect Key Columns
            transaction_date_col = self.find_column(
                ["date", "txn_date", "transaction_date", "value_date"], df.columns
            )
            credit_col = self.find_column(
                [
                    "credit",
                    "cr",
                    "credit_amount",
                    "received",
                    "deposit",
                    "cr_amount",
                    "deposits",
                ],
                df.columns,
            )
            debit_col = self.find_column(
                [
                    "debit",
                    "dr",
                    "debit_amount",
                    "withdraw",
                    "paid",
                    "dr_amount",
                    "withdrawals",
                ],
                df.columns,
            )
            amount_col = self.find_column(
                ["amount", "transaction_amount", "value"],
                [col for col in df.columns if "date" not in col.lower()],
            )
            # Fallback if credit/debit not explicit
            narration_col = self.find_column(
                [
                    "narration",
                    "description",
                    "remark",
                    "details",
                    "transaction_details",
                ],
                df.columns,
            )
            balance_col = self.find_column(
                ["balance", "closing_balance", "available_balance", "current_balance"],
                df.columns,
            )
            account_col = self.find_column(
                ["account", "acc_no", "account_number"], df.columns
            )

            required_cols = [transaction_date_col, narration_col]
            if not all(required_cols) or not (credit_col or debit_col or amount_col):
                print(
                    "Missing critical columns for transaction parsing in %s. Date: %s, Narration: %s, Amount: %s/%s/%s",
                    self.file_path,
                    transaction_date_col,
                    narration_col,
                    credit_col,
                    debit_col,
                    amount_col,
                )
                return {
                    "success": 0,
                    "status_code": get_status_code("BAD_REQUEST"),
                    "message": "Missing critical columns (Date, Narration, and at least one of Credit/Debit/Amount).",
                    "result": {},
                }

            transactions = []
            verification_tasks = []
            txn_peer_map = []  
            
            for index, row in df.iterrows():
                try:
                    amount = None
                    txn_type = None

                    credit = self.parse_amount(row.get(credit_col))
                    debit = self.parse_amount(row.get(debit_col))

                    general_amount = (
                        self.parse_amount(row.get(amount_col)) if amount_col else None
                    )

                    if credit is not None and credit > 0:
                        amount = credit
                        txn_type = "CREDIT"
                    elif debit is not None and debit > 0:
                        amount = debit
                        txn_type = "DEBIT"
                    elif general_amount is not None:
                        amount = general_amount
                        txn_type = "CREDIT" if amount >= 0 else "DEBIT"
                        amount = abs(amount)

                    if (
                        amount is None
                        and credit is None
                        and debit is None
                        and general_amount is None
                    ):
                        print("Skipping row %s: No valid amount found.", index)
                        continue

                    narration = (
                        str(row.get(narration_col)).strip()
                        if pd.notna(row.get(narration_col))
                        else ""
                    )

                    if not narration and amount is None:
                        print(
                            "Skipping row %s: No narration or amount present.", index
                        )
                        continue

                    transaction_date_str = (
                        str(row.get(transaction_date_col)).strip()
                        if pd.notna(row.get(transaction_date_col))
                        else None
                    )
                    parsed_date = parsed_date = self.normalize_date(
                        transaction_date_str, index
                    )

                    balance = self.parse_amount(row.get(balance_col))
                    account = (
                        str(row.get(account_col)).strip()
                        if account_col and pd.notna(row.get(account_col))
                        else None
                    )

                    narration_details = self.analyze_narration_details(narration)
                    ml_payment = predict_payment_method(narration)
                    ml_txn_type = predict_transaction_type(narration)


                    # Attempt verification only if we have both account and IFSC
                    peer_account = narration_details.get("receiver_details", {}).get(
                        "account"
                    )
                    peer_ifsc = narration_details.get("bank_peer")
                    print("Account Details ---> %s %s", peer_account, peer_ifsc)

                    txn_obj = {
                        "ml_payment_method": ml_payment,
                        "ml_transaction_type": ml_txn_type,

                        "transaction_date": parsed_date,
                        "transaction_type": txn_type,
                        "amount": amount,
                        "narration": narration,
                        "balance": balance,
                        "account": account,
                        **narration_details,
                    }
                    transactions.append(txn_obj)

                except Exception as inner_err:
                    print("Skipping row %s due to parsing error", index)

            meta_info = self._extract_metadata_from_df(raw_df)

            for txn in transactions:
                txn["confidence_score"] = self.calculate_confidence_score(txn)

            if transactions:
                overall_confidence = round(
                    sum(txn["confidence_score"] for txn in transactions)
                    / len(transactions),
                    2,
                )
            else:
                overall_confidence = 0.0

            return {
                "success": 1,
                "status_code": get_status_code("SUCCESS"),
                "message": f"{len(transactions)} transactions parsed from Excel/CSV",
                "result": {
                    "account_info": meta_info,
                    "transactions": transactions,
                    "confidence_summary": {
                        "overall_score": overall_confidence,
                        "total_transactions": len(transactions),
                        "high_confidence_txns": sum(
                            1 for txn in transactions if txn["confidence_score"] >= 0.85
                        ),
                    },
                    "merchant_insights": TransactionPatternTrainer().analyze(
                        transactions
                    ),
                },
            }

        except Exception as e:
            print(
                "Failed to analyze Excel/CSV bank statement: %s", self.file_path
            )
            return {
                "success": 0,
                "status_code": get_status_code("INTERNAL_SERVER_ERROR"),
                "message": "Failed to analyze Excel/CSV bank statement",
                "result": {"error": str(e)},
            }

    def _process_pdf_transactions(self):
        try:
            transactions = []
            all_text = ""
            tables_df_list = []

            with pdfplumber.open(self.file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = (
                        page.extract_text(x_tolerance=1) or ""
                    )  # x_tolerance helps with column alignment
                    all_text += text + "\n"

                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if (
                            table and len(table) > 1
                        ):  # Ensure there's a header and at least one row
                            try:
                                df = pd.DataFrame(table[1:], columns=table[0])
                                df.columns = [
                                    self.clean_column_name(col) for col in df.columns
                                ]
                                tables_df_list.append(df)
                                print(
                                    "Page %s, Table %s: Successfully extracted with columns: %s",
                                    page_num + 1,
                                    table_idx + 1,
                                    df.columns.tolist(),
                                )
                            except Exception as df_create_err:
                                print(
                                    "Could not create DataFrame from PDF table on page %s, table %s: %s",
                                    page_num + 1,
                                    table_idx + 1,
                                    df_create_err,
                                )

            if not tables_df_list:
                print(
                    "No tables found or extracted successfully from PDF: %s",
                    self.file_path,
                )
                meta_info = self._extract_metadata_from_text(all_text)
                return {
                    "success": 0,
                    "status_code": get_status_code("BAD_REQUEST"),
                    "message": "No structured transaction tables could be extracted from the PDF.",
                    "result": {"account_info": meta_info, "transactions": []},
                }

            for df in tables_df_list:
                date_col = self.find_column(
                    ["date", "txn_date", "transaction_date", "value_date"], df.columns
                )
                credit_col = self.find_column(
                    [
                        "credit",
                        "cr",
                        "credit_amount",
                        "received",
                        "deposit",
                        "cr_amount",
                        "deposits",
                    ],
                    df.columns,
                )
                debit_col = self.find_column(
                    [
                        "debit",
                        "dr",
                        "debit_amount",
                        "withdraw",
                        "paid",
                        "dr_amount",
                        "withdrawals",
                    ],
                    df.columns,
                )
                amount_col = self.find_column(
                    ["amount", "transaction_amount", "value"], df.columns
                )
                narration_col = self.find_column(
                    [
                        "narration",
                        "description",
                        "details",
                        "remark",
                        "particulars",
                        "transaction_details",
                    ],
                    df.columns,
                )
                balance_col = self.find_column(
                    [
                        "balance",
                        "closing_balance",
                        "available_balance",
                        "current_balance",
                    ],
                    df.columns,
                )
                account_col = self.find_column(
                    ["account", "acc_no", "account_number"], df.columns
                )

                required_cols_pdf = [date_col, narration_col]
                if not all(required_cols_pdf) and not (
                    credit_col or debit_col or amount_col
                ):
                    print(
                        f"Skipping a PDF table due to missing critical columns. Date: {date_col}, Narration: {narration_col}, Amount: {credit_col}/{debit_col}/{amount_col}"
                    )
                    continue  # Skip this specific table if it lacks essential columns

                for _, row in df.iterrows():
                    try:
                        amount = None
                        txn_type = None

                        credit = self.parse_amount(row.get(credit_col))
                        debit = self.parse_amount(row.get(debit_col))
                        general_amount = self.parse_amount(row.get(amount_col))

                        if credit is not None and credit > 0:
                            amount = credit
                            txn_type = "CREDIT"
                        elif debit is not None and debit > 0:
                            amount = debit
                            txn_type = "DEBIT"
                        elif general_amount is not None:
                            amount = general_amount
                            txn_type = "CREDIT" if amount >= 0 else "DEBIT"
                            amount = abs(amount)  # Ensure amount is positive

                        if (
                            amount is None
                            and credit is None
                            and debit is None
                            and general_amount is None
                        ):
                            print(
                                "Skipping row: No amount information found in PDF table."
                            )
                            continue

                        narration = (
                            str(row.get(narration_col)).strip()
                            if pd.notna(row.get(narration_col))
                            else ""
                        )
                        if not narration and amount is None:
                            print(
                                "Skipping row: No narration or amount present in PDF table."
                            )
                            continue

                        transaction_date_str = (
                            str(row.get(date_col)).strip()
                            if pd.notna(row.get(date_col))
                            else None
                        )
                        parsed_date = None
                        if transaction_date_str:
                            try:
                                parsed_date = pd.to_datetime(
                                    transaction_date_str, errors="coerce", dayfirst=True
                                )
                                if pd.isna(parsed_date):
                                    print(
                                        "Failed to parse date '%s' from PDF. Keeping original string.",
                                        transaction_date_str,
                                    )
                                    parsed_date = transaction_date_str
                                else:
                                    parsed_date = parsed_date.strftime("%Y-%m-%d")
                            except Exception as e:
                                print(
                                    "Error converting date '%s' from PDF: %s. Keeping original string.",
                                    transaction_date_str,
                                    e,
                                )
                                parsed_date = transaction_date_str

                        balance = self.parse_amount(row.get(balance_col))
                        account = (
                            str(row.get(account_col)).strip()
                            if account_col and pd.notna(row.get(account_col))
                            else None
                        )

                        narration_details = self.analyze_narration_details(narration)
                        ml_payment = predict_payment_method(narration)
                        ml_txn_type = predict_transaction_type(narration)

                        txn_obj = {
                            "transaction_date": parsed_date,
                            "transaction_type": txn_type,
                            "amount": amount,
                            "narration": narration,
                            "ml_payment_method": ml_payment,
                            "ml_transaction_type": ml_txn_type,

                            "balance": balance,
                            "account": account,
                            **narration_details,  # Merged analysis results
                        }
                        transactions.append(txn_obj)

                    except Exception as row_err:
                        print(
                            "[PDF Table Row Skip] Error processing row: %s", row_err
                        )

            # Account metadata from full text (for more robust PDF metadata extraction)
            meta_info = self._extract_metadata_from_text(all_text)

            return {
                "success": 1,
                "status_code": get_status_code("SUCCESS"),
                "message": f"{len(transactions)} transactions parsed from PDF",
                "result": {
                    "account_info": meta_info,
                    "transactions": transactions,
                    "merchant_insights": TransactionPatternTrainer().analyze(
                        transactions
                    ),
                },
            }

        except Exception as e:
            print("Failed to analyze PDF bank statement: %s", self.file_path)
            return {
                "success": 0,
                "status_code": get_status_code("INTERNAL_SERVER_ERROR"),
                "message": "Failed to analyze PDF bank statement",
                "result": {"error": str(e)},
            }

    def normalize_date(self, date_input, row_index=None):
        """
        Normalize various date formats to 'YYYY-MM-DD'. Handles:
        - string dates in various formats
        - datetime-like objects
        - fallback to pandas only if needed
        """

        if not date_input:
            return None

        # ✅ Step 1: Already a datetime object (from Excel engine)
        if isinstance(date_input, (datetime, pd.Timestamp)):
            return date_input.strftime("%Y-%m-%d")

        # ✅ Step 2: Looks like a datetime string (e.g., '2025-02-04 00:00:00')
        if isinstance(date_input, str):
            date_input = date_input.strip()

            # Special case: looks like full datetime (YYYY-MM-DD HH:MM:SS)
            if re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", date_input):
                try:
                    parsed = datetime.strptime(date_input, "%Y-%m-%d %H:%M:%S")
                    return parsed.strftime("%Y-%m-%d")
                except ValueError:
                    pass

            # ✅ Step 3: Try known formats
            possible_formats = [
                "%d-%m-%Y",
                "%d/%m/%Y",
                "%d-%b-%y",  # 01-Feb-25
                "%d-%b-%Y",
                "%d - %b - %Y",  # 01 - Feb - 2025
                "%Y-%m-%d",
            ]

            for fmt in possible_formats:
                try:
                    parsed = datetime.strptime(date_input, fmt)
                    print(
                        "[Parsed] Row %s: '%s' → %s using %s",
                        row_index,
                        date_input,
                        parsed.strftime("%Y-%m-%d"),
                        fmt,
                    )
                    return parsed.strftime("%Y-%m-%d")
                except ValueError:
                    continue

        # ✅ Step 4: Fallback to pandas only if nothing else worked
        try:
            parsed = pd.to_datetime(date_input, errors="coerce", dayfirst=False)
            if pd.notna(parsed):
                print(
                    "[Pandas Parsed] Row %s: '%s' → %s",
                    row_index,
                    date_input,
                    parsed.strftime("%Y-%m-%d"),
                )
                return parsed.strftime("%Y-%m-%d")
        except Exception as e:
            print(
                "[Fallback Error] Row %s: '%s' — %s", row_index, date_input, e
            )

        print("[Failed to Parse] Row %s: '%s'", row_index, date_input)
        return date_input

    def _extract_metadata_from_df(self, raw_df, max_lines=30):
        try:
            lines = (
                raw_df.iloc[:max_lines].fillna("").astype(str).values.flatten().tolist()
            )
            text_blob = " ".join([line.strip() for line in lines if line.strip()])

            metadata = self._extract_metadata_from_text(text_blob)

            # Re-read df with detected header to get accurate date column for range
            try:
                header_row_index = self.detect_header_row(raw_df)
                if self.file_path.endswith(".csv"):
                    df_for_dates = pd.read_csv(
                        self.file_path, header=header_row_index, dtype=str
                    )
                else:
                    df_for_dates = pd.read_excel(
                        self.file_path, header=header_row_index, dtype=str
                    )
                df_for_dates.columns = [
                    self.clean_column_name(col) for col in df_for_dates.columns
                ]
                metadata["statement_period"] = self._get_statement_range_from_df(
                    df_for_dates
                )
            except Exception as e:
                print(
                    "Could not determine statement range from excel for metadata: %s", e
                )
                metadata["statement_period"] = {}

            print("Extracted Metadata: %s", metadata)
            return metadata

        except Exception as e:
            print("[Metadata Extraction Error - Excel/CSV]: %s", e)
            return {}

    def _extract_metadata_from_text(self, text_blob):
        metadata = {
            "account_number": None,
            "account_holder": None,
            "bank_name": None,
            "branch": None,
            "ifsc_code": None,
            "phone": None,
            "email": None,
            "statement_period": None,
        }

        # More robust patterns for key fields
        patterns = {
            "account_number": [
                r"(?:account|a/c|acct)\s*(?:no|num|number)?\s*[:\.]?\s*(\d{9,18})\b",  # General account numbers
                r"\b(\d{3,5}(?:-\d{2,5}){2,})\b",  # Some specific formatted account numbers
                r"\b(?:[Ii]nd[Oo]\s*)?(\d{11})\b",  # For Indian Bank accounts if specific format
            ],
            "account_holder": [
                r"(?:account\s*name|account\s*holder|customer\s*name|name)\s*:?\s*([A-Z][A-Z\s\.&,']+)(?=\s*(?:account|bank|address|statement))",
                r"(?:^|\n)\s*([A-Z][A-Z\s\.&,']+)\s+(?:A/C|Account|No)\s*:",  # Name followed by A/C
                r"(\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b(?=\s*\d{6,})",  # Name before an account number
            ],
            "bank_name": [
                r"(?:bank\s*name|issued\s*by|bank)\s*:?\s*([A-Z][A-Z\s,.]+)\b",
                r"\b(STATE BANK OF INDIA|HDFC BANK|ICICI BANK|AXIS BANK|PUNJAB NATIONAL BANK|YES BANK|KOTAK MAHINDRA BANK|UNION BANK OF INDIA|CANARA BANK|INDIAN BANK|INDUSIND BANK|FEDERAL BANK|RBL BANK|BANDHAN BANK|IDFC FIRST BANK)\b",
                r"BANK NAME\s*:\s*([A-Z\s&.]+)",
            ],
            "branch": [
                r"(?:branch\s*name|branch)\s*:?\s*([A-Z][A-Z\s,.-]+)\b",
                r"BRANCH\s*:\s*([A-Z\s&.]+)",
            ],
            "ifsc_code": [
                r"\b([A-Z]{4}0[A-Z0-9]{6})\b",  # Standard IFSC code pattern
                r"(?:IFSC\s*Code|IFSC)\s*[:\.]?\s*([A-Z]{4}0[A-Z0-9]{6})\b",
            ],
            "phone": [
                r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b",
                r"(?:tel|phone|mobile|ph\.?)\s*[:\.]?\s*(\+?\d[\d\s-]{7,}\d)\b",
            ],
            "email": [
                r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}\b",
            ],
        }

        for field, regex_list in patterns.items():
            for regex in regex_list:
                match = re.search(regex, text_blob, re.IGNORECASE)
                if match:
                    metadata[field] = match.group(1).strip()
                    break

        date_patterns = [
            r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b",  # DD/MM/YYYY or DD-MM-YYYY
            r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})\b",  # DD Mon YYYY
            r"\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b",  # YYYY-MM-DD or YYYY/MM/DD
        ]

        all_found_dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text_blob, re.IGNORECASE)
            for d_str in matches:
                try:
                    # Attempt to parse, coerce errors
                    parsed_d = pd.to_datetime(d_str, errors="coerce", dayfirst=True)
                    if pd.notna(parsed_d):
                        all_found_dates.append(parsed_d)
                except Exception:
                    pass

        if len(all_found_dates) >= 2:
            min_date = min(all_found_dates).strftime("%Y-%m-%d")
            max_date = max(all_found_dates).strftime("%Y-%m-%d")
            metadata["statement_period"] = {"from": min_date, "to": max_date}
        elif len(all_found_dates) == 1:
            metadata["statement_period"] = {
                "date": all_found_dates[0].strftime("%Y-%m-%d")
            }
        else:
            metadata["statement_period"] = {}

        print("Extracted Metadata: %s", metadata)
        return metadata

    @staticmethod
    def analyze_narration_details(narration):
        result = {
            "payment_method": None,
            "upi_id": None,
            "transaction_reference": None,
            "receiver_details": {
                "name": None,
                "account": None,
                "vpa": None,
            },  # Expanded receiver details
            "bank_peer": None,
            "merchant": None,
            "category": [],
            "remarks": [],
            "payment_gateway": None,  # e.g., PAYTM, RAZORPAY
        }

        if not narration:
            return result

        narration_upper = narration.upper()

        upi_structured_match = re.search(
            r"UPI\/(?P<upi_id>[^\/]+)\/(?P<remark>[^\/]+)\/(?P<bank>[^\/]+)\/(?P<txn_id>[^\s\/]+)",
            narration_upper,
        )
        if upi_structured_match:
            result["payment_method"] = "UPI"
            result["upi_id"] = upi_structured_match.group("upi_id").strip()
            result["transaction_reference"] = upi_structured_match.group(
                "txn_id"
            ).strip()
            result["bank_peer"] = upi_structured_match.group("bank").strip()
            result["remarks"].append(upi_structured_match.group("remark").strip())
            return result

        vsi_pattern = re.search(
            r"VSI\/(?P<merchant>[^\/]+)\/(?P<datetime>[^\/]+)\/(?P<txn_id>[^\s\/]+)",
            narration_upper,
        )
        if vsi_pattern:
            result["payment_method"] = "CARD"
            result["merchant"] = vsi_pattern.group("merchant").strip()
            result["transaction_reference"] = vsi_pattern.group("txn_id").strip()
            return result

        imps_transfer_match = re.search(
            r"IMPS/(\d{10,})/([^/]+)/([^/]+)", narration_upper
        )
        if imps_transfer_match:
            result["payment_method"] = "IMPS"
            result["transaction_reference"] = imps_transfer_match.group(1).strip()
            result["receiver_details"]["name"] = imps_transfer_match.group(2).strip()
            result["bank_peer"] = imps_transfer_match.group(3).strip()
            result["remarks"].append("IMPS TRANSFER")
            return result

        payment_methods_keywords = {
            "UPI": ["UPI", "IMPS/P2M", "PHONEPE", "GPAY", "PAYTM"],
            "IMPS": ["IMPS", "IMPS/P2A"],
            "NEFT": ["NEFT"],
            "RTGS": ["RTGS"],
            "BBPS": ["BBPS"],
            "CARD": ["CARD", "DEBIT CARD", "CREDIT CARD", "POS", "VPA/MMT", "VPA/MMS"],
            "CASH": ["CASH DEP", "CASH WDL"],
            "CHEQUE": ["CHQ", "CHEQUE", "CQ", "CLR"],
            "DIVIDEND": ["DIVIDEND", "DIV"],
            "INTEREST": ["INT PAID", "INT CR"],
            "ECS": ["ECS"],
            "SALARY": ["SALARY"],
            "BILL PAY": ["BILLPAY"],
            "ATM": ["ATM"],
        }
        for method, keywords in payment_methods_keywords.items():
            if any(kw in narration_upper for kw in keywords):
                result["payment_method"] = method
                break

        if not result["upi_id"]:
            upi_id_match = re.search(
                r"[a-z0-9.\-_]+@[a-z]{2,}", narration_upper, re.IGNORECASE
            )
            if upi_id_match:
                result["upi_id"] = upi_id_match.group().strip()
                result["receiver_details"]["vpa"] = result["upi_id"]

        if not result["transaction_reference"]:
            txn_ref_patterns = [
                r"\b(?:RRN|REF|TRF|TXN|UTR|UTR NO|NFS|CMS|ID)\s*[:\.]?\s*([A-Z0-9]{10,25})\b",
                r"\b(YBL|AXI|ICI|KOT|PNB|PYTM|PTM|HDFC|ICICI|YES|SBI)[a-zA-Z0-9]{6,25}\b",
                r"\b(?:\d{10,})\b",
            ]
            for pattern in txn_ref_patterns:
                match = re.search(pattern, narration_upper)
                if match:
                    try:
                        result["transaction_reference"] = match.group(1).strip()
                    except IndexError:
                        result["transaction_reference"] = match.group().strip()
                    break

        receiver_patterns = [
            r"(?:TO|FROM|BY)\s+([A-Z0-9\s.&,-_']{3,}(?:\s(?:A/C|ACC|AC|ACCOUNT|NO)\s*\d+)?)\b",  # Name or Name + A/C
            r"(?:TRANSFER TO|PAYMENT TO)\s+([A-Z\s.&,-_']{3,})",  # Transfer/Payment To Name
            r"CR BY\s+([A-Z\s.&,-_']{3,})",  # Credited by Name
        ]
        for pattern in receiver_patterns:
            match = re.search(pattern, narration_upper)
            if match:
                potential_receiver = match.group(1).strip()
                if re.search(r"\d{6,}", potential_receiver) and not re.search(
                    r"[A-Z]{3,}", potential_receiver
                ):
                    result["receiver_details"]["account"] = potential_receiver
                else:
                    result["receiver_details"]["name"] = potential_receiver
                break

        bank_keywords = [
            "STATE BANK OF INDIA",
            "HDFC BANK",
            "ICICI BANK",
            "AXIS BANK",
            "YES BANK",
            "KOTAK MAHINDRA BANK",
            "PUNJAB NATIONAL BANK",
            "UNION BANK OF INDIA",
            "CANARA BANK",
            "INDIAN BANK",
            "INDUSIND BANK",
            "FEDERAL BANK",
            "RBL BANK",
            "BANDHAN BANK",
            "IDFC FIRST BANK",
            "BANK OF BARODA",
            "UCO BANK",
            "CENTRAL BANK OF INDIA",
            "SBI",
            "HDFC",
            "ICICI",
            "AXIS",
            "KOTAK",
            "PNB",
            "UNION",
            "CANARA",
            "INDUSIND",
            "BOB",
            "UBI",
            "IOB",
            "BOI",
            "CORP",
        ]
        for bank in bank_keywords:
            if bank in narration_upper:
                result["bank_peer"] = bank
                break

        merchants_and_categories = {
            "AMAZON": {"merchant": "AMAZON", "category": "E-COMMERCE"},
            "ZOMATO": {"merchant": "ZOMATO", "category": "FOOD_DELIVERY"},
            "SWIGGY": {"merchant": "SWIGGY", "category": "FOOD_DELIVERY"},
            "GOOGLE PAY": {
                "merchant": "GOOGLE PAY",
                "category": "PAYMENT_APP",
                "payment_gateway": "GOOGLE",
            },
            "PHONEPE": {
                "merchant": "PHONEPE",
                "category": "PAYMENT_APP",
                "payment_gateway": "PHONEPE",
            },
            "PAYTM": {
                "merchant": "PAYTM",
                "category": "PAYMENT_APP",
                "payment_gateway": "PAYTM",
            },
            "RELIANCE": {"merchant": "RELIANCE", "category": "RETAIL"},
            "VODAFONE": {"merchant": "VODAFONE", "category": "TELECOM_BILL"},
            "AIRTEL": {"merchant": "AIRTEL", "category": "TELECOM_BILL"},
            "JIO": {"merchant": "JIO", "category": "TELECOM_BILL"},
            "IRCTC": {"merchant": "IRCTC", "category": "TRAVEL"},
            "UBER": {"merchant": "UBER", "category": "TRANSPORT"},
            "OLA": {"merchant": "OLA", "category": "TRANSPORT"},
            "NETFLIX": {"merchant": "NETFLIX", "category": "SUBSCRIPTION"},
            "SPOTIFY": {"merchant": "SPOTIFY", "category": "SUBSCRIPTION"},
            "CRED": {
                "merchant": "CRED",
                "category": "LOAN_REPAYMENT",
                "payment_gateway": "CRED",
            },
            "ELECTRICITY": {"category": "UTILITY_BILL"},
            "WATER": {"category": "UTILITY_BILL"},
            "GAS": {"category": "UTILITY_BILL"},
            "LOAN EMI": {"category": "LOAN_REPAYMENT"},
            "RENT": {"category": "HOUSING"},
            "SALARY": {"category": "INCOME"},
            "SCHOOL FEES": {"category": "EDUCATION"},
            "INSURANCE": {"category": "INSURANCE"},
            "INVESTMENT": {"category": "INVESTMENT"},
            "SIP": {"category": "INVESTMENT"},
            "MUTUAL FUND": {"category": "INVESTMENT"},
            "FOOD": {"category": "FOOD_EXPENSE"},
            "MEDICAL": {"category": "HEALTH_EXPENSE"},
            "PHARMACY": {"category": "HEALTH_EXPENSE"},
            "CHEMIST": {"category": "HEALTH_EXPENSE"},
            "ECOM": {"category": "E-COMMERCE"},  # Generic e-commerce
            "GROCERY": {"category": "GROCERIES"},
            "FUEL": {"category": "TRANSPORT_FUEL"},
            "TAX": {"category": "TAXES"},
            "LOAN DISB": {"category": "LOAN_DISBURSEMENT"},
        }

        for keyword, details in merchants_and_categories.items():
            if keyword in narration_upper:
                if details.get("merchant") and not result["merchant"]:
                    result["merchant"] = details["merchant"]
                if (
                    details.get("category")
                    and details["category"] not in result["category"]
                ):
                    result["category"].append(details["category"])
                if details.get("payment_gateway") and not result["payment_gateway"]:
                    result["payment_gateway"] = details["payment_gateway"]

        if "REFUND" in narration_upper and "REFUND" not in result["remarks"]:
            result["remarks"].append("REFUND")
        if "TRANSFER" in narration_upper and "TRANSFER" not in result["remarks"]:
            result["remarks"].append("TRANSFER")
        if "DEBITED" in narration_upper and "DEBITED" not in result["remarks"]:
            result["remarks"].append("DEBITED")
        if "CREDITED" in narration_upper and "CREDITED" not in result["remarks"]:
            result["remarks"].append("CREDITED")

        possible_accounts = BankStatementAnalyzer.extract_possible_account_numbers(
            narration_upper
        )
        if possible_accounts:
            result["receiver_details"]["account"] = possible_accounts[0]

        return result

    @staticmethod
    def extract_possible_account_numbers(description):
        if not description:
            return []

        numbers = set()

        # Pattern 1: Account numbers like 1234 5678 9012
        account_pattern = re.findall(
            r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4,12}\b", description
        )
        for match in account_pattern:
            numbers.add(match.replace(" ", "").replace("-", ""))

        # Pattern 2: Long sequences of 8–20 digits (no formatting)
        long_number_pattern = re.findall(r"\b\d{8,20}\b", description)
        for match in long_number_pattern:
            if len(match) <= 20:
                numbers.add(match)

        # Pattern 3: UPI / REF / TXN references
        upi_ref_pattern = re.findall(
            r"(?:UPI|REF|TXN)[\s\-:]*(\d{8,16})", description, re.IGNORECASE
        )
        numbers.update(upi_ref_pattern)

        # Pattern 4: NEFT / RTGS / IMPS references
        transfer_ref_pattern = re.findall(
            r"(?:NEFT|RTGS|IMPS)[\s\-:]*[A-Z]*(\d{8,16})", description, re.IGNORECASE
        )
        numbers.update(transfer_ref_pattern)

        return sorted(numbers, key=lambda x: -len(x))  # Sort by length descending

    @staticmethod
    def verify_bank_account_with_pennyless(account_number: str, ifsc_code: str) -> dict:
        name = "stco"
        mobile = "9999999999"

        url = f"{Config.INTEGRATION_URL}/bank/verify/account/pennyless"
        headers = {
            "Content-Type": "application/json",
            "Authorization": Config.INTEGRATION_AUTH,
        }

        payload = {
            "account_number": account_number,
            "ifsc_code": ifsc_code,
            "name": name,
            "mobile": mobile,
        }

        try:
            response = requests.get(url, params=payload, headers=headers)

            print(
                f"[Pennyless API] Status Code: {response.status_code}, Response: {response.text}"
            )

            response.raise_for_status()
            response_json = response.json()

            if response_json.get("success") == 1:
                result_data = response_json.get("result", {}).get("data", {})

                if result_data.get("account_exists") is True:
                    return {
                        "success": True,
                        "data": result_data,
                    }

                elif result_data.get("account_exists") is False:
                    return {
                        "success": False,
                        "error": result_data.get("message", "Account does not exist"),
                        "data": {},
                    }

                elif (
                    response_json.get("result", {}).get("message")
                    == "Invalid IFSC pattern"
                ):
                    return {
                        "success": False,
                        "error": "Invalid IFSC code pattern",
                        "data": {},
                    }

                elif (
                    response_json.get("result", {}).get("message")
                    == "Source Unavailable"
                ):
                    return {
                        "success": False,
                        "error": "Source Unavailable",
                        "data": {},
                    }

            return {
                "success": False,
                "error": response_json.get("result", {}).get(
                    "message", "Unknown error"
                ),
                "data": {},
            }

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e),
                "status_code": getattr(e.response, "status_code", None),
            }

    def calculate_confidence_score(self, txn: dict) -> float:
        score = 1.0  # Start with full score, then subtract penalties

        # 1. Transaction Date
        transaction_date = txn.get("transaction_date")
        if not transaction_date or not isinstance(transaction_date, str):
            score -= 0.25

        # 2. Amount
        amount = txn.get("amount")
        if amount is None or not isinstance(amount, (int, float)) or amount <= 0:
            score -= 0.25

        # 3. Narration
        narration = txn.get("narration")
        if not narration:
            score -= 0.15
        elif isinstance(narration, str) and len(narration.strip()) < 5:
            score -= 0.05  # weak/short narration

        # 4. Transaction Type
        if not txn.get("transaction_type"):
            score -= 0.10

        # 5. Receiver Details
        receiver = txn.get("receiver_details", {})
        if (
            not receiver.get("name")
            and not receiver.get("account")
            and not receiver.get("vpa")
        ):
            score -= 0.10

        # 6. Balance Field
        if txn.get("balance") is None:
            score -= 0.05

        # Clamp score between 0 and 1
        final_score = max(0.0, min(round(score, 2), 1.0))
        return final_score


class TransactionPatternTrainer:
    def __init__(self):
        pass

    def analyze(self, transactions: list) -> dict:
        merchants = defaultdict(list)

        for txn in transactions:
            merchant = txn.get("merchant")
            if not merchant:
                merchant = (
                    txn.get("receiver_details", {}).get("name")
                    or txn.get("receiver_details", {}).get("account")
                    or "UNKNOWN"
                )
            merchants[merchant].append(txn)

        insights = {}
        for m, txns in merchants.items():
            amounts = [
                t.get("amount")
                for t in txns
                if isinstance(t.get("amount"), (int, float))
            ]
            dates = [
                t.get("transaction_date") for t in txns if t.get("transaction_date")
            ]

            parsed_dates = []
            for d in dates:
                try:
                    pd_dt = pd.to_datetime(d, errors="coerce")
                    if pd.notna(pd_dt):
                        parsed_dates.append(pd_dt)
                except Exception:
                    continue

            count = len(txns)
            avg = float(pd.Series(amounts).mean()) if amounts else None
            median = float(pd.Series(amounts).median()) if amounts else None
            std = (
                float(pd.Series(amounts).std())
                if amounts and len(amounts) > 1
                else None
            )

            first = min(parsed_dates).strftime("%Y-%m-%d") if parsed_dates else None
            last = max(parsed_dates).strftime("%Y-%m-%d") if parsed_dates else None

            days = [p.day for p in parsed_dates]
            common_days = sorted({d for d in days if days.count(d) > 1})

            insights[m] = {
                "count": count,
                "avg_amount": round(avg, 2) if avg is not None else None,
                "median_amount": round(median, 2) if median is not None else None,
                "std_amount": round(std, 2) if std is not None else None,
                "first_seen": first,
                "last_seen": last,
                "common_days": common_days,
            }

        return insights


class EnhancedNarrationAnalyzer:
    def analyze(self, narration):
        # Use NER to extract entities
        doc = self.nlp(narration)
        entities = {"PERSON": [], "ORG": [], "MONEY": [], "DATE": [], "CARDINAL": []}

        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)

        # Use word embeddings for similarity matching
        embeddings = self.get_embeddings(narration)
        similar_categories = self.find_similar_categories(embeddings)

        return {
            "entities": entities,
            "suggested_categories": similar_categories,
            "confidence": self.calculate_semantic_confidence(embeddings),
        }


class TransactionPatternLearner:
    def learn_patterns(self, transactions):
        # Group by merchants/categories
        merchant_patterns = defaultdict(list)
        for txn in transactions:
            merchant = txn.get("merchant")
            if merchant:
                merchant_patterns[merchant].append(
                    {
                        "amount_range": (txn["amount"], txn["amount"]),
                        "frequency": (
                            "monthly" if self._is_monthly(txn) else "irregular"
                        ),
                        "day_of_month": txn["transaction_date"].day,
                        "category": txn.get("category", []),
                    }
                )

        # Analyze patterns
        merchant_insights = {}
        for merchant, patterns in merchant_patterns.items():
            merchant_insights[merchant] = {
                "typical_amount_range": self._get_amount_range(patterns),
                "payment_frequency": self._get_dominant_frequency(patterns),
                "preferred_days": self._get_preferred_days(patterns),
                "common_categories": self._get_common_categories(patterns),
            }

        return merchant_insights


class BalanceValidator:
    def validate_running_balance(self, transactions):
        sorted_txns = sorted(transactions, key=lambda x: x["transaction_date"])
        running_balance = None
        anomalies = []

        for i, txn in enumerate(sorted_txns):
            if txn.get("balance") is not None:
                if running_balance is None:
                    running_balance = txn["balance"]
                else:
                    expected_balance = (
                        running_balance + txn["amount"]
                        if txn["transaction_type"] == "CREDIT"
                        else running_balance - txn["amount"]
                    )
                    if (
                        abs(expected_balance - txn["balance"]) > 0.01
                    ):  # Allow small float differences
                        anomalies.append(
                            {
                                "transaction": txn,
                                "expected_balance": expected_balance,
                                "actual_balance": txn["balance"],
                                "difference": abs(expected_balance - txn["balance"]),
                            }
                        )
                running_balance = txn["balance"]

        return anomalies


class EnhancedConfidenceScorer:
    def calculate_score(self, transaction):
        base_score = self.calculate_base_score(transaction)
        pattern_score = self.calculate_pattern_score(transaction)
        semantic_score = self.calculate_semantic_score(transaction)
        verification_score = self.calculate_verification_score(transaction)

        weights = {"base": 0.4, "pattern": 0.2, "semantic": 0.2, "verification": 0.2}

        final_score = (
            base_score * weights["base"]
            + pattern_score * weights["pattern"]
            + semantic_score * weights["semantic"]
            + verification_score * weights["verification"]
        )

        return {
            "overall_score": round(final_score, 2),
            "component_scores": {
                "base_score": base_score,
                "pattern_score": pattern_score,
                "semantic_score": semantic_score,
                "verification_score": verification_score,
            },
            "confidence_level": self.get_confidence_level(final_score),
        }