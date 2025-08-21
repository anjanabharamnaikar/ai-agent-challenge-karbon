import pandas as pd
import camelot
import re
import numpy as np

def parse(pdf_path: str) -> pd.DataFrame:
    """
    Parses transaction data from a bank statement PDF file using camelot, cleans it,
    and returns a pandas DataFrame conforming to a strict schema.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        pd.DataFrame: A DataFrame with columns ['Date', 'Description', 'Amount', 'Balance']
                      and specified data types.

    Raises:
        IOError: If there's an error extracting tables from the PDF.
        ValueError: If no transaction tables are found or if critical columns are missing
                    after parsing and cleaning.
    """

    # --- Step 1: Extract tables from the PDF using camelot ---
    try:
        # Use 'stream' flavor for bank statements as they often have clear lines.
        # pages='all' ensures all pages are processed.
        # strip_text='\n' helps clean up newlines within cells.
        # line_scale can be adjusted for better line detection.
        tables = camelot.read_pdf(pdf_path, flavor='stream', pages='all',
                                  strip_text='\n', line_scale=40)
    except Exception as e:
        raise IOError(f"Error extracting tables with camelot: {e}")

    # --- Step 2: Identify and process the correct transaction table(s) ---
    all_transaction_dfs = []

    # Define common variations for expected column names
    column_variations = {
        'Date': ['date', 'transaction date', 'post date', 'value date', 'eff. date', 'txn date'],
        'Description': ['description', 'particulars', 'details', 'transaction details', 'narration'],
        'Debit': ['debit', 'withdrawal', 'dr', 'amount out', 'withdrawals'],
        'Credit': ['credit', 'deposit', 'cr', 'amount in', 'deposits'],
        'Balance': ['balance', 'closing balance', 'running balance', 'new balance', 'closing bal'],
        'Amount_Raw': ['amount', 'transaction amount', 'txn amount'] # For cases where amount is already combined
    }

    # Helper function to normalize column names for robust matching
    def normalize_col_name(col_name):
        # Convert to string, remove non-alphanumeric characters (except spaces), convert to lowercase, then strip spaces
        return re.sub(r'[^a-z0-9\s]', '', str(col_name).lower()).strip()

    for table in tables:
        df = table.df.copy()

        # Skip empty tables
        if df.empty:
            continue

        # Camelot often extracts the header as the first row of data.
        # Use the first row as potential header, then drop it from the data.
        # Normalize these potential header names for robust matching.
        if not df.empty:
            # Ensure the first row is treated as strings before normalization
            header_row = df.iloc[0].astype(str).apply(normalize_col_name)
            df = df[1:].reset_index(drop=True)
            
            # Assign normalized header to columns, handling potential duplicates by making them unique
            new_columns = []
            seen_cols = set()
            for col in header_row:
                original_col = col # Keep original normalized name
                if col in seen_cols:
                    i = 1
                    while f"{original_col}_{i}" in seen_cols:
                        i += 1
                    col = f"{original_col}_{i}"
                new_columns.append(col)
                seen_cols.add(col)
            df.columns = new_columns

        # Find actual column names in the current df based on variations
        found_cols_map = {} # Maps standard name -> normalized_df_column_name
        for std_name, variations in column_variations.items():
            for var in variations:
                normalized_var = normalize_col_name(var)
                if normalized_var in df.columns:
                    found_cols_map[std_name] = normalized_var
                    break # Found a match, move to next standard name

        # Check for presence of key columns to identify a transaction table
        has_date = 'Date' in found_cols_map
        has_desc = 'Description' in found_cols_map
        has_amount_or_debit_credit = ('Debit' in found_cols_map or 'Credit' in found_cols_map or 'Amount_Raw' in found_cols_map)
        has_balance = 'Balance' in found_cols_map

        # A table is considered a transaction table if it meets these criteria
        # and has a reasonable number of columns to avoid picking up small, irrelevant tables.
        # The number of columns check (len(df.columns) >= 4) is a heuristic.
        if has_date and has_desc and has_amount_or_debit_credit and has_balance and len(df.columns) >= 4:
            # Rename columns to standard names for this df
            df_renamed = df.rename(columns={v: k for k, v in found_cols_map.items()})

            # Select only the columns we care about for concatenation
            cols_to_keep = [k for k in ['Date', 'Description', 'Debit', 'Credit', 'Amount_Raw', 'Balance'] if k in df_renamed.columns]
            all_transaction_dfs.append(df_renamed[cols_to_keep])

    if not all_transaction_dfs:
        raise ValueError("No transaction tables found in the PDF that match the expected structure.")

    # Concatenate all identified transaction DataFrames
    df_combined = pd.concat(all_transaction_dfs, ignore_index=True)

    # --- Step 3: Clean and convert numeric data, then create 'Amount' column ---

    # Helper function to clean and convert numeric columns
    def clean_numeric_series(series):
        # Convert to string, handle NaNs/empty strings
        s = series.astype(str).str.strip()
        # Remove currency symbols, commas, spaces, and other non-numeric characters
        s = s.str.replace(r'[$,€£¥₹\s]', '', regex=True)
        # Handle parentheses for negative numbers (e.g., (100.00) -> -100.00)
        s = s.str.replace(r'^\((.*)\)$', r'-\1', regex=True)
        # Convert to numeric, coercing errors to NaN
        return pd.to_numeric(s, errors='coerce')

    # Apply numeric cleaning to relevant columns
    for col_name in ['Debit', 'Credit', 'Amount_Raw', 'Balance']:
        if col_name in df_combined.columns:
            df_combined[col_name] = clean_numeric_series(df_combined[col_name])
            # Fill NaNs with 0 for numeric calculations, especially for Debit/Credit
            # Balance NaNs will be handled later if the column is missing entirely
            if col_name in ['Debit', 'Credit', 'Amount_Raw']:
                df_combined[col_name].fillna(0.0, inplace=True)

    # Create the final 'Amount' column based on available data
    if 'Debit' in df_combined.columns and 'Credit' in df_combined.columns:
        # Credits are positive, Debits are negative
        df_combined['Amount'] = df_combined['Credit'] - df_combined['Debit']
    elif 'Amount_Raw' in df_combined.columns:
        # If only a raw amount column exists, use it directly.
        # Assume positive for deposits, negative for withdrawals if signs are already present.
        df_combined['Amount'] = df_combined['Amount_Raw']
    else:
        raise ValueError("Could not find 'Debit'/'Credit' or 'Amount_Raw' columns to create the 'Amount' column.")

    # --- Step 4: Clean and format all columns to precisely match the required schema ---
    # Build a temporary DataFrame to ensure consistent row count and index alignment
    temp_df = pd.DataFrame(index=df_combined.index)

    # Date Column: Convert to datetime, coercing errors, using specified format.
    if 'Date' in df_combined.columns:
        # Strictly use the specified format '%d-%m-%Y' for parsing
        temp_df['Date'] = pd.to_datetime(df_combined['Date'], format='%d-%m-%Y', errors='coerce')
    else:
        raise ValueError("Required 'Date' column not found after processing.")

    # Description Column: Convert to string, strip whitespace, handle empty/missing values.
    if 'Description' in df_combined.columns:
        temp_df['Description'] = df_combined['Description'].astype(str).str.strip()
        temp_df['Description'].replace('', pd.NA, inplace=True)
        temp_df['Description'].fillna('No Description', inplace=True)
    else:
        temp_df['Description'] = 'No Description' # Default if column is missing

    # Amount Column (already created)
    temp_df['Amount'] = df_combined['Amount']

    # Balance Column: Use cleaned balance, fill with NaN if missing.
    if 'Balance' in df_combined.columns:
        temp_df['Balance'] = df_combined['Balance']
    else:
        temp_df['Balance'] = np.nan # Fill with NaN if balance column was not found

    # Drop rows where 'Date' or 'Amount' are invalid (critical for transaction data)
    temp_df.dropna(subset=['Date', 'Amount'], inplace=True)

    # Ensure all required columns are present and in the exact order
    required_columns = ['Date', 'Description', 'Amount', 'Balance']
    if not all(col in temp_df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in temp_df.columns]
        raise ValueError(f"Final DataFrame is missing required columns: {missing_cols}")

    final_df = temp_df[required_columns].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Enforce strict data types as per schema
    final_df['Date'] = pd.to_datetime(final_df['Date']) # datetime64[ns]
    final_df['Description'] = final_df['Description'].astype(str) # object
    final_df['Amount'] = final_df['Amount'].astype(float) # float64
    final_df['Balance'] = final_df['Balance'].astype(float) # float64

    # Remove any completely duplicate rows that might arise from parsing multiple pages
    final_df.drop_duplicates(inplace=True)
    final_df.reset_index(drop=True, inplace=True)

    return final_df